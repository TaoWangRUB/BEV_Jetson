//! The 4-DoF pose graph (openspec design D4): position and yaw per keyframe, roll and pitch from
//! the VIO. A node's rotation is `Rz(yaw) * R_rp`, with `R_rp` the VIO rotation with its own yaw
//! removed, so optimising yaw never disturbs gravity alignment. An edge i->j carries the relative
//! pose `T_ij = T_i^-1 T_j` as `t_ij` (in frame i) and `dyaw_ij = yaw(R_rp_i R_ij)`, the constant
//! that makes `yaw_j = yaw_i + dyaw_ij` exact.
//!
//! Residuals: `(R_i^T (p_j - p_i) - t_ij) / sigma_t` and `wrap(yaw_j - yaw_i - dyaw_ij) / sigma_yaw`.
//! Sequential edges carry a covariance that grows with travelled distance (the VIO exposes none);
//! loop edges carry the verifier's and a Huber weight applied by iteratively reweighted least
//! squares (cuVSLAM's robustifier). Solved by Levenberg-Marquardt on the sparse normal
//! equations with a block-Jacobi-preconditioned conjugate gradient; node 0 is fixed. All of it
//! deterministic.
use std::collections::BTreeMap;

use nalgebra::{Matrix3, Matrix4, Rotation3, SMatrix, Vector3, Vector4};

use crate::config::Graph as GraphCfg;
use crate::geom::{Pose, rz, wrap_angle, yaw_of};

#[derive(Debug, Clone)]
pub struct Edge {
    pub i: usize,
    pub j: usize,
    pub t_ij: Vector3<f64>,
    pub dyaw: f64,
    pub sigma_t: f64,
    pub sigma_yaw: f64,
    pub is_loop: bool,
    pub weight: f64,
}

pub struct PoseGraph4DoF {
    cfg: GraphCfg,
    pub t_ns: Vec<i64>,
    pub p: Vec<Vector3<f64>>,
    pub yaw: Vec<f64>,
    pub r_rp: Vec<Rotation3<f64>>,
    pub p_vio: Vec<Vector3<f64>>,
    pub r_vio: Vec<Rotation3<f64>>,
    pub edges: Vec<Edge>,
}

type Block = SMatrix<f64, 4, 4>;

impl PoseGraph4DoF {
    pub fn new(cfg: GraphCfg) -> Self {
        Self {
            cfg,
            t_ns: Vec::new(),
            p: Vec::new(),
            yaw: Vec::new(),
            r_rp: Vec::new(),
            p_vio: Vec::new(),
            r_vio: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.p.len()
    }

    fn rot(&self, k: usize) -> Rotation3<f64> {
        rz(self.yaw[k]) * self.r_rp[k]
    }

    pub fn pose(&self, k: usize) -> Pose {
        Pose::from_parts(
            nalgebra::Translation3::from(self.p[k]),
            nalgebra::UnitQuaternion::from_rotation_matrix(&self.rot(k)),
        )
    }

    pub fn poses(&self) -> Vec<Pose> {
        (0..self.len()).map(|k| self.pose(k)).collect()
    }

    /// Add the next keyframe with its VIO pose; a sequential edge joins it to the previous one.
    pub fn add_node(&mut self, t_ns: i64, t_vio: &Pose) -> usize {
        let r: Rotation3<f64> = t_vio.rotation.to_rotation_matrix();
        let p = t_vio.translation.vector;
        let y = yaw_of(&r);
        let k = self.len();
        self.t_ns.push(t_ns);
        self.r_vio.push(r);
        self.p_vio.push(p);
        self.r_rp.push(rz(-y) * r);
        if k == 0 {
            self.p.push(p);
            self.yaw.push(y);
        } else {
            let i = k - 1;
            // the VIO delta, placed from the previous (already corrected) node
            let r_ij = self.r_vio[i].transpose() * r;
            let t_ij = self.r_vio[i].transpose() * (p - self.p_vio[i]);
            let dyaw = yaw_of(&(self.r_rp[i] * r_ij));
            self.p.push(self.p[i] + self.rot(i) * t_ij);
            self.yaw.push(wrap_angle(self.yaw[i] + dyaw));
            let d = t_ij.norm();
            let c = &self.cfg;
            self.edges.push(Edge {
                i,
                j: k,
                t_ij,
                dyaw,
                sigma_t: c.seq_sigma_t0_m + c.seq_sigma_t_per_m * d,
                sigma_yaw: (c.seq_sigma_yaw0_deg + c.seq_sigma_yaw_per_m_deg * d).to_radians(),
                is_loop: false,
                weight: 1.0,
            });
        }
        k
    }

    /// `t_ij` = T_i^-1 T_j: the pose of j in i's frame.
    pub fn add_loop_edge(
        &mut self,
        i: usize,
        j: usize,
        t_ij: &Pose,
        sigma_t: f64,
        sigma_yaw: f64,
    ) -> usize {
        let r_ij: Rotation3<f64> = t_ij.rotation.to_rotation_matrix();
        self.edges.push(Edge {
            i,
            j,
            t_ij: t_ij.translation.vector,
            dyaw: yaw_of(&(self.r_rp[i] * r_ij)),
            sigma_t,
            sigma_yaw,
            is_loop: true,
            weight: 1.0,
        });
        self.edges.len() - 1
    }

    pub fn drop_edge(&mut self, idx: usize) {
        self.edges.remove(idx);
    }

    pub fn loop_edge_indices(&self) -> Vec<usize> {
        (0..self.edges.len())
            .filter(|&e| self.edges[e].is_loop)
            .collect()
    }

    /// Sigma-scaled, unweighted residual of one edge at the current state.
    pub fn edge_residual(&self, e: &Edge) -> Vector4<f64> {
        let ri = self.rot(e.i);
        let rt = (ri.transpose() * (self.p[e.j] - self.p[e.i]) - e.t_ij) / e.sigma_t;
        Vector4::new(
            rt.x,
            rt.y,
            rt.z,
            wrap_angle(self.yaw[e.j] - self.yaw[e.i] - e.dyaw) / e.sigma_yaw,
        )
    }

    fn cost(&self) -> f64 {
        self.edges
            .iter()
            .map(|e| {
                let r = self.edge_residual(e) * e.weight;
                r.norm_squared()
            })
            .sum()
    }

    /// Jacobian blocks of one edge's weighted residual with respect to nodes i and j (4x4 each).
    fn edge_jacobians(&self, e: &Edge) -> (Block, Block) {
        let w = e.weight;
        let (cy, sy) = (self.yaw[e.i].cos(), self.yaw[e.i].sin());
        let rzt = Matrix3::new(cy, sy, 0.0, -sy, cy, 0.0, 0.0, 0.0, 1.0); // Rz(-yaw_i)
        let drzt = Matrix3::new(-sy, cy, 0.0, -cy, -sy, 0.0, 0.0, 0.0, 0.0); // d/dyaw of Rz(-yaw_i)
        let rit = self.r_rp[e.i].transpose() * rzt;
        let d = self.p[e.j] - self.p[e.i];
        let mut jj = Block::zeros();
        let mut ji = Block::zeros();
        let bj = w * rit / e.sigma_t;
        jj.fixed_view_mut::<3, 3>(0, 0).copy_from(&bj);
        jj[(3, 3)] = w / e.sigma_yaw;
        ji.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-bj));
        let dyaw_i = w * (self.r_rp[e.i].transpose() * drzt * d) / e.sigma_t;
        ji[(0, 3)] = dyaw_i.x;
        ji[(1, 3)] = dyaw_i.y;
        ji[(2, 3)] = dyaw_i.z;
        ji[(3, 3)] = -w / e.sigma_yaw;
        (ji, jj)
    }

    /// Solve; returns the loop edges' normalised residual norms after the final solve.
    pub fn optimize(&mut self) -> Vec<f64> {
        if self.len() < 2 || self.edges.is_empty() {
            return Vec::new();
        }
        let huber = self.cfg.huber_m;
        for e in self.edges.iter_mut() {
            e.weight = 1.0;
        }
        let rounds = if self.edges.iter().any(|e| e.is_loop) {
            3
        } else {
            1
        };
        for _ in 0..rounds {
            self.lm_solve(40);
            let mut changed = false;
            let weights: Vec<f64> = self
                .edges
                .iter()
                .map(|e| {
                    if !e.is_loop {
                        return 1.0;
                    }
                    let r = self.edge_residual(e);
                    let dm = r.fixed_rows::<3>(0).norm() * e.sigma_t;
                    if dm <= huber {
                        1.0
                    } else {
                        (huber / dm).sqrt()
                    }
                })
                .collect();
            for (e, w) in self.edges.iter_mut().zip(weights) {
                if (w - e.weight).abs() > 1e-6 {
                    changed = true;
                }
                e.weight = w;
            }
            if !changed {
                break;
            }
        }
        self.edges
            .iter()
            .filter(|e| e.is_loop)
            .map(|e| self.edge_residual(e).norm())
            .collect()
    }

    /// Levenberg-Marquardt on the sparse normal equations; node 0 is fixed.
    fn lm_solve(&mut self, max_iters: usize) {
        let n = self.len();
        let mut lambda = 1e-3;
        let mut cost = self.cost();
        for _ in 0..max_iters {
            // assemble H (block sparse, upper blocks keyed (a, b) with a <= b) and g
            let mut blocks: BTreeMap<(usize, usize), Block> = BTreeMap::new();
            let mut g = vec![Vector4::<f64>::zeros(); n];
            let mut diag = vec![Block::zeros(); n];
            for e in &self.edges {
                let r = self.edge_residual(e) * e.weight;
                let (ji, jj) = self.edge_jacobians(e);
                let (a, b) = (e.i, e.j);
                diag[a] += ji.transpose() * ji;
                diag[b] += jj.transpose() * jj;
                g[a] += ji.transpose() * r;
                g[b] += jj.transpose() * r;
                let (lo, hi, jl, jh) = if a < b {
                    (a, b, ji, jj)
                } else {
                    (b, a, jj, ji)
                };
                *blocks.entry((lo, hi)).or_insert_with(Block::zeros) += jl.transpose() * jh;
            }
            let mut accepted = false;
            for _ in 0..8 {
                let damped: Vec<Block> = diag
                    .iter()
                    .map(|d| {
                        let mut m = *d;
                        for k in 0..4 {
                            m[(k, k)] += lambda * (1.0 + d[(k, k)]);
                        }
                        m
                    })
                    .collect();
                let step = pcg(n, &damped, &blocks, &g);
                let (p_old, yaw_old) = (self.p.clone(), self.yaw.clone());
                for k in 1..n {
                    self.p[k] -= step[k].fixed_rows::<3>(0).into_owned();
                    self.yaw[k] = wrap_angle(self.yaw[k] - step[k][3]);
                }
                let c = self.cost();
                if c < cost {
                    let small = step
                        .iter()
                        .skip(1)
                        .map(|s| s.norm_squared())
                        .sum::<f64>()
                        .sqrt()
                        < 1e-9;
                    cost = c;
                    lambda = (lambda / 3.0).max(1e-12);
                    accepted = true;
                    if small {
                        return;
                    }
                    break;
                }
                self.p = p_old;
                self.yaw = yaw_old;
                lambda *= 4.0;
            }
            if !accepted {
                return;
            }
        }
    }
}

/// Block-Jacobi preconditioned conjugate gradient on `H x = g` with node 0 fixed
/// (its rows and columns are dropped). `diag` holds the damped diagonal blocks, `off` the
/// upper off-diagonal blocks keyed (a, b) with a < b.
fn pcg(
    n: usize,
    diag: &[Block],
    off: &BTreeMap<(usize, usize), Block>,
    g: &[Vector4<f64>],
) -> Vec<Vector4<f64>> {
    let apply = |x: &[Vector4<f64>]| -> Vec<Vector4<f64>> {
        let mut y: Vec<Vector4<f64>> = (0..n)
            .map(|k| {
                if k == 0 {
                    Vector4::zeros()
                } else {
                    diag[k] * x[k]
                }
            })
            .collect();
        for (&(a, b), blk) in off {
            if a == 0 || b == 0 {
                continue;
            }
            y[a] += blk * x[b];
            y[b] += blk.transpose() * x[a];
        }
        y
    };
    let minv: Vec<Matrix4<f64>> = diag
        .iter()
        .map(|d| d.try_inverse().unwrap_or_else(Matrix4::identity))
        .collect();
    let mut x = vec![Vector4::<f64>::zeros(); n];
    let mut r: Vec<Vector4<f64>> = g.to_vec();
    r[0] = Vector4::zeros();
    let mut z: Vec<Vector4<f64>> = (0..n)
        .map(|k| {
            if k == 0 {
                Vector4::zeros()
            } else {
                minv[k] * r[k]
            }
        })
        .collect();
    let mut p = z.clone();
    let mut rz: f64 = (1..n).map(|k| r[k].dot(&z[k])).sum();
    let g_norm: f64 = (1..n)
        .map(|k| g[k].norm_squared())
        .sum::<f64>()
        .sqrt()
        .max(1e-300);
    for _ in 0..(4 * n + 50) {
        let ap = apply(&p);
        let pap: f64 = (1..n).map(|k| p[k].dot(&ap[k])).sum();
        if pap.abs() < 1e-300 {
            break;
        }
        let alpha = rz / pap;
        for k in 1..n {
            x[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        let r_norm: f64 = (1..n).map(|k| r[k].norm_squared()).sum::<f64>().sqrt();
        if r_norm / g_norm < 1e-12 {
            break;
        }
        for k in 1..n {
            z[k] = minv[k] * r[k];
        }
        let rz_new: f64 = (1..n).map(|k| r[k].dot(&z[k])).sum();
        let beta = rz_new / rz;
        rz = rz_new;
        for k in 1..n {
            p[k] = z[k] + beta * p[k];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};

    fn cfg() -> GraphCfg {
        GraphCfg {
            seq_sigma_t0_m: 0.02,
            seq_sigma_t_per_m: 0.05,
            seq_sigma_yaw0_deg: 0.2,
            seq_sigma_yaw_per_m_deg: 1.0,
            huber_m: 0.1,
            edge_check_sigma: 3.0,
            max_nodes: 0,
        }
    }

    fn se3(yaw: f64, p: [f64; 3]) -> Pose {
        Pose::from_parts(
            Translation3::new(p[0], p[1], p[2]),
            UnitQuaternion::from_rotation_matrix(&rz(yaw)),
        )
    }

    fn square_lap() -> Vec<Pose> {
        let (n, side) = (10, 4.0);
        let mut out = Vec::new();
        for leg in 0..4 {
            let yaw = leg as f64 * std::f64::consts::FRAC_PI_2;
            let d = [yaw.cos(), yaw.sin(), 0.0];
            let start = match leg {
                0 => [0.0, 0.0, 0.0],
                1 => [side, 0.0, 0.0],
                2 => [side, side, 0.0],
                _ => [0.0, side, 0.0],
            };
            for k in 0..n {
                let f = side * k as f64 / n as f64;
                out.push(se3(yaw, [start[0] + d[0] * f, start[1] + d[1] * f, 0.0]));
            }
        }
        out.push(se3(0.0, [0.0, 0.0, 0.0]));
        out
    }

    fn drifted(truth: &[Pose], yaw_rate: f64, scale: f64) -> Vec<Pose> {
        let mut out = vec![truth[0]];
        for k in 1..truth.len() {
            let rel = truth[k - 1].inverse() * truth[k];
            let rel = Pose::from_parts(
                Translation3::from(rel.translation.vector * scale),
                UnitQuaternion::from_rotation_matrix(
                    &(rz(yaw_rate) * rel.rotation.to_rotation_matrix()),
                ),
            );
            out.push(out[k - 1] * rel);
        }
        out
    }

    #[test]
    fn closes_injected_drift() {
        let truth = square_lap();
        let vio = drifted(&truth, 0.004, 1.05);
        let mut g = PoseGraph4DoF::new(cfg());
        for (k, t) in vio.iter().enumerate() {
            g.add_node(k as i64 * 1_000_000_000, t);
        }
        let end_err = (vio.last().unwrap().translation.vector
            - truth.last().unwrap().translation.vector)
            .norm();
        assert!(end_err > 0.3, "drift {end_err}");
        let t_last_from_first = truth.last().unwrap().inverse() * truth[0];
        g.add_loop_edge(
            vio.len() - 1,
            0,
            &t_last_from_first,
            0.05,
            1.0f64.to_radians(),
        );
        let r = g.optimize();
        assert!(r.iter().all(|&x| x < 3.0), "{r:?}");
        let p = g.poses();
        let gap = (p.last().unwrap().translation.vector - p[0].translation.vector).norm();
        assert!(gap < 0.08, "gap {gap}");
        for t in &p {
            assert!((t.rotation.to_rotation_matrix()[(2, 2)] - 1.0).abs() < 1e-9); // pure yaw
        }
        // the drift is redistributed, not teleported
        let mut worst = 0.0f64;
        for k in 1..p.len() {
            let s = (p[k].translation.vector - p[k - 1].translation.vector).norm();
            let t = (truth[k].translation.vector - truth[k - 1].translation.vector).norm();
            worst = worst.max(s - t);
        }
        assert!(worst < 0.1, "step excess {worst}");
    }
}
