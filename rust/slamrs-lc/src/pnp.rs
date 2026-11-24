//! Multi-camera PnP for the verification step (openspec design D3): RANSAC on four-point samples
//! solved by Gauss-Newton from the odometry guess, then a Huber-weighted joint refinement over
//! every correspondence. A minimal closed-form P3P is not needed here: every hypothesis the
//! backend verifies comes with a pose guess inside the tracker's reach (D1a), and a four-point
//! Gauss-Newton from that guess converges on the sample's own pose; RANSAC's consensus does the
//! rest, as it would for a closed-form solver. Seeded, so a replay is reproducible.
use nalgebra::{Matrix6, Vector2, Vector3, Vector6};

use crate::geom::{Pose, Rig, from_rotvec_t, normalised, tf};

/// One correspondence: a point in the candidate rig frame seen as a bearing in a current camera.
#[derive(Debug, Clone, Copy)]
pub struct Corr {
    pub p: Vector3<f64>,
    pub bearing: Vector3<f64>,
    pub cam: usize,
}

pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn params(t: &Pose) -> Vector6<f64> {
    let rv = t.rotation.scaled_axis();
    Vector6::new(
        rv.x,
        rv.y,
        rv.z,
        t.translation.x,
        t.translation.y,
        t.translation.z,
    )
}

fn pose(x: &Vector6<f64>) -> Pose {
    from_rotvec_t(
        &Vector3::new(x[0], x[1], x[2]),
        &Vector3::new(x[3], x[4], x[5]),
    )
}

/// Normalised-plane residual of one correspondence under `t` (= T_cur_from_cand), or None when
/// the point is behind the camera or the bearing too oblique.
fn residual(t: &Pose, rig: &Rig, c: &Corr, min_z: f64) -> Option<Vector2<f64>> {
    let pc = tf(&rig.t_cam_imu[c.cam], &tf(t, &c.p));
    if pc.z <= 1e-3 {
        return None;
    }
    let obs = normalised(&c.bearing, min_z)?;
    Some(Vector2::new(pc.x / pc.z, pc.y / pc.z) - obs)
}

const HUGE: f64 = 10.0;

fn residual_or_huge(t: &Pose, rig: &Rig, c: &Corr, min_z: f64) -> Vector2<f64> {
    residual(t, rig, c, min_z).unwrap_or_else(|| Vector2::new(HUGE, HUGE))
}

/// Gauss-Newton / Levenberg-Marquardt over the six pose parameters with a numeric Jacobian.
/// `weight` returns the IRLS weight of a residual norm (1.0 for plain least squares).
fn solve(
    t0: &Pose,
    rig: &Rig,
    corrs: &[Corr],
    min_z: f64,
    iters: usize,
    weight: impl Fn(f64) -> f64,
) -> Pose {
    let mut x = params(t0);
    let mut lambda = 1e-4;
    let cost = |x: &Vector6<f64>| -> f64 {
        let t = pose(x);
        corrs
            .iter()
            .map(|c| {
                let r = residual_or_huge(&t, rig, c, min_z);
                let n = r.norm();
                weight(n) * n * n
            })
            .sum()
    };
    let mut cur = cost(&x);
    for _ in 0..iters {
        let t = pose(&x);
        let mut h = Matrix6::<f64>::zeros();
        let mut g = Vector6::<f64>::zeros();
        let eps = 1e-6;
        for c in corrs {
            let r = residual_or_huge(&t, rig, c, min_z);
            let w = weight(r.norm());
            if w == 0.0 {
                continue;
            }
            let mut j = nalgebra::SMatrix::<f64, 2, 6>::zeros();
            for k in 0..6 {
                let mut xp = x;
                xp[k] += eps;
                let mut xm = x;
                xm[k] -= eps;
                let d = (residual_or_huge(&pose(&xp), rig, c, min_z)
                    - residual_or_huge(&pose(&xm), rig, c, min_z))
                    / (2.0 * eps);
                j.set_column(k, &d);
            }
            h += w * j.transpose() * j;
            g += w * j.transpose() * r;
        }
        let mut improved = false;
        for _ in 0..6 {
            let mut hd = h;
            for k in 0..6 {
                hd[(k, k)] += lambda * (1.0 + h[(k, k)]);
            }
            let Some(chol) = hd.cholesky() else {
                lambda *= 10.0;
                continue;
            };
            let step = chol.solve(&(-g));
            let xn = x + step;
            let cn = cost(&xn);
            if cn < cur {
                let small = step.norm() < 1e-10;
                x = xn;
                cur = cn;
                lambda = (lambda / 3.0).max(1e-12);
                improved = true;
                if small {
                    return pose(&x);
                }
                break;
            }
            lambda *= 4.0;
        }
        if !improved {
            break;
        }
    }
    pose(&x)
}

/// RANSAC over `corrs` (one camera pair) from the guess. Returns the pose and the inlier mask.
pub fn ransac(
    corrs: &[Corr],
    rig: &Rig,
    guess: &Pose,
    thresh: f64,
    iters: usize,
    seed: u64,
    min_z: f64,
) -> Option<(Pose, Vec<bool>)> {
    let usable: Vec<usize> = (0..corrs.len())
        .filter(|&i| normalised(&corrs[i].bearing, min_z).is_some())
        .collect();
    if usable.len() < 6 {
        return None;
    }
    let mut rng = XorShift::new(seed);
    let mut best: Option<(usize, Pose)> = None;
    for _ in 0..iters {
        let mut pick = [0usize; 4];
        for k in 0..4 {
            loop {
                let i = usable[rng.below(usable.len())];
                if !pick[..k].contains(&i) {
                    pick[k] = i;
                    break;
                }
            }
        }
        let sample: Vec<Corr> = pick.iter().map(|&i| corrs[i]).collect();
        let t = solve(guess, rig, &sample, min_z, 8, |_| 1.0);
        let n = usable
            .iter()
            .filter(|&&i| residual(&t, rig, &corrs[i], min_z).is_some_and(|r| r.norm() <= thresh))
            .count();
        if best.as_ref().is_none_or(|b| n > b.0) {
            best = Some((n, t));
        }
    }
    let (n, t) = best?;
    if n < 6 {
        return None;
    }
    // refine on the consensus set, then recount
    let inl: Vec<Corr> = usable
        .iter()
        .filter(|&&i| residual(&t, rig, &corrs[i], min_z).is_some_and(|r| r.norm() <= thresh))
        .map(|&i| corrs[i])
        .collect();
    let t = solve(&t, rig, &inl, min_z, 10, |_| 1.0);
    let mask: Vec<bool> = (0..corrs.len())
        .map(|i| residual(&t, rig, &corrs[i], min_z).is_some_and(|r| r.norm() <= thresh))
        .collect();
    if mask.iter().filter(|&&m| m).count() < 6 {
        return None;
    }
    Some((t, mask))
}

/// Huber-weighted joint refinement over every correspondence from `t0`.
/// Returns the pose, the inlier mask at `thresh` and every residual norm.
pub fn refine(
    t0: &Pose,
    rig: &Rig,
    corrs: &[Corr],
    thresh: f64,
    min_z: f64,
) -> (Pose, Vec<bool>, Vec<f64>) {
    let huber = move |n: f64| if n <= thresh { 1.0 } else { thresh / n };
    let t = solve(t0, rig, corrs, min_z, 30, huber);
    let err: Vec<f64> = corrs
        .iter()
        .map(|c| residual_or_huge(&t, rig, c, min_z).norm())
        .collect();
    let mask = err.iter().map(|&e| e <= thresh).collect();
    (t, mask, err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion, Vector3};
    use slam_rs::calib::Calibration;

    const CALIB: &str = include_str!("../../../config/slamrs/bev_calib_s2.json");

    #[test]
    fn ransac_recovers_a_pose_through_outliers() {
        let calibration: Calibration<f64> = Calibration::from_json_str(CALIB).unwrap();
        let rig = Rig::new(&calibration, 98.0).unwrap();
        let truth = Pose::from_parts(
            Translation3::new(0.3, -0.2, 0.05),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.02, -0.01, 0.08)),
        );
        let mut rng = XorShift::new(7);
        let mut unit = || (rng.next_u64() % 10_000) as f64 / 10_000.0;
        let mut corrs = Vec::new();
        let mut n_in = 0;
        for cam in 0..rig.cams.len() {
            let mut made = 0;
            while made < 40 {
                // a point in the current camera's frame, 1 to 5 m ahead and inside the fisheye view
                let depth = 1.0 + 4.0 * unit();
                let pc = Vector3::new(
                    (unit() - 0.5) * 2.0 * depth,
                    (unit() - 0.5) * 1.4 * depth,
                    depth,
                );
                let Some(uv) = rig.project(cam, &pc) else {
                    continue;
                };
                let p_cur = tf(&rig.t_imu_cam[cam], &pc);
                let p_cand = tf(&truth.inverse(), &p_cur);
                let outlier = unit() < 0.3;
                let bearing = if outlier {
                    rig.unproject(
                        cam,
                        &nalgebra::Vector2::new(unit() * rig.width(cam), unit() * rig.height(cam)),
                    )
                } else {
                    n_in += 1;
                    rig.unproject(cam, &uv)
                };
                corrs.push(Corr {
                    p: p_cand,
                    bearing,
                    cam,
                });
                made += 1;
            }
        }
        let guess = Pose::from_parts(
            Translation3::new(0.1, 0.0, 0.0),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.03, 0.03)),
        );
        let cam0: Vec<Corr> = corrs.iter().copied().filter(|c| c.cam == 0).collect();
        let (t1, mask) = ransac(&cam0, &rig, &guess, 0.02, 300, 0, 0.2).expect("a solution");
        let d = truth.inverse() * t1;
        assert!(
            d.translation.vector.norm() < 0.01,
            "step 1 translation off by {}",
            d.translation.vector.norm()
        );
        assert!(
            d.rotation.angle().to_degrees() < 0.5,
            "step 1 rotation off by {} deg",
            d.rotation.angle().to_degrees()
        );
        let n_in0 = mask.iter().filter(|&&m| m).count();
        assert!(n_in0 >= 20, "only {n_in0} inliers on camera 0");
        let (t2, inl, _) = refine(&t1, &rig, &corrs, 0.02, 0.2);
        let d = truth.inverse() * t2;
        assert!(
            d.translation.vector.norm() < 0.005,
            "step 2 translation off by {}",
            d.translation.vector.norm()
        );
        let n2 = inl.iter().filter(|&&m| m).count();
        assert!(
            n2 >= n_in - 5 && n2 <= n_in + 10,
            "step 2 inliers {n2} vs {n_in} true"
        );
    }
}
