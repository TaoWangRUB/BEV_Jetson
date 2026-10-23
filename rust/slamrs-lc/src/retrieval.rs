//! Loop-candidate retrieval (openspec design D1, D1a, D1b): keyframes within the tracker's reach
//! of the current VIO pose, older than `min_separation_s` and with at least `min_path_m` of
//! travelled path between them. The appearance route (ORB bag of words) stays in the Python
//! reference, off by default there and measured never to produce a closure on this rig.
use nalgebra::Vector3;

use crate::config::Retrieval;

pub struct Retriever {
    cfg: Retrieval,
    t: Vec<i64>,
    p: Vec<Vector3<f64>>,
    /// Travelled path up to each keyframe, metres along the VIO trajectory.
    s: Vec<f64>,
}

impl Retriever {
    pub fn new(cfg: Retrieval) -> Self {
        Self {
            cfg,
            t: Vec::new(),
            p: Vec::new(),
            s: Vec::new(),
        }
    }

    pub fn add(&mut self, t_ns: i64, p: Vector3<f64>) {
        let s = match self.p.last() {
            Some(prev) => self.s[self.s.len() - 1] + (p - prev).norm(),
            None => 0.0,
        };
        self.t.push(t_ns);
        self.p.push(p);
        self.s.push(s);
    }

    /// Candidates for keyframe `index`: (candidate index, distance), nearest first, capped.
    pub fn candidates(&self, index: usize) -> Vec<(usize, f64)> {
        let min_sep = (self.cfg.min_separation_s * 1e9) as i64;
        let mut out: Vec<(usize, f64)> = (0..index)
            .filter(|&i| {
                self.t[index] - self.t[i] >= min_sep
                    && self.s[index] - self.s[i] >= self.cfg.min_path_m
            })
            .map(|i| (i, (self.p[i] - self.p[index]).norm()))
            .filter(|&(_, d)| d <= self.cfg.spatial_radius_m)
            .collect();
        out.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        out.truncate(self.cfg.max_candidates);
        out
    }
}
