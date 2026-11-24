//! Keyframe records taken from the VIO (openspec design D2a, D2b, D2c), the port of
//! scripts/slamrs/loop_closure/keyframes.py.
//!
//! A record is finalised when the estimator marginalises the keyframe. It carries every landmark
//! the keyframe hosted or observed, each stored in the keyframe's own rig frame at the snapshot
//! it was last seen in (the window re-optimises poses and landmarks together, so a world position
//! frozen at an earlier snapshot is inconsistent with the keyframe's later pose). After
//! finalisation the record stays open: while a landmark is alive its position follows the latest
//! snapshot, and its parallax (the angle at the landmark between the first and latest rig
//! position that observed it) and observation count keep growing. Landmarks under the parallax or
//! observation floor are weak and never tracked.
use std::collections::{BTreeMap, HashMap, HashSet};

use nalgebra::{Vector2, Vector3};

use crate::config::Features;
use crate::geom::{Pose, Rig, pose7_to_iso, tf};
use crate::klt::Pyramid;
use crate::vio::{FlowObs, Snapshot};

pub const CLASS_3D: u8 = 0;
pub const CLASS_FAR: u8 = 1;
pub const CLASS_WEAK: u8 = 2;

#[allow(dead_code)] // lm_hosted and index mirror the reference's record
pub struct KeyframeRecord {
    pub t_ns: i64,
    /// world_from_imu at finalisation.
    pub t_vio: Pose,
    pub images: Vec<Vec<u8>>,
    pub pyramids: Option<Vec<Pyramid>>,
    pub lm_ids: Vec<i64>,
    /// World positions, consistent with `t_vio`.
    pub lm_pos: Vec<Vector3<f64>>,
    pub lm_age: Vec<i64>,
    pub lm_hosted: Vec<bool>,
    pub lm_parallax_deg: Vec<f64>,
    pub lm_nobs: Vec<i64>,
    pub lm_baseline: Vec<f64>,
    pub lm_refreshed: Vec<i64>,
    /// Per camera: the frontend's keypoint ids and pixels at keyframe time.
    pub lm_obs: Vec<(Vec<i64>, Vec<Vector2<f64>>)>,
    /// Landmark id -> (camera, observed pixel), the first camera that saw it.
    pub obs_map: HashMap<i64, (usize, Vector2<f64>)>,
    /// Landmark id -> index into the lm_* vectors.
    pub id_index: HashMap<i64, usize>,
    /// Rig-frame positions, cached from `lm_pos` and `t_vio`.
    pub lm_pos_rig: Vec<Vector3<f64>>,
    pub index: usize,
}

impl KeyframeRecord {
    pub fn refresh_rig(&mut self) {
        let t_iw = self.t_vio.inverse();
        self.lm_pos_rig = self.lm_pos.iter().map(|p| tf(&t_iw, p)).collect();
    }

    /// Per landmark: CLASS_3D (parallax and observations enough for a point), CLASS_FAR (seen
    /// enough but with no parallax: a direction), CLASS_WEAK (neither; not used).
    pub fn depth_class(&self, cfg: &Features) -> Vec<u8> {
        (0..self.lm_ids.len())
            .map(|k| {
                let enough = self.lm_nobs[k] >= cfg.min_obs;
                if enough && self.lm_parallax_deg[k] < cfg.far_parallax_deg {
                    CLASS_FAR
                } else if enough && self.lm_parallax_deg[k] >= cfg.min_parallax_deg {
                    CLASS_3D
                } else {
                    CLASS_WEAK
                }
            })
            .collect()
    }

    pub fn ensure_pyramids(&mut self, levels: usize, win: usize, width: usize, height: usize) {
        if self.pyramids.is_none() {
            self.pyramids = Some(
                self.images
                    .iter()
                    .map(|im| Pyramid::build(im, width, height, levels + 1, win))
                    .collect(),
            );
        }
    }
}

struct Pending {
    images: Vec<Vec<u8>>,
    pose7: Option<[f64; 7]>,
    /// id -> (rig-frame position, snapshots alive, hosted)
    lm: BTreeMap<i64, (Vector3<f64>, i64, bool)>,
    obs: Vec<(Vec<i64>, Vec<Vector2<f64>>)>,
    obs_ids: HashSet<i64>,
}

/// Where and how often a landmark was observed: first and latest rig position, frames.
struct Seen {
    p_first: Vector3<f64>,
    p_last: Vector3<f64>,
    n_obs: i64,
}

pub struct KeyframeTracker {
    pending: BTreeMap<i64, Pending>,
    seen: HashSet<i64>,
    table: HashMap<i64, Seen>,
    pub done: Vec<KeyframeRecord>,
    refresh: bool,
    refresh_max_px: f64,
}

fn parallax_deg(p_first: &Vector3<f64>, p_last: &Vector3<f64>, l: &Vector3<f64>) -> f64 {
    let a = p_first - l;
    let b = p_last - l;
    let d = (a.norm() * b.norm()).max(1e-12);
    (a.dot(&b) / d).clamp(-1.0, 1.0).acos().to_degrees()
}

impl KeyframeTracker {
    pub fn new(cfg: &Features) -> Self {
        Self {
            pending: BTreeMap::new(),
            seen: HashSet::new(),
            table: HashMap::new(),
            done: Vec::new(),
            refresh: cfg.refresh_enabled,
            refresh_max_px: cfg.refresh_max_px,
        }
    }

    /// Call after every `track()`. Returns the indices into `done` of the records finalised now.
    pub fn update(
        &mut self,
        t_ns: i64,
        images: &[Vec<u8>],
        snap: &Snapshot,
        flow: Option<&FlowObs>,
        rig: &Rig,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        // 1. the observation table: every id tracked on this frame, from this frame's rig position
        let w_now = snap.window_t_ns.iter().position(|&t| t == t_ns);
        if let (Some(flow), Some(w)) = (flow, w_now) {
            if flow.t_ns == t_ns {
                let p_now = Vector3::new(
                    snap.window_poses[w][0],
                    snap.window_poses[w][1],
                    snap.window_poses[w][2],
                );
                let mut ids: Vec<i64> = flow
                    .cams
                    .iter()
                    .flat_map(|(ids, _)| ids.iter().copied())
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                for id in ids {
                    self.table
                        .entry(id)
                        .and_modify(|e| {
                            e.p_last = p_now;
                            e.n_obs += 1;
                        })
                        .or_insert(Seen {
                            p_first: p_now,
                            p_last: p_now,
                            n_obs: 1,
                        });
                }
            }
        }
        // 2. a new keyframe starts a pending record
        if snap.kf_ids.contains(&t_ns)
            && !self.pending.contains_key(&t_ns)
            && !self.seen.contains(&t_ns)
        {
            let obs: Vec<(Vec<i64>, Vec<Vector2<f64>>)> = match flow {
                Some(f) if f.t_ns == t_ns => f
                    .cams
                    .iter()
                    .map(|(ids, uv)| {
                        (
                            ids.clone(),
                            uv.iter().map(|p| Vector2::new(p[0], p[1])).collect(),
                        )
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let obs_ids: HashSet<i64> = obs
                .iter()
                .flat_map(|(ids, _)| ids.iter().copied())
                .collect();
            self.pending.insert(
                t_ns,
                Pending {
                    images: images.to_vec(),
                    pose7: None,
                    lm: BTreeMap::new(),
                    obs,
                    obs_ids,
                },
            );
            self.seen.insert(t_ns);
        }
        // 3. pending records follow the window
        if !self.pending.is_empty() {
            for (tk, rec) in self.pending.iter_mut() {
                let Some(w) = snap.window_t_ns.iter().position(|t| t == tk) else {
                    continue; // out of the window: nothing in this snapshot is consistent with it
                };
                rec.pose7 = Some(snap.window_poses[w]);
                let t_iw = pose7_to_iso(&snap.window_poses[w]).inverse();
                for (k, &id) in snap.landmark_ids.iter().enumerate() {
                    let hosted = snap.landmark_hosts[k] == *tk;
                    if !hosted && !rec.obs_ids.contains(&id) {
                        continue;
                    }
                    let p = snap.landmark_positions[k];
                    if !p.iter().all(|v| v.is_finite()) {
                        continue;
                    }
                    let p_rig = tf(&t_iw, &Vector3::new(p[0], p[1], p[2]));
                    rec.lm
                        .entry(id)
                        .and_modify(|e| {
                            e.0 = p_rig;
                            e.1 += 1;
                            e.2 = e.2 || hosted;
                        })
                        .or_insert((p_rig, 1, hosted));
                }
            }
            let marg: Vec<i64> = snap
                .marginalized
                .iter()
                .copied()
                .filter(|t| self.pending.contains_key(t))
                .collect();
            for tm in marg {
                out.push(self.finalise(tm));
            }
        }
        // 4. finalised records stay open while the VIO still observes their landmarks
        if !self.done.is_empty() && !snap.landmark_ids.is_empty() {
            self.refresh_done(snap, rig);
        }
        out
    }

    /// End of log: finalise what is still in the window, oldest first.
    pub fn flush(&mut self) -> Vec<usize> {
        let keys: Vec<i64> = self.pending.keys().copied().collect();
        keys.into_iter().map(|t| self.finalise(t)).collect()
    }

    fn confidence(&self, ids: &[i64], pos: &[Vector3<f64>]) -> (Vec<f64>, Vec<i64>, Vec<f64>) {
        let mut par = Vec::with_capacity(ids.len());
        let mut nobs = Vec::with_capacity(ids.len());
        let mut base = Vec::with_capacity(ids.len());
        for (k, id) in ids.iter().enumerate() {
            match self.table.get(id) {
                Some(e) => {
                    par.push(parallax_deg(&e.p_first, &e.p_last, &pos[k]));
                    nobs.push(e.n_obs);
                    base.push((e.p_last - e.p_first).norm());
                }
                None => {
                    par.push(0.0);
                    nobs.push(0);
                    base.push(0.0);
                }
            }
        }
        (par, nobs, base)
    }

    fn finalise(&mut self, t: i64) -> usize {
        let rec = self.pending.remove(&t).expect("pending record");
        let pose7 = rec.pose7.expect("a keyframe never had a window pose");
        let t_vio = pose7_to_iso(&pose7);
        let lm_ids: Vec<i64> = rec.lm.keys().copied().collect();
        let lm_pos: Vec<Vector3<f64>> = lm_ids.iter().map(|id| tf(&t_vio, &rec.lm[id].0)).collect();
        let lm_age: Vec<i64> = lm_ids.iter().map(|id| rec.lm[id].1).collect();
        let lm_hosted: Vec<bool> = lm_ids.iter().map(|id| rec.lm[id].2).collect();
        let (lm_parallax_deg, lm_nobs, lm_baseline) = self.confidence(&lm_ids, &lm_pos);
        let mut obs_map = HashMap::new();
        for (c, (ids, uv)) in rec.obs.iter().enumerate() {
            for (k, id) in ids.iter().enumerate() {
                obs_map.entry(*id).or_insert((c, uv[k]));
            }
        }
        let id_index = lm_ids.iter().enumerate().map(|(k, id)| (*id, k)).collect();
        let n = lm_ids.len();
        let mut r = KeyframeRecord {
            t_ns: t,
            t_vio,
            images: rec.images,
            pyramids: None,
            lm_ids,
            lm_pos,
            lm_age,
            lm_hosted,
            lm_parallax_deg,
            lm_nobs,
            lm_baseline,
            lm_refreshed: vec![0; n],
            lm_obs: rec.obs,
            obs_map,
            id_index,
            lm_pos_rig: Vec::new(),
            index: self.done.len(),
        };
        r.refresh_rig();
        self.done.push(r);
        self.done.len() - 1
    }

    fn refresh_done(&mut self, snap: &Snapshot, rig: &Rig) {
        let alive: HashMap<i64, Vector3<f64>> = snap
            .landmark_ids
            .iter()
            .zip(snap.landmark_positions.iter())
            .filter(|(_, p)| p.iter().all(|v| v.is_finite()))
            .map(|(&id, p)| (id, Vector3::new(p[0], p[1], p[2])))
            .collect();
        let refresh = self.refresh;
        let max_px = self.refresh_max_px;
        // the table is read while records are written: split the borrow
        let table = std::mem::take(&mut self.table);
        for rec in self.done.iter_mut() {
            let mut any = false;
            let t_iw = rec.t_vio.inverse();
            for k in 0..rec.lm_ids.len() {
                let id = rec.lm_ids[k];
                let Some(p_new) = alive.get(&id) else {
                    continue;
                };
                match table.get(&id) {
                    Some(e) => {
                        rec.lm_parallax_deg[k] =
                            parallax_deg(&e.p_first, &e.p_last, &rec.lm_pos[k]);
                        rec.lm_nobs[k] = e.n_obs;
                        rec.lm_baseline[k] = (e.p_last - e.p_first).norm();
                    }
                    None => {
                        rec.lm_parallax_deg[k] = 0.0;
                        rec.lm_nobs[k] = 0;
                        rec.lm_baseline[k] = 0.0;
                    }
                }
                if !refresh || (p_new - rec.lm_pos[k]).norm() <= 1e-9 {
                    continue;
                }
                // the consistency rule, forward in time: the keyframe's own observation must still
                // project within refresh_max_px of the new position under the frozen pose
                let Some(&(c, uv_obs)) = rec.obs_map.get(&id) else {
                    continue;
                };
                let pc = tf(&rig.t_cam_imu[c], &tf(&t_iw, p_new));
                if pc.z <= 0.02 {
                    continue;
                }
                let Some(uv) = rig.project(c, &pc) else {
                    continue;
                };
                if (uv - uv_obs).norm() <= max_px {
                    rec.lm_pos[k] = *p_new;
                    rec.lm_refreshed[k] += 1;
                    any = true;
                }
            }
            if any {
                rec.refresh_rig();
            }
        }
        self.table = table;
    }
}
