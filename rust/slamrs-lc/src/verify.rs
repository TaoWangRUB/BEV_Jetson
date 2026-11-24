//! Geometric verification of a loop candidate (openspec design D2, D2c, D3): cuVSLAM's
//! correspondence step and two-step solver, as scripts/slamrs/loop_closure/verify.py does it.
//!
//! The candidate's landmarks, and those of the keyframes in sight of it, are projected into the
//! current cameras with the pose guess; their stored appearance (the frontend's own observation
//! at keyframe time) is tracked there with Lucas-Kanade, forward-backward checked and NCC-gated.
//! Step 1 solves the pair with most RANSAC inliers; step 2 refines jointly over everything with
//! a Huber loss and applies cuVSLAM's gates: a landmark floor, an inlier ratio that rises as the
//! count falls, and agreement between the two steps (waived when the refined pose already agrees
//! with the guess). The covariance is scaled by `1 + (max - 1) * (1 - ratio)`.
use nalgebra::Vector2;

use crate::config::{Features, Verify as VerifyCfg};
use crate::geom::{Pose, Rig, rot_angle_deg, tf};
use crate::keyframes::{CLASS_WEAK, KeyframeRecord};
use crate::klt::{KltParams, ncc, track};
use crate::pnp::{Corr, ransac, refine};

#[derive(Debug, Clone, Default)]
pub struct Hypothesis {
    pub cand: usize,
    pub cur: usize,
    pub route: &'static str,
    pub score: f64,
    pub accepted: bool,
    pub reason: String,
    pub t_cur_from_cand: Option<Pose>,
    pub sigma_t: f64,
    pub sigma_yaw: f64,
    pub tracked: usize,
    pub inliers1: usize,
    pub inliers2: usize,
    pub ratio: f64,
    pub best_pair: (i64, i64),
    pub pair_counts: Vec<((usize, usize), usize)>,
    pub guess_diff_m: f64,
    pub guess_diff_deg: f64,
    pub method: &'static str,
    pub n_sel: usize,
    pub n_proj: usize,
    pub n_klt: usize,
    pub n_fb: usize,
    pub n_err: usize,
    pub n_ncc: usize,
    pub n_weak: usize,
    pub tracked3d: usize,
    pub far_tracked: usize,
    pub far_inliers: usize,
}

#[derive(Default)]
pub struct Stages {
    pub sel: usize,
    pub proj: usize,
    pub klt: usize,
    pub fb: usize,
    pub err: usize,
    pub ncc: usize,
    pub weak: usize,
}

/// One camera pair's correspondences: source camera i tracked into current camera j.
pub struct Pair {
    pub i: usize,
    pub j: usize,
    pub corrs: Vec<Corr>,
}

/// cuVSLAM's correspondence step for one source keyframe against the current one.
/// `t_cur_from_src` is the guess; the 3D points come out in the SOURCE rig frame.
pub fn klt_pairs(
    src: &KeyframeRecord,
    cur: &KeyframeRecord,
    t_cur_from_src: &Pose,
    rig: &Rig,
    vc: &VerifyCfg,
    fc: &Features,
    out: &mut Vec<Pair>,
    st: &mut Stages,
) {
    let (Some(src_pyr), Some(cur_pyr)) = (&src.pyramids, &cur.pyramids) else {
        return;
    };
    if src.lm_ids.is_empty() {
        return;
    }
    let prm = KltParams {
        win: vc.klt_win,
        levels: vc.klt_levels,
        iters: 30,
        eps: 0.01,
    };
    let cls = src.depth_class(fc);
    for (i, (ids, uv_s)) in src.lm_obs.iter().enumerate() {
        // the landmarks this camera observed, with enough age and a trusted depth
        let mut sel: Vec<(usize, usize)> = Vec::new(); // (observation index, landmark index)
        for (n, id) in ids.iter().enumerate() {
            let Some(&l) = src.id_index.get(id) else {
                continue;
            };
            if src.lm_age[l] < fc.min_landmark_age {
                continue;
            }
            if cls[l] == CLASS_WEAK {
                st.weak += 1;
                continue;
            }
            sel.push((n, l));
        }
        st.sel += sel.len();
        if sel.len() < 3 {
            continue;
        }
        for (j, _cam) in rig.cams.iter().enumerate() {
            let mut corrs = Vec::new();
            let mut kept = 0usize;
            let mut proj: Vec<(usize, usize, Vector2<f64>)> = Vec::with_capacity(sel.len());
            for &(n, l) in &sel {
                let p_src = src.lm_pos_rig[l];
                let p_cur = tf(t_cur_from_src, &p_src);
                let pc = tf(&rig.t_cam_imu[j], &p_cur);
                if pc.z <= 0.02 {
                    continue;
                }
                let Some(pred) = rig.project(j, &pc) else {
                    continue;
                };
                proj.push((n, l, pred));
            }
            st.proj += proj.len();
            if proj.len() < 3 {
                continue;
            }
            for &(n, l, pred) in &proj {
                let p_src = src.lm_pos_rig[l];
                let p0 = uv_s[n];
                let Some((p1, err)) = track(&src_pyr[i], &cur_pyr[j], p0, pred, &prm) else {
                    continue;
                };
                st.klt += 1;
                // forward-backward check: track back to the source and require a return
                let Some((back, _)) = track(&cur_pyr[j], &src_pyr[i], p1, p0, &prm) else {
                    continue;
                };
                if (back - p0).norm() > vc.klt_fb_max_px {
                    continue;
                }
                st.fb += 1;
                if f64::from(err) >= vc.klt_max_err {
                    continue;
                }
                st.err += 1;
                let Some(c) = ncc(
                    &src_pyr[i].levels[0],
                    p0,
                    &cur_pyr[j].levels[0],
                    p1,
                    vc.klt_patch,
                ) else {
                    continue;
                };
                if f64::from(c) < vc.klt_ncc_min {
                    continue;
                }
                kept += 1;
                corrs.push(Corr {
                    p: p_src,
                    bearing: rig.unproject(j, &p1),
                    cam: j,
                });
            }
            st.ncc += kept;
            if corrs.len() >= 3 {
                out.push(Pair { i, j, corrs });
            }
        }
    }
}

fn lerp_inv(a: f64, b: f64, v: f64) -> f64 {
    ((v - a) / (b - a)).clamp(0.0, 1.0)
}

/// Verify candidate `cand` against `cur`. `neighbours`: the keyframes in sight of the candidate
/// with `T_cand_from_nb`, whose landmarks join every step (cuVSLAM fetches every landmark near
/// the guess, then picks the single keyframe owning most of them for step 1).
pub fn verify(
    recs: &[KeyframeRecord],
    cand: usize,
    cur: usize,
    route: &'static str,
    rig: &Rig,
    t_guess: &Pose,
    vc: &VerifyCfg,
    fc: &Features,
    neighbours: &[(usize, Pose)],
) -> Hypothesis {
    let mut h = Hypothesis {
        cand,
        cur,
        route,
        method: "klt",
        ..Default::default()
    };
    let mut st = Stages::default();
    let mut pairs: Vec<(bool, Pair)> = Vec::new(); // (own keyframe, pair) with points in the CANDIDATE frame
    let mut sources: Vec<(usize, Pose, bool)> = vec![(cand, Pose::identity(), true)];
    sources.extend(neighbours.iter().map(|&(k, t)| (k, t, false)));
    for (k, t_cand_from_src, own) in &sources {
        let mut found = Vec::new();
        klt_pairs(
            &recs[*k],
            &recs[cur],
            &(t_guess * t_cand_from_src),
            rig,
            vc,
            fc,
            &mut found,
            &mut st,
        );
        for mut pair in found {
            for c in pair.corrs.iter_mut() {
                c.p = tf(t_cand_from_src, &c.p);
            }
            if *own {
                h.pair_counts.push(((pair.i, pair.j), pair.corrs.len()));
            }
            pairs.push((*own, pair));
        }
    }
    h.n_sel = st.sel;
    h.n_proj = st.proj;
    h.n_klt = st.klt;
    h.n_fb = st.fb;
    h.n_err = st.err;
    h.n_ncc = st.ncc;
    h.n_weak = st.weak;
    h.pair_counts.sort();
    h.tracked = pairs.iter().map(|(_, p)| p.corrs.len()).sum();
    h.tracked3d = h.tracked;
    if h.tracked == 0 {
        h.reason = "no_matches".into();
        return h;
    }
    // step 1: the single pair with most RANSAC inliers, from any source keyframe
    let mut best: Option<(usize, Pose, (usize, usize))> = None;
    for (_, pair) in &pairs {
        if pair.corrs.len() < vc.min_pair_matches {
            continue;
        }
        let Some((t1, mask)) = ransac(
            &pair.corrs,
            rig,
            t_guess,
            vc.ransac_thresh_norm,
            vc.ransac_iters,
            vc.seed,
            fc.min_bearing_z,
        ) else {
            continue;
        };
        let n = mask.iter().filter(|&&m| m).count();
        if best.as_ref().is_none_or(|b| n > b.0) {
            best = Some((n, t1, (pair.i, pair.j)));
        }
    }
    let Some((n1, t1, bp)) = best else {
        h.reason = "no_pair_solution".into();
        return h;
    };
    h.inliers1 = n1;
    h.best_pair = (bp.0 as i64, bp.1 as i64);
    // step 2: joint refinement from t1 over everything
    let all: Vec<Corr> = pairs
        .iter()
        .flat_map(|(_, p)| p.corrs.iter().copied())
        .collect();
    let (t2, inl, _err) = refine(&t1, rig, &all, vc.ransac_thresh_norm, fc.min_bearing_z);
    h.inliers2 = inl.iter().filter(|&&m| m).count();
    h.ratio = h.inliers2 as f64 / h.tracked3d.max(1) as f64;
    h.t_cur_from_cand = Some(t2);
    let d = t_guess.inverse() * t2;
    h.guess_diff_m = d.translation.vector.norm();
    h.guess_diff_deg = rot_angle_deg(&d.rotation.to_rotation_matrix());
    // gates (cuVSLAM LoopClosureSolverTwoStepsEasy)
    let d12 = t1.inverse() * t2;
    let agrees =
        h.guess_diff_m <= vc.consistency_tol_m && h.guess_diff_deg <= vc.consistency_tol_deg;
    if !agrees
        && d12.translation.vector.norm() > h.guess_diff_m
        && rot_angle_deg(&d12.rotation.to_rotation_matrix()) > h.guess_diff_deg
    {
        h.reason = "inconsistent_steps".into();
        return h;
    }
    if h.inliers2 < vc.min_landmarks {
        h.reason = "few_landmarks".into();
        return h;
    }
    let f = lerp_inv(vc.ratio_lo_at, vc.ratio_hi_at, h.inliers2 as f64);
    let thr = ((1.0 - f) * vc.ratio_lo + f * vc.ratio_hi).max(vc.ratio_hi);
    if h.ratio < thr {
        h.reason = format!("low_ratio<{thr:.2}");
        return h;
    }
    let scale = 1.0 + (vc.cov_scale_max - 1.0) * (1.0 - h.ratio);
    h.sigma_t = vc.base_sigma_t_m * scale.sqrt();
    h.sigma_yaw = vc.base_sigma_yaw_deg.to_radians() * scale.sqrt();
    h.accepted = true;
    h.reason = "ok".into();
    h
}
