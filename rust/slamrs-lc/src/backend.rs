//! The loop-closing backend (openspec design D6): on_keyframe = retrieve -> verify -> accept ->
//! optimise, bounded. Synchronous here; the interface is what a thread would call. Every
//! hypothesis, accepted or not, is recorded with its reason. One accepted closure per keyframe,
//! then a throttle interval before the next verification (cuVSLAM's throttling_time_ms, 0 by
//! default offline). The output files are those of run_slamrs_lc.py, column for column.
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::config::LcConfig;
use crate::geom::{Pose, Rig, iso_to_pose7};
use crate::keyframes::{CLASS_3D, CLASS_FAR, CLASS_WEAK, KeyframeRecord};
use crate::posegraph::PoseGraph4DoF;
use crate::retrieval::Retriever;
use crate::verify::{Hypothesis, verify};

pub struct Timing {
    pub t_ns: i64,
    pub index: usize,
    pub n_assoc: usize,
    pub candidates: usize,
    pub feat_ms: f64,
    pub retrieve_ms: f64,
    pub verify_ms: f64,
    pub optimize_ms: f64,
    pub accepted: bool,
    pub note: &'static str,
}

pub struct LoopClosureBackend {
    cfg: LcConfig,
    pub graph: PoseGraph4DoF,
    retriever: Retriever,
    pub hypotheses: Vec<Hypothesis>,
    pub timings: Vec<Timing>,
    /// Accepted hypotheses with their edge index in the graph.
    pub edges: Vec<(Hypothesis, usize)>,
    last_accept_t: Option<i64>,
    pub dropped: Vec<(usize, usize, f64)>,
    width: usize,
    height: usize,
}

impl LoopClosureBackend {
    pub fn new(cfg: LcConfig, width: usize, height: usize) -> Self {
        Self {
            graph: PoseGraph4DoF::new(cfg.graph.clone()),
            retriever: Retriever::new(cfg.retrieval.clone()),
            cfg,
            hypotheses: Vec::new(),
            timings: Vec::new(),
            edges: Vec::new(),
            last_accept_t: None,
            dropped: Vec::new(),
            width,
            height,
        }
    }

    /// Keyframes near the candidate (by VIO pose and index) and old enough with respect to the
    /// current keyframe; their landmarks are expressed in the candidate's frame through the VIO
    /// relative pose, which is locally consistent.
    fn in_sight(&self, recs: &[KeyframeRecord], ci: usize, cur: usize) -> Vec<(usize, Pose)> {
        let vc = &self.cfg.verify;
        let min_sep = (self.cfg.retrieval.min_separation_s * 1e9) as i64;
        let kc = &recs[ci];
        let kk = &recs[cur];
        let lo = (ci as i64 - vc.neighbour_max_kf).max(0) as usize;
        let hi = ((ci as i64 + vc.neighbour_max_kf) as usize).min(recs.len() - 1);
        (lo..=hi)
            .filter(|&k| k != ci && kk.t_ns - recs[k].t_ns >= min_sep)
            .filter(|&k| {
                (recs[k].t_vio.translation.vector - kc.t_vio.translation.vector).norm()
                    <= vc.neighbour_radius_m
            })
            .map(|k| (k, kc.t_vio.inverse() * recs[k].t_vio))
            .collect()
    }

    /// The record at `idx` has just been finalised. Returns the accepted hypothesis, if any.
    pub fn on_keyframe(
        &mut self,
        recs: &mut [KeyframeRecord],
        idx: usize,
        rig: &Rig,
    ) -> Option<Hypothesis> {
        let t0 = Instant::now();
        let (t_ns, t_vio, n_assoc) = {
            let r = &recs[idx];
            (
                r.t_ns,
                r.t_vio,
                r.lm_obs
                    .iter()
                    .map(|(ids, _)| {
                        ids.iter()
                            .filter(|id| {
                                r.id_index.get(id).is_some_and(|&l| {
                                    r.lm_age[l] >= self.cfg.features.min_landmark_age
                                })
                            })
                            .count()
                    })
                    .sum(),
            )
        };
        self.graph.add_node(t_ns, &t_vio);
        self.retriever.add(t_ns, t_vio.translation.vector);
        let t_feat = Instant::now();
        let mut row = Timing {
            t_ns,
            index: idx,
            n_assoc,
            candidates: 0,
            feat_ms: (t_feat - t0).as_secs_f64() * 1e3,
            retrieve_ms: 0.0,
            verify_ms: 0.0,
            optimize_ms: 0.0,
            accepted: false,
            note: "",
        };
        let throttle = (self.cfg.backend.throttle_s * 1e9) as i64;
        if let Some(last) = self.last_accept_t {
            if throttle > 0 && t_ns - last < throttle {
                row.note = "throttled";
                self.timings.push(row);
                return None;
            }
        }
        let cands = self.retriever.candidates(idx);
        let t_ret = Instant::now();
        row.candidates = cands.len();
        row.retrieve_ms = (t_ret - t_feat).as_secs_f64() * 1e3;
        let mut accepted = None;
        for (ci, dist) in cands {
            let nbs = self.in_sight(recs, ci, idx);
            // pyramids for every keyframe the verification reads
            let mut needed: Vec<usize> = vec![ci, idx];
            needed.extend(nbs.iter().map(|(k, _)| *k));
            for k in needed {
                recs[k].ensure_pyramids(
                    self.cfg.verify.klt_levels,
                    self.cfg.verify.klt_win,
                    self.width,
                    self.height,
                );
            }
            let t_guess = recs[idx].t_vio.inverse() * recs[ci].t_vio; // odometry's T_cur_from_cand
            let mut h = verify(
                recs,
                ci,
                idx,
                "spatial",
                rig,
                &t_guess,
                &self.cfg.verify,
                &self.cfg.features,
                &nbs,
            );
            h.score = dist;
            let ok = h.accepted;
            self.hypotheses.push(h.clone());
            if ok {
                accepted = Some(h);
                break;
            }
        }
        let t_ver = Instant::now();
        row.verify_ms = (t_ver - t_ret).as_secs_f64() * 1e3;
        if let Some(h) = &accepted {
            let t = h.t_cur_from_cand.expect("accepted hypothesis has a pose");
            let e = self
                .graph
                .add_loop_edge(h.cur, h.cand, &t, h.sigma_t, h.sigma_yaw);
            self.edges.push((h.clone(), e));
            self.solve_and_check();
            self.last_accept_t = Some(t_ns);
            row.accepted = true;
        }
        row.optimize_ms = t_ver.elapsed().as_secs_f64() * 1e3;
        self.timings.push(row);
        accepted
    }

    fn solve_and_check(&mut self) {
        let chk = self.cfg.graph.edge_check_sigma;
        loop {
            let r = self.graph.optimize();
            if r.is_empty() {
                return;
            }
            let (worst, &max) = r
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("non-empty");
            if max <= chk {
                return;
            }
            let loops = self.graph.loop_edge_indices();
            let bad = loops[worst];
            let (i, j) = (self.graph.edges[bad].i, self.graph.edges[bad].j);
            eprintln!("  loop edge {i}->{j} residual {max:.1} sigma > {chk}: dropped, re-solving");
            self.graph.drop_edge(bad);
            self.dropped.push((i, j, max));
            self.edges.retain(|(_, e)| *e != bad);
            for (_, e) in self.edges.iter_mut() {
                if *e > bad {
                    *e -= 1;
                }
            }
        }
    }

    pub fn finish(&mut self) {
        if !self.graph.loop_edge_indices().is_empty() {
            self.solve_and_check();
        }
    }

    /// Per-frame correction from the preceding keyframe (design D5).
    pub fn correct_frames(&self, recs: &[KeyframeRecord], frames: &[(i64, Pose)]) -> Vec<Pose> {
        if recs.is_empty() {
            return frames.iter().map(|(_, t)| *t).collect();
        }
        let corr = self.graph.poses();
        let kf_t: Vec<i64> = recs.iter().map(|r| r.t_ns).collect();
        frames
            .iter()
            .map(|(t, tf)| {
                let k = kf_t.partition_point(|&x| x <= *t).saturating_sub(1);
                corr[k] * recs[k].t_vio.inverse() * tf
            })
            .collect()
    }

    pub fn write(
        &self,
        out: &Path,
        recs: &[KeyframeRecord],
        frames: &[(i64, Pose, i64)],
    ) -> Result<Summary, String> {
        let io = |e: std::io::Error| e.to_string();
        let corr = self.graph.poses();
        // slamrs_lc.csv: every frame's corrected pose
        let corrected = self.correct_frames(
            recs,
            &frames.iter().map(|(t, p, _)| (*t, *p)).collect::<Vec<_>>(),
        );
        let mut f = fs::File::create(out.join("slamrs_lc.csv")).map_err(io)?;
        writeln!(f, "t_ns,status,tx,ty,tz,qx,qy,qz,qw").map_err(io)?;
        for ((t, _, status), c) in frames.iter().zip(corrected.iter()) {
            let p = iso_to_pose7(c);
            writeln!(
                f,
                "{t},{status},{}",
                p.iter()
                    .map(|v| crate::fmtg::g9(*v))
                    .collect::<Vec<_>>()
                    .join(",")
            )
            .map_err(io)?;
        }
        // keyframes_lc.csv
        let mut f = fs::File::create(out.join("keyframes_lc.csv")).map_err(io)?;
        writeln!(f, "t_ns,index,vio_tx,vio_ty,vio_tz,vio_qx,vio_qy,vio_qz,vio_qw,lc_tx,lc_ty,lc_tz,lc_qx,lc_qy,lc_qz,lc_qw,n_assoc").map_err(io)?;
        for (k, r) in recs.iter().enumerate() {
            let v = iso_to_pose7(&r.t_vio);
            let c = iso_to_pose7(&corr[k]);
            let n_assoc = self.timings.get(k).map_or(0, |t| t.n_assoc);
            writeln!(
                f,
                "{},{k},{},{},{n_assoc}",
                r.t_ns,
                v.iter()
                    .map(|x| format!("{x}"))
                    .collect::<Vec<_>>()
                    .join(","),
                c.iter()
                    .map(|x| format!("{x}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
            .map_err(io)?;
        }
        // loop_edges.csv
        let mut f = fs::File::create(out.join("loop_edges.csv")).map_err(io)?;
        writeln!(f, "cur,cand,t_cur_ns,t_cand_ns,route,inliers,tracked,ratio,sigma_t,sigma_yaw_deg,best_pair,sep_vio_m,sep_lc_m,tracked3d,far_tracked,far_inliers").map_err(io)?;
        for (h, _) in &self.edges {
            let (kc, kk) = (&recs[h.cand], &recs[h.cur]);
            let sv = (kc.t_vio.translation.vector - kk.t_vio.translation.vector).norm();
            let sl = (corr[h.cand].translation.vector - corr[h.cur].translation.vector).norm();
            writeln!(
                f,
                "{},{},{},{},{},{},{},{:.3},{:.4},{:.3},{}-{},{sv:.3},{sl:.3},{},{},{}",
                h.cur,
                h.cand,
                kk.t_ns,
                kc.t_ns,
                h.route,
                h.inliers2,
                h.tracked,
                h.ratio,
                h.sigma_t,
                h.sigma_yaw.to_degrees(),
                h.best_pair.0 + 1,
                h.best_pair.1 + 1,
                h.tracked3d,
                h.far_tracked,
                h.far_inliers
            )
            .map_err(io)?;
        }
        // hypotheses.csv
        let mut f = fs::File::create(out.join("hypotheses.csv")).map_err(io)?;
        writeln!(f, "cur,cand,route,score,accepted,reason,tracked,inliers1,inliers2,ratio,best_pair,guess_diff_m,guess_diff_deg,method,n_sel,n_proj,n_klt,n_fb,n_err,n_ncc,n_weak,tracked3d,far_tracked,far_inliers,pairs").map_err(io)?;
        for h in &self.hypotheses {
            let pairs = h
                .pair_counts
                .iter()
                .map(|((i, j), n)| format!("{}-{}:{n}", i + 1, j + 1))
                .collect::<Vec<_>>()
                .join(";");
            writeln!(f, "{},{},{},{:.3},{},{},{},{},{},{:.3},{}-{},{:.3},{:.2},{},{},{},{},{},{},{},{},{},{},{},{pairs}", h.cur, h.cand, h.route, h.score, u8::from(h.accepted), h.reason, h.tracked, h.inliers1, h.inliers2, h.ratio, h.best_pair.0 + 1, h.best_pair.1 + 1, h.guess_diff_m, h.guess_diff_deg, h.method, h.n_sel, h.n_proj, h.n_klt, h.n_fb, h.n_err, h.n_ncc, h.n_weak, h.tracked3d, h.far_tracked, h.far_inliers).map_err(io)?;
        }
        // timings.csv
        let mut f = fs::File::create(out.join("timings.csv")).map_err(io)?;
        writeln!(
            f,
            "t_ns,index,n_assoc,candidates,feat_ms,retrieve_ms,verify_ms,optimize_ms,accepted,note"
        )
        .map_err(io)?;
        for t in &self.timings {
            writeln!(
                f,
                "{},{},{},{},{:.2},{:.2},{:.2},{:.2},{},{}",
                t.t_ns,
                t.index,
                t.n_assoc,
                t.candidates,
                t.feat_ms,
                t.retrieve_ms,
                t.verify_ms,
                t.optimize_ms,
                u8::from(t.accepted),
                t.note
            )
            .map_err(io)?;
        }
        // summary.json
        let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
        for h in &self.hypotheses {
            *reasons.entry(h.reason.clone()).or_insert(0) += 1;
        }
        let (mut c3, mut cf, mut cw, mut refreshed) = (0usize, 0usize, 0usize, 0usize);
        for r in recs {
            for c in r.depth_class(&self.cfg.features) {
                match c {
                    CLASS_3D => c3 += 1,
                    CLASS_FAR => cf += 1,
                    _ => cw += 1,
                }
            }
            refreshed += r.lm_refreshed.iter().filter(|&&x| x > 0).count();
        }
        let _ = CLASS_WEAK;
        let mut lm_per_kf: Vec<usize> = recs.iter().map(|r| r.lm_ids.len()).collect();
        lm_per_kf.sort_unstable();
        let median = |v: &[f64]| -> f64 {
            if v.is_empty() {
                return 0.0;
            }
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if s.len() % 2 == 1 {
                s[s.len() / 2]
            } else {
                0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2])
            }
        };
        let summary = Summary {
            keyframes: recs.len(),
            hypotheses: self.hypotheses.len(),
            accepted: self.edges.len(),
            dropped: self.dropped.clone(),
            reasons,
            classes_3d: c3,
            classes_far: cf,
            classes_weak: cw,
            refreshed_landmarks: refreshed,
            landmarks_per_keyframe_median: if lm_per_kf.is_empty() {
                0.0
            } else {
                lm_per_kf[lm_per_kf.len() / 2] as f64
            },
            feat_ms_median: median(&self.timings.iter().map(|t| t.feat_ms).collect::<Vec<_>>()),
            verify_ms_median: median(
                &self
                    .timings
                    .iter()
                    .filter(|t| t.candidates > 0)
                    .map(|t| t.verify_ms)
                    .collect::<Vec<_>>(),
            ),
        };
        let reasons_json = summary
            .reasons
            .iter()
            .map(|(k, v)| format!("\"{k}\": {v}"))
            .collect::<Vec<_>>()
            .join(", ");
        let dropped_json = summary
            .dropped
            .iter()
            .map(|(i, j, r)| format!("[{i}, {j}, {r}]"))
            .collect::<Vec<_>>()
            .join(", ");
        let text = format!(
            "{{\n \"keyframes\": {},\n \"hypotheses\": {},\n \"accepted\": {},\n \"dropped\": [{dropped_json}],\n \"reasons\": {{{reasons_json}}},\n \"landmark_classes\": {{\"3d\": {}, \"far\": {}, \"weak\": {}, \"refreshed_landmarks\": {}}},\n \"landmarks_per_keyframe_median\": {},\n \"orb\": false,\n \"feat_ms_median\": {},\n \"verify_ms_median\": {}\n}}\n",
            summary.keyframes,
            summary.hypotheses,
            summary.accepted,
            summary.classes_3d,
            summary.classes_far,
            summary.classes_weak,
            summary.refreshed_landmarks,
            summary.landmarks_per_keyframe_median,
            summary.feat_ms_median,
            summary.verify_ms_median
        );
        fs::write(out.join("summary.json"), text).map_err(io)?;
        Ok(summary)
    }
}

#[derive(Debug, Clone)]
pub struct Summary {
    pub keyframes: usize,
    pub hypotheses: usize,
    pub accepted: usize,
    pub dropped: Vec<(usize, usize, f64)>,
    pub reasons: BTreeMap<String, usize>,
    pub classes_3d: usize,
    pub classes_far: usize,
    pub classes_weak: usize,
    pub refreshed_landmarks: usize,
    pub landmarks_per_keyframe_median: f64,
    pub feat_ms_median: f64,
    pub verify_ms_median: f64,
}
