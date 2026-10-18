"""The loop-closing backend: on_keyframe = retrieve -> verify -> accept -> optimise, bounded.

Synchronous here (design D6); the interface is what a thread would call. Every hypothesis,
accepted or not, is recorded with its reason. One accepted closure per keyframe, then a throttle
interval before the next verification (cuVSLAM's throttling_time_ms).
"""
import json
import time
from pathlib import Path

import numpy as np

from .camera import T_to_pose7, inv
from .features import HashBow, extract, make_orb
from .posegraph import PoseGraph4DoF
from .retrieval import Retriever
from .verify import verify


class LoopClosureBackend:
    def __init__(self, rig, cfg, out_dir=None, log=print):
        self.rig, self.cfg, self.log = rig, cfg, log
        self.out_dir = Path(out_dir) if out_dir else None
        self.orb = make_orb(cfg["features"])
        self.hashbow = HashBow(int(cfg["retrieval"]["bow_num_bits"]), int(cfg["verify"]["seed"]))
        self.retriever = Retriever(self.hashbow, cfg["retrieval"])
        self.graph = PoseGraph4DoF(cfg["graph"])
        self.min_z = float(cfg["features"]["min_bearing_z"])
        self.keyframes = []        # KeyframeRecord, index == position
        self.hypotheses = []
        self.timings = []
        self.edges = []            # (i, j, Hypothesis)
        self.last_accept_t = None
        self.dropped = []

    # -- the interface a thread would call -----------------------------------------------
    def on_keyframe(self, kf):
        t0 = time.perf_counter()
        kf.index = len(self.keyframes)
        kf.features = extract(kf, self.rig, self.orb, self.hashbow, self.cfg["features"])
        kf.images = None                           # consumed; the features are what we keep
        self.keyframes.append(kf)
        self.graph.add_node(kf.t_ns, kf.T_vio)
        self.retriever.add(kf.index, kf, kf.features)
        t_feat = time.perf_counter()
        row = dict(t_ns=kf.t_ns, index=kf.index, n_assoc=kf.features.n_assoc, candidates=0,
                   feat_ms=(t_feat - t0) * 1e3, retrieve_ms=0.0, verify_ms=0.0, optimize_ms=0.0, accepted=0)
        throttle = int(float(self.cfg["backend"]["throttle_s"]) * 1e9)
        if self.last_accept_t is not None and kf.t_ns - self.last_accept_t < throttle:
            row["note"] = "throttled"
            self.timings.append(row)
            return None
        cands = self.retriever.candidates(kf.index, kf.features)
        t_ret = time.perf_counter()
        row["candidates"], row["retrieve_ms"] = len(cands), (t_ret - t_feat) * 1e3
        accepted = None
        for ci, route, score in cands:
            kc = self.keyframes[ci]
            T_guess = inv(kf.T_vio) @ kc.T_vio          # odometry's T_cur_from_cand
            h = verify(ci, kf.index, route, kc.features, kf.features, self.rig, T_guess,
                       self.cfg["verify"], self.min_z)
            h.score = score
            self.hypotheses.append(h)
            if h.accepted:
                accepted = h
                break
        t_ver = time.perf_counter()
        row["verify_ms"] = (t_ver - t_ret) * 1e3
        if accepted is not None:
            self._accept(accepted)
            self.last_accept_t = kf.t_ns
            row["accepted"] = 1
        row["optimize_ms"] = (time.perf_counter() - t_ver) * 1e3
        self.timings.append(row)
        return accepted

    def _accept(self, h):
        e = self.graph.add_loop_edge(h.cur, h.cand, h.T_cur_from_cand, h.sigma_t, h.sigma_yaw)
        self.edges.append((h, e))
        self._solve_and_check()

    def _solve_and_check(self):
        chk = float(self.cfg["graph"]["edge_check_sigma"])
        while True:
            r = self.graph.optimize()
            loops = self.graph.loop_edges()
            if len(r) == 0 or r.max() <= chk:
                return
            worst = int(np.argmax(r))
            bad = loops[worst]
            self.log(f"  loop edge {bad.i}->{bad.j} residual {r.max():.1f} sigma > {chk}: dropped, re-solving")
            self.graph.drop_edge(bad)
            self.dropped.append((bad.i, bad.j, float(r.max())))
            self.edges = [(h, e) for h, e in self.edges if e is not bad]
            for h, _ in self.edges:
                pass

    # -- outputs --------------------------------------------------------------------------
    def finish(self):
        if self.keyframes and self.graph.loop_edges():
            self._solve_and_check()
        return self.graph.poses()

    def correct_frames(self, t_frames, T_frames):
        """Per-frame correction from the preceding keyframe (design D5). Returns (N,4,4)."""
        if not self.keyframes:
            return np.array(T_frames)
        kf_t = np.array([k.t_ns for k in self.keyframes])
        corr = self.graph.poses()
        out = np.empty((len(t_frames), 4, 4))
        for n, (t, T) in enumerate(zip(t_frames, T_frames)):
            k = int(np.searchsorted(kf_t, t, side="right") - 1)
            k = max(k, 0)
            A = corr[k] @ inv(self.keyframes[k].T_vio)
            out[n] = A @ T
        return out

    def write(self, out_dir=None):
        d = Path(out_dir or self.out_dir)
        d.mkdir(parents=True, exist_ok=True)
        corr = self.graph.poses()
        rows = []
        for k, kf in enumerate(self.keyframes):
            rows.append([kf.t_ns, k, *T_to_pose7(kf.T_vio), *T_to_pose7(corr[k]), kf.features.n_assoc])
        np.savetxt(d / "keyframes_lc.csv", np.array(rows, dtype=object), delimiter=",", fmt="%s",
                   header="t_ns,index,vio_tx,vio_ty,vio_tz,vio_qx,vio_qy,vio_qz,vio_qw,"
                          "lc_tx,lc_ty,lc_tz,lc_qx,lc_qy,lc_qz,lc_qw,n_assoc", comments="")
        with open(d / "loop_edges.csv", "w") as f:
            f.write("cur,cand,t_cur_ns,t_cand_ns,route,inliers,tracked,ratio,sigma_t,sigma_yaw_deg,best_pair,"
                    "sep_vio_m,sep_lc_m\n")
            for h, e in self.edges:
                kc, kk = self.keyframes[h.cand], self.keyframes[h.cur]
                sv = np.linalg.norm(kc.T_vio[:3, 3] - kk.T_vio[:3, 3])
                sl = np.linalg.norm(corr[h.cand][:3, 3] - corr[h.cur][:3, 3])
                f.write(f"{h.cur},{h.cand},{kk.t_ns},{kc.t_ns},{h.route},{h.inliers2},{h.tracked},{h.ratio:.3f},"
                        f"{h.sigma_t:.4f},{np.degrees(h.sigma_yaw):.3f},{h.best_pair[0] + 1}-{h.best_pair[1] + 1},"
                        f"{sv:.3f},{sl:.3f}\n")
        with open(d / "hypotheses.csv", "w") as f:
            f.write("cur,cand,route,score,accepted,reason,tracked,inliers1,inliers2,ratio,best_pair,"
                    "guess_diff_m,guess_diff_deg,pairs\n")
            for h in self.hypotheses:
                pairs = ";".join(f"{k}:{v}" for k, v in h.pair_counts.items())
                f.write(f"{h.cur},{h.cand},{h.route},{getattr(h, 'score', 0):.3f},{int(h.accepted)},{h.reason},"
                        f"{h.tracked},{h.inliers1},{h.inliers2},{h.ratio:.3f},{h.best_pair[0] + 1}-{h.best_pair[1] + 1},"
                        f"{h.guess_diff_m:.3f},{h.guess_diff_deg:.2f},{pairs}\n")
        keys = ["t_ns", "index", "n_assoc", "candidates", "feat_ms", "retrieve_ms", "verify_ms", "optimize_ms", "accepted"]
        with open(d / "timings.csv", "w") as f:
            f.write(",".join(keys) + ",note\n")
            for r in self.timings:
                f.write(",".join(f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k]) for k in keys)
                        + f",{r.get('note', '')}\n")
        summary = dict(keyframes=len(self.keyframes), hypotheses=len(self.hypotheses),
                       accepted=len(self.edges), dropped=self.dropped,
                       feat_ms_median=float(np.median([r["feat_ms"] for r in self.timings])) if self.timings else 0.0,
                       verify_ms_median=float(np.median([r["verify_ms"] for r in self.timings if r["candidates"]]))
                       if any(r["candidates"] for r in self.timings) else 0.0)
        json.dump(summary, open(d / "summary.json", "w"), indent=1)
        return summary
