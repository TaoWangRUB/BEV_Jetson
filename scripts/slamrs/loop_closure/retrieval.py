"""Loop-candidate retrieval: spatial (cuVSLAM's query) and appearance (HashBow), both time-gated.

A candidate must also lie at least `min_path_m` of travelled path behind the current keyframe
(design D1b): with the post-closure throttle off, a rig standing still would otherwise close
trivially against every keyframe it made at the same spot - correct edges that constrain nothing."""
import numpy as np


class Retriever:
    def __init__(self, hashbow, cfg, appearance=True):
        self.hashbow = hashbow
        self.cfg = cfg
        self.appearance = bool(appearance)     # the bag-of-words route needs descriptors (ORB on)
        self.t = []          # keyframe stamps, ns
        self.p = []          # keyframe VIO positions (world)
        self.s = []          # travelled path up to each keyframe, metres along the VIO trajectory

    def add(self, index, kf, feats):
        assert index == len(self.t)
        self.t.append(kf.t_ns)
        self.p.append(kf.T_vio[:3, 3].copy())
        self.s.append((self.s[-1] + float(np.linalg.norm(self.p[-1] - self.p[-2]))) if len(self.p) > 1 else 0.0)
        self.hashbow.add(index, feats.bow)

    def candidates(self, index, feats):
        """[(candidate index, route, score)] for keyframe `index`; spatial first, then appearance,
        capped at max_candidates. Nothing younger than min_separation_s."""
        c = self.cfg
        t_now = self.t[index]
        min_sep = int(float(c["min_separation_s"]) * 1e9)
        min_path = float(c.get("min_path_m", 0.0))
        old = np.array([i for i in range(index) if t_now - self.t[i] >= min_sep and self.s[index] - self.s[i] >= min_path],
                       dtype=np.int64)
        out, seen = [], set()
        if len(old):
            P = np.asarray(self.p)[old]
            d = np.linalg.norm(P - self.p[index], axis=1)
            near = old[d <= float(c["spatial_radius_m"])]
            for i in near[np.argsort(d[d <= float(c["spatial_radius_m"])], kind="stable")]:
                out.append((int(i), "spatial", float(np.linalg.norm(self.p[i] - self.p[index]))))
                seen.add(int(i))
        oldset = set(int(i) for i in old)
        bow_hits = self.hashbow.query(feats.bow, int(c["bow_top_k"]), keep=lambda f: f in oldset) if self.appearance else []
        for fid, score in bow_hits:
            if score >= float(c["bow_min_score"]) and fid not in seen:
                out.append((fid, "appearance", float(score)))
                seen.add(fid)
        return out[: int(c["max_candidates"])]
