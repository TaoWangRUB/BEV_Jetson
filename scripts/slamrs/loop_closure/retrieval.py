"""Loop-candidate retrieval: spatial (cuVSLAM's query) and appearance (HashBow), both time-gated."""
import numpy as np


class Retriever:
    def __init__(self, hashbow, cfg):
        self.hashbow = hashbow
        self.cfg = cfg
        self.t = []          # keyframe stamps, ns
        self.p = []          # keyframe VIO positions (world)

    def add(self, index, kf, feats):
        assert index == len(self.t)
        self.t.append(kf.t_ns)
        self.p.append(kf.T_vio[:3, 3].copy())
        self.hashbow.add(index, feats.bow)

    def candidates(self, index, feats):
        """[(candidate index, route, score)] for keyframe `index`; spatial first, then appearance,
        capped at max_candidates. Nothing younger than min_separation_s."""
        c = self.cfg
        t_now = self.t[index]
        min_sep = int(float(c["min_separation_s"]) * 1e9)
        old = np.array([i for i in range(index) if t_now - self.t[i] >= min_sep], dtype=np.int64)
        out, seen = [], set()
        if len(old):
            P = np.asarray(self.p)[old]
            d = np.linalg.norm(P - self.p[index], axis=1)
            near = old[d <= float(c["spatial_radius_m"])]
            for i in near[np.argsort(d[d <= float(c["spatial_radius_m"])], kind="stable")]:
                out.append((int(i), "spatial", float(np.linalg.norm(self.p[i] - self.p[index]))))
                seen.add(int(i))
        oldset = set(int(i) for i in old)
        for fid, score in self.hashbow.query(feats.bow, int(c["bow_top_k"]), keep=lambda f: f in oldset):
            if score >= float(c["bow_min_score"]) and fid not in seen:
                out.append((fid, "appearance", float(score)))
                seen.add(fid)
        return out[: int(c["max_candidates"])]
