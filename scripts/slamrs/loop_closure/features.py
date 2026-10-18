"""ORB per camera, landmark association, and the hash bag-of-words.

ORB is Basalt's choice for the mapping layer ("distinct from VIO points"); the VIO's own KLT points
carry no descriptor. A keypoint gets a *depth* by association with the nearest VIO landmark that
projects within `assoc_px` of it at the keyframe's own pose - the equivalent of cuVSLAM tracking its
stored landmark patches, done here with the descriptor instead of the patch.

HashBow is Basalt's retrieval structure (include/basalt/hash_bow/hash_bow.h): a word is a fixed
subset of the descriptor's bits, a frame is an L1-normalised word histogram, and the score is the
L1 overlap. No vocabulary, deterministic, microseconds per query.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class CamFeatures:
    kp: np.ndarray        # (n, 2) pixels
    desc: np.ndarray      # (n, 32) uint8
    bearing: np.ndarray   # (n, 3) unit, camera frame
    lm: np.ndarray        # (n,) index into the keyframe's landmark arrays, -1 if none


@dataclass
class KeyframeFeatures:
    cams: list            # CamFeatures per camera
    lm_pos_rig: np.ndarray  # landmarks in the keyframe's own rig (IMU) frame, (m, 3)
    bow: dict             # word -> normalised weight
    n_assoc: int          # keypoints with a landmark, all cameras


class HashBow:
    def __init__(self, num_bits=16, seed=0):
        self.num_bits = min(int(num_bits), 32)
        self.bits = np.random.RandomState(seed).permutation(256)[: self.num_bits]
        self.weights = (1 << np.arange(self.num_bits)).astype(np.int64)
        self.index = {}   # word -> list of (frame id, weight)

    def words(self, desc):
        if len(desc) == 0:
            return np.zeros(0, np.int64)
        bits = np.unpackbits(np.asarray(desc, dtype=np.uint8), axis=1)[:, self.bits]
        return bits.astype(np.int64) @ self.weights

    def bow(self, desc):
        w = self.words(desc)
        if len(w) == 0:
            return {}
        u, c = np.unique(w, return_counts=True)
        c = c / c.sum()
        return {int(k): float(v) for k, v in zip(u, c)}

    def add(self, frame_id, bow):
        for w, v in bow.items():
            self.index.setdefault(w, []).append((frame_id, v))

    def query(self, bow, num_results, keep=None):
        """Top frames by L1 overlap sum(min(a_w, b_w)) in [0, 1]. `keep(frame_id)` filters."""
        scores = {}
        for w, v in bow.items():
            for fid, u in self.index.get(w, ()):
                if keep is not None and not keep(fid):
                    continue
                scores[fid] = scores.get(fid, 0.0) + min(v, u)
        res = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return res[:num_results]


def make_orb(cfg):
    return cv2.ORB_create(nfeatures=int(cfg["orb_features"]), fastThreshold=int(cfg["orb_fast_threshold"]))


def extract(kf, rig, orb, hashbow, cfg):
    """ORB on each camera, landmark association, bearings and the bag of words for one keyframe."""
    T_imu_world = np.linalg.inv(kf.T_vio)
    lm_rig = (T_imu_world[:3, :3] @ kf.lm_pos.T).T + T_imu_world[:3, 3] if len(kf.lm_pos) else np.zeros((0, 3))
    cams, descs, n_assoc = [], [], 0
    for k, (cam, im) in enumerate(zip(rig.cams, kf.images)):
        kps, desc = orb.detectAndCompute(im, None)
        if desc is None or len(kps) == 0:
            cams.append(CamFeatures(np.zeros((0, 2)), np.zeros((0, 32), np.uint8), np.zeros((0, 3)), np.zeros(0, np.int64)))
            continue
        kp = np.array([p.pt for p in kps], dtype=np.float64)
        order = np.lexsort((kp[:, 0], kp[:, 1]))        # deterministic order independent of ORB's
        kp, desc = kp[order], desc[order]
        bearing = cam.unproject(kp)
        lm = -np.ones(len(kp), np.int64)
        if len(lm_rig):
            Tci = rig.T_cam_imu[k]
            P = (Tci[:3, :3] @ lm_rig.T).T + Tci[:3, 3]
            uv, ok = cam.project(P)
            ok &= P[:, 2] > -0.05
            if ok.any():
                idx = np.nonzero(ok)[0]
                tree = cKDTree(uv[idx])
                d, j = tree.query(kp, distance_upper_bound=float(cfg["assoc_px"]))
                hit = np.isfinite(d)
                lm[hit] = idx[j[hit]]
                # one landmark per keypoint, and one keypoint per landmark: keep the nearest
                for l in np.unique(lm[hit]):
                    members = np.nonzero(lm == l)[0]
                    if len(members) > 1:
                        best = members[np.argmin(d[members])]
                        lm[members] = -1
                        lm[best] = l
        n_assoc += int((lm >= 0).sum())
        cams.append(CamFeatures(kp, desc, bearing, lm))
        descs.append(desc)
    bow = hashbow.bow(np.concatenate(descs) if descs else np.zeros((0, 32), np.uint8))
    return KeyframeFeatures(cams, lm_rig, bow, n_assoc)
