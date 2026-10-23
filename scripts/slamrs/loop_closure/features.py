"""ORB per camera, landmark association, and the hash bag-of-words.

ORB is Basalt's choice for the mapping layer ("distinct from VIO points"); the VIO's own KLT points
carry no descriptor. A keypoint gets a *depth* by association with the nearest VIO landmark that
projects within `assoc_px` of it at the keyframe's own pose - the equivalent of cuVSLAM tracking its
stored landmark patches, done here with the descriptor instead of the patch.

HashBow is Basalt's retrieval structure (include/basalt/hash_bow/hash_bow.h): a word is a fixed
subset of the descriptor's bits, a frame is an L1-normalised word histogram, and the score is the
L1 overlap. No vocabulary, deterministic, microseconds per query.

ORB is off by default (`features.orb_enabled`): cuVSLAM's design needs no descriptor - candidates
come from the pose and correspondences from KLT - and on run5/run6 the descriptor route found
no closure at all while costing 36-44 ms per keyframe. It stays as the switch for appearance
retrieval after drift beyond the spatial radius (design D1).
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CamFeatures:
    kp: np.ndarray        # (n, 2) detected keypoints, pixels
    desc: np.ndarray      # (n, 32) uint8
    bearing: np.ndarray   # (n, 3) unit, camera frame
    lm_kp: np.ndarray     # (m, 2) projections of the keyframe's landmarks into this camera
    lm_desc: np.ndarray   # (m, 32) ORB descriptors computed at those projections
    lm_idx: np.ndarray    # (m,) index into the keyframe's landmark arrays


@dataclass
class KeyframeFeatures:
    cams: list            # CamFeatures per camera
    lm_pos_rig: np.ndarray  # landmarks in the keyframe's own rig (IMU) frame, (m, 3)
    bow: dict             # word -> normalised weight ({} with ORB off)
    n_assoc: int          # landmarks with an observation at keyframe time, summed over cameras


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


_EMPTY = dict(kp=np.zeros((0, 2)), desc=np.zeros((0, 32), np.uint8), bearing=np.zeros((0, 3)),
              lm_kp=np.zeros((0, 2)), lm_desc=np.zeros((0, 32), np.uint8), lm_idx=np.zeros(0, np.int64))


def describe_at(orb, im, uv):
    """ORB descriptors at given pixel positions (cuVSLAM stores a patch per landmark; this is the
    descriptor equivalent). Returns (indices into uv that got a descriptor, descriptors)."""
    if len(uv) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 32), np.uint8)
    kps = [cv2.KeyPoint(float(u), float(v), 31.0, 0.0) for u, v in uv]   # angle 0: see extract()
    for n, k in enumerate(kps):
        k.class_id = n
    kps2, desc = orb.compute(im, kps)             # drops points too close to the border
    if desc is None or len(kps2) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 32), np.uint8)
    idx = np.array([k.class_id for k in kps2], dtype=np.int64)
    o = np.argsort(idx, kind="stable")
    return idx[o], desc[o]


def extract(kf, rig, orb, hashbow, cfg):
    """Landmark bookkeeping per camera and, with `orb` not None, ORB detections, descriptors at
    the landmark observations, bearings and the bag of words. With ORB off only the landmark
    observations are kept (what KLT verification needs); the bag of words is empty."""
    T_imu_world = np.linalg.inv(kf.T_vio)
    lm_rig = (T_imu_world[:3, :3] @ kf.lm_pos.T).T + T_imu_world[:3, 3] if len(kf.lm_pos) else np.zeros((0, 3))
    cams, descs, n_assoc = [], [], 0
    for k, (cam, im) in enumerate(zip(rig.cams, kf.images)):
        f = dict(_EMPTY)
        # Descriptors are computed UNROTATED on both sides (angle 0): the landmark side has no
        # corner orientation to rotate by, and on a ground rig roll is constant, so rotation
        # invariance buys nothing and would only make the two sides disagree.
        kps = orb.detect(im, None) if orb is not None else []
        for p in kps:
            p.angle = 0.0
        kps, desc = orb.compute(im, kps) if kps else ([], None)
        if desc is not None and len(kps):
            kp = np.array([p.pt for p in kps], dtype=np.float64)
            order = np.lexsort((kp[:, 0], kp[:, 1]))        # deterministic order independent of ORB's
            f["kp"], f["desc"] = kp[order], desc[order]
            f["bearing"] = cam.unproject(f["kp"])
            descs.append(f["desc"])
        if len(lm_rig):
            if kf.lm_obs is not None and k in kf.lm_obs:
                # the frontend's own observation of each landmark at keyframe time: the exact corner
                ids, uv_obs = kf.lm_obs[k]
                pos = {int(i): n for n, i in enumerate(kf.lm_ids)}
                min_age = int(cfg.get("min_landmark_age", 1))
                sel = np.array([n for n, i in enumerate(ids) if int(i) in pos and kf.lm_age[pos[int(i)]] >= min_age],
                               dtype=np.int64)
                idx = np.array([pos[int(ids[n])] for n in sel], dtype=np.int64)
                uv = uv_obs[sel] if len(sel) else np.zeros((0, 2))
            else:
                # no observations recorded (synthetic records): fall back to the projection
                Tci = rig.T_cam_imu[k]
                P = (Tci[:3, :3] @ lm_rig.T).T + Tci[:3, 3]
                uv_all, ok = cam.project(P)
                ok &= P[:, 2] > -0.05
                idx = np.nonzero(ok)[0]
                uv = uv_all[idx]
            n_assoc += len(idx)
            if orb is not None:
                got, d = describe_at(orb, im, uv)
                f["lm_kp"], f["lm_desc"], f["lm_idx"] = uv[got], d, idx[got]
            else:
                f["lm_kp"], f["lm_idx"] = np.asarray(uv, dtype=np.float64), np.asarray(idx, dtype=np.int64)
        cams.append(CamFeatures(**f))
    bow = hashbow.bow(np.concatenate(descs)) if (orb is not None and descs) else {}
    return KeyframeFeatures(cams, lm_rig, bow, n_assoc)
