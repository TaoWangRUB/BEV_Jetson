"""Keyframe records taken from the VIO, and their export.

The VIO's snapshot says which window frames are keyframes and which landmarks each hosts. A
keyframe's landmarks keep being refined while it sits in the window, so a record is *finalised*
when the estimator marginalises the keyframe (Basalt's mapper consumes marginalisation data for
the same reason). Until then the tracker refreshes its pose and landmarks on every frame.

A record carries every landmark the keyframe OBSERVED (the frontend's keypoint ids at keyframe
time, as soon as they have a 3D position in some later snapshot), not only the ones it hosts: a
keyframe hosts the landmarks born at it (~45 per keyframe on run5) but sees two to six times as
many, and it is the observed set that a loop candidate can be verified against (design D2b).

Each landmark carries a depth confidence (design D2c): the parallax its observations have
subtended so far (the angle at the landmark between the first and the latest rig position that
saw it), how many frames saw it, and the baseline between those positions. A landmark whose
depth was frozen at low parallax fits its own observations at any depth, so it is reliable as a
direction but not as a point; the verifier treats it as bearing-only. After a keyframe is
finalised its landmarks live on in the window, so the tracker keeps the record open: a later
position replaces the stored one only if the keyframe's own observation still projects within
`refresh_max_px` under the frozen pose (the D2a consistency rule applied forward in time), and the
confidence keeps growing with every frame that observes the landmark.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import T_to_pose7, inv, pose7_to_T

CLASS_3D, CLASS_FAR, CLASS_WEAK = 0, 1, 2


@dataclass
class KeyframeRecord:
    t_ns: int
    T_vio: np.ndarray                 # world_from_imu at finalisation, 4x4
    images: list                      # 4 uint8 arrays (h, w), the frames the VIO saw; None once consumed
    lm_ids: np.ndarray                # landmark ids this keyframe hosted or observed
    lm_pos: np.ndarray                # their world positions (n, 3), consistent with T_vio
    index: int = -1                   # position in the backend's keyframe list, set by the backend
    features: object = field(default=None, repr=False)
    lm_age: np.ndarray = None         # snapshots each landmark stayed alive while this record was pending
    lm_obs: dict = None               # camera -> (ids, pixels): the frontend's keypoints at keyframe time
    lm_hosted: np.ndarray = None      # bool per landmark: hosted by this keyframe (else observed only)
    lm_parallax_deg: np.ndarray = None  # angle at the landmark between the first and latest observing rig positions
    lm_nobs: np.ndarray = None        # frames that observed the landmark (any camera), up to the latest refresh
    lm_baseline: np.ndarray = None    # metres between those two rig positions
    lm_refreshed: np.ndarray = None   # times the position was replaced from a later snapshot
    dirty: bool = False               # positions changed since the features' rig-frame copy was made
    obs_map: dict = field(default=None, repr=False)   # landmark id -> (camera, observed pixel), for the refresh check

    def __post_init__(self):
        n = len(self.lm_ids)
        if self.lm_age is None:
            self.lm_age = np.full(n, 10 ** 6, dtype=np.int64)
        if self.lm_hosted is None:
            self.lm_hosted = np.ones(n, dtype=bool)
        if self.lm_parallax_deg is None:          # records without a tracker (synthetic): fully confident
            self.lm_parallax_deg = np.full(n, 90.0)
        if self.lm_nobs is None:
            self.lm_nobs = np.full(n, 10 ** 6, dtype=np.int64)
        if self.lm_baseline is None:
            self.lm_baseline = np.full(n, 10 ** 6, dtype=np.float64)
        if self.lm_refreshed is None:
            self.lm_refreshed = np.zeros(n, dtype=np.int64)
        if self.obs_map is None:
            self.obs_map = {}
            for c, (ids, uv) in (self.lm_obs or {}).items():
                for k, i in enumerate(ids):
                    self.obs_map.setdefault(int(i), (int(c), np.asarray(uv[k], dtype=np.float64)))

    def depth_class(self, cfg):
        """Per landmark: CLASS_3D (parallax and observations enough for a point), CLASS_FAR (seen
        enough but with no parallax: a direction), CLASS_WEAK (neither; not used)."""
        min_obs = int(cfg.get("min_obs", 1))
        enough = self.lm_nobs >= min_obs
        cls = np.full(len(self.lm_ids), CLASS_WEAK, dtype=np.int64)
        cls[enough & (self.lm_parallax_deg >= float(cfg.get("min_parallax_deg", 0.0)))] = CLASS_3D
        cls[enough & (self.lm_parallax_deg < float(cfg.get("far_parallax_deg", 0.0)))] = CLASS_FAR
        return cls

    def lm_pos_rig(self):
        T_iw = inv(self.T_vio)
        return self.lm_pos @ T_iw[:3, :3].T + T_iw[:3, 3]


def _parallax(p_first, p_last, L):
    """Degrees subtended at landmark L (n,3) by rig positions p_first and p_last (n,3)."""
    a, b = p_first - L, p_last - L
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    c = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


class KeyframeTracker:
    """Watch the VIO's snapshots; yield finalised KeyframeRecords in marginalisation order, and keep
    the finalised records' landmarks current while the VIO still observes them."""

    def __init__(self, rig=None, cfg=None):
        self.pending = {}   # t_ns -> dict(images, pose7, lm, obs, obs_ids)
        self.seen = set()
        self.done = []      # finalised records, the same objects the backend holds
        self.table = {}     # landmark id -> [p_first (3,), p_last (3,), n_obs]: where and how often it was seen
        self.rig = rig
        self.cfg = dict(cfg or {})
        self.refresh = bool(self.cfg.get("refresh_enabled", False)) and rig is not None
        self.refresh_max_px = float(self.cfg.get("refresh_max_px", 2.0))

    def update(self, t_ns, images, snap, flow=None):
        """Call after every `track()` with the snapshot and the flow frame. Returns the records
        finalised by this frame."""
        out = []
        if snap is None:
            return out
        wt = np.asarray(snap.window_t_ns)
        wp = np.asarray(snap.window_poses)
        hosts = np.asarray(snap.landmark_hosts)
        lids = np.asarray(snap.landmark_ids)
        lpos = np.asarray(snap.landmark_positions)
        # 1. the observation table: every id the frontend tracked on this frame, seen from this
        #    frame's rig position (its window pose)
        w_now = np.nonzero(wt == t_ns)[0]
        if flow is not None and int(flow.t_ns) == int(t_ns) and len(w_now):
            p_now = wp[w_now[0]][:3].copy()
            seen_now = [np.asarray(flow.ids(c)) for c in range(flow.camera_count)]
            seen_now = np.unique(np.concatenate(seen_now)) if seen_now else np.zeros(0, np.int64)
            for i in seen_now.tolist():
                e = self.table.get(int(i))
                if e is None:
                    self.table[int(i)] = [p_now, p_now, 1]
                else:
                    e[1] = p_now
                    e[2] += 1
        # 2. a new keyframe starts a pending record
        kf_ids = set(int(k) for k in snap.kf_ids)
        if t_ns in kf_ids and t_ns not in self.pending and t_ns not in self.seen:
            obs = None
            if flow is not None and int(flow.t_ns) == int(t_ns):
                obs = {c: (np.asarray(flow.ids(c)).astype(np.int64), np.asarray(flow.positions(c), dtype=np.float64))
                       for c in range(flow.camera_count)}
            obs_ids = (np.unique(np.concatenate([ids for ids, _ in obs.values()])).astype(np.int64)
                       if obs else np.zeros(0, np.int64))
            self.pending[t_ns] = dict(images=[im.copy() for im in images], pose7=None, lm={}, obs=obs, obs_ids=obs_ids)
            self.seen.add(t_ns)
        # 3. pending records follow the window
        if self.pending:
            for tk, rec in self.pending.items():
                w = np.nonzero(wt == tk)[0]
                if not len(w):
                    continue          # out of the window: nothing in this snapshot is consistent with it
                rec["pose7"] = wp[w[0]].copy()
                T_iw = inv(pose7_to_T(rec["pose7"]))
                # every landmark this keyframe ever hosted (the VIO marginalises lost landmarks long
                # before it marginalises the keyframe, so the set alive at the end is a remnant)
                # PLUS every landmark it observed at keyframe time that has a position in this
                # snapshot, whoever hosts it (D2b) - all stored in THIS keyframe's rig frame at this
                # same snapshot: the window re-optimises poses and landmarks together, so a world
                # position frozen at an earlier snapshot is inconsistent with the keyframe's later
                # pose (run5: 5.7 px median, 0.6 px once fixed)
                hosted = hosts == tk
                sel = hosted | np.isin(lids, rec["obs_ids"])
                for i in np.nonzero(sel)[0]:
                    p = lpos[i]
                    if not np.all(np.isfinite(p)):
                        continue
                    e = rec["lm"].get(int(lids[i]))
                    rec["lm"][int(lids[i])] = (T_iw[:3, :3] @ p + T_iw[:3, 3], (e[1] + 1) if e else 1,
                                               bool(hosted[i]) or (e[2] if e else False))
            for tm in (int(x) for x in snap.marginalized):
                if tm in self.pending:
                    out.append(self._finalise(tm))
        # 4. finalised records stay open while the VIO still observes their landmarks
        if self.done and len(lids):
            self._refresh_done(lids, lpos)
        return out

    def flush(self):
        """End of log: finalise what is still in the window, oldest first."""
        return [self._finalise(t) for t in sorted(self.pending)]

    def _confidence(self, ids, L):
        """Parallax, observation count and baseline for landmark ids with positions L (n,3)."""
        n = len(ids)
        pf, pl, nobs = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n, np.int64)
        have = np.zeros(n, bool)
        for k, i in enumerate(ids.tolist()):
            e = self.table.get(int(i))
            if e is not None:
                pf[k], pl[k], nobs[k], have[k] = e[0], e[1], e[2], True
        par = np.zeros(n)
        par[have] = _parallax(pf[have], pl[have], L[have])
        base = np.linalg.norm(pl - pf, axis=1)
        return par, nobs, base

    def _finalise(self, t):
        rec = self.pending.pop(t)
        if rec["pose7"] is None:
            raise RuntimeError(f"keyframe {t} never had a window pose")
        ids = np.array(sorted(rec["lm"]), dtype=np.int64)
        T = pose7_to_T(rec["pose7"])
        pos_rig = np.array([rec["lm"][i][0] for i in ids]).reshape(-1, 3)
        pos = pos_rig @ T[:3, :3].T + T[:3, 3]          # world frame, consistent with this final pose
        age = np.array([rec["lm"][i][1] for i in ids], dtype=np.int64)
        hosted = np.array([rec["lm"][i][2] for i in ids], dtype=bool)
        par, nobs, base = self._confidence(ids, pos)
        kf = KeyframeRecord(t, T, rec["images"], ids, pos, lm_age=age, lm_obs=rec["obs"], lm_hosted=hosted,
                            lm_parallax_deg=par, lm_nobs=nobs, lm_baseline=base)
        self.done.append(kf)
        return kf

    def _refresh_done(self, lids, lpos):
        """For every finalised record, update the confidence of its landmarks still alive and, with
        refresh on, replace positions that stay consistent with the record's own observation."""
        for rec in self.done:
            alive = np.isin(rec.lm_ids, lids)
            if not alive.any():
                continue
            idx = np.nonzero(alive)[0]
            j = np.searchsorted(lids, rec.lm_ids[idx]) if np.all(np.diff(lids) > 0) else \
                np.array([np.nonzero(lids == i)[0][0] for i in rec.lm_ids[idx]])
            p_new = lpos[j]
            finite = np.all(np.isfinite(p_new), axis=1)
            idx, p_new = idx[finite], p_new[finite]
            if not len(idx):
                continue
            moved = np.linalg.norm(p_new - rec.lm_pos[idx], axis=1) > 1e-9    # a refresh is a change
            par, nobs, base = self._confidence(rec.lm_ids[idx], rec.lm_pos[idx])
            rec.lm_parallax_deg[idx], rec.lm_nobs[idx], rec.lm_baseline[idx] = par, nobs, base
            idx, p_new = idx[moved], p_new[moved]
            if not len(idx) or not self.refresh:
                continue
            # the consistency rule, forward in time: the keyframe's own observation of the landmark
            # must still project within refresh_max_px of the new position under the frozen pose
            T_iw = inv(rec.T_vio)
            P_rig = p_new @ T_iw[:3, :3].T + T_iw[:3, 3]
            cams = np.array([rec.obs_map.get(int(i), (-1, None))[0] for i in rec.lm_ids[idx]])
            accept = np.zeros(len(idx), bool)
            for c in range(self.rig.n):
                m = cams == c
                if not m.any():
                    continue
                Tc = self.rig.T_cam_imu[c]
                Pc = P_rig[m] @ Tc[:3, :3].T + Tc[:3, 3]
                uv, ok = self.rig.cams[c].project(Pc)
                ok &= Pc[:, 2] > 0.02
                uv_obs = np.array([rec.obs_map[int(i)][1] for i in rec.lm_ids[idx[m]]])
                err = np.linalg.norm(uv - uv_obs, axis=1)
                accept[m] = ok & (err <= self.refresh_max_px)
            if accept.any():
                rec.lm_pos[idx[accept]] = p_new[accept]
                rec.lm_refreshed[idx[accept]] += 1
                rec.dirty = True
                if rec.features is not None:
                    rec.features.lm_pos_rig = rec.lm_pos_rig()
                    rec.dirty = False


def export_keyframe(out_dir, kf):
    """Write one keyframe for inspection: PNGs, pose and landmarks. Deterministic file names."""
    d = Path(out_dir) / f"kf_{kf.t_ns}"
    d.mkdir(parents=True, exist_ok=True)
    for k, im in enumerate(kf.images):
        cv2.imwrite(str(d / f"cam{k + 1}.png"), im)
    np.savetxt(d / "landmarks.csv",
               np.column_stack([kf.lm_ids, kf.lm_pos, kf.lm_age, kf.lm_hosted.astype(int), kf.lm_parallax_deg,
                                kf.lm_nobs, kf.lm_baseline, kf.lm_refreshed]),
               delimiter=",", header="id,x,y,z,age,hosted,parallax_deg,nobs,baseline,refreshed", comments="",
               fmt=["%d", "%.6f", "%.6f", "%.6f", "%d", "%d", "%.4f", "%d", "%.4f", "%d"])
    json.dump({"t_ns": kf.t_ns, "world_from_imu": T_to_pose7(kf.T_vio).tolist()}, open(d / "pose.json", "w"), indent=1)
    with open(d / "observations.csv", "w") as f:
        f.write("cam,id,u,v\n")
        for c, (ids, uv) in (kf.lm_obs or {}).items():
            for i, (u, v) in zip(ids, uv):
                f.write(f"{c},{i},{u:.3f},{v:.3f}\n")


def load_keyframes(out_dir):
    """Read an export back, oldest first. Older exports without the confidence columns load as
    fully confident."""
    out = []
    for d in sorted(Path(out_dir).glob("kf_*"), key=lambda p: int(p.name[3:])):
        meta = json.load(open(d / "pose.json"))
        with open(d / "landmarks.csv") as f:
            body = f.read().splitlines()[1:]
        lm = np.loadtxt(body, delimiter=",", ndmin=2) if body else np.zeros((0, 10))
        if lm.shape[1] == 5:                       # export without the hosted column
            lm = np.column_stack([lm, np.ones(len(lm))])
        ims = [cv2.imread(str(d / f"cam{k + 1}.png"), cv2.IMREAD_GRAYSCALE) for k in range(4)]
        obs = None
        if (d / "observations.csv").exists():
            with open(d / "observations.csv") as f:
                ob = f.read().splitlines()[1:]
            o = np.loadtxt(ob, delimiter=",", ndmin=2) if ob else np.zeros((0, 4))
            obs = {c: (o[o[:, 0] == c, 1].astype(np.int64), o[o[:, 0] == c, 2:4]) for c in range(4)}
        n = len(lm)
        extra = dict(lm_parallax_deg=lm[:, 6], lm_nobs=lm[:, 7].astype(np.int64), lm_baseline=lm[:, 8],
                     lm_refreshed=lm[:, 9].astype(np.int64)) if (n and lm.shape[1] >= 10) else {}
        out.append(KeyframeRecord(int(meta["t_ns"]), pose7_to_T(meta["world_from_imu"]), ims,
                                  lm[:, 0].astype(np.int64) if n else np.zeros(0, np.int64),
                                  lm[:, 1:4] if n else np.zeros((0, 3)),
                                  lm_age=lm[:, 4].astype(np.int64) if n else np.zeros(0, np.int64), lm_obs=obs,
                                  lm_hosted=lm[:, 5].astype(bool) if n else np.zeros(0, bool), **extra))
    return out
