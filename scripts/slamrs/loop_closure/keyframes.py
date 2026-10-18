"""Keyframe records taken from the VIO, and their export.

The VIO's snapshot says which window frames are keyframes and which landmarks each hosts. A
keyframe's landmarks keep being refined while it sits in the window, so a record is *finalised*
when the estimator marginalises the keyframe (Basalt's mapper consumes marginalisation data for
the same reason). Until then the tracker refreshes its pose and landmarks on every frame.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import T_to_pose7, inv, pose7_to_T


@dataclass
class KeyframeRecord:
    t_ns: int
    T_vio: np.ndarray                 # world_from_imu at finalisation, 4x4
    images: list                      # 4 uint8 arrays (h, w), the frames the VIO saw; None once consumed
    lm_ids: np.ndarray                # landmark ids hosted by this keyframe
    lm_pos: np.ndarray                # their world positions (n, 3), last refined value
    index: int = -1                   # position in the backend's keyframe list, set by the backend
    features: object = field(default=None, repr=False)
    lm_age: np.ndarray = None         # snapshots each landmark stayed alive (a depth-quality proxy)
    lm_obs: dict = None               # camera -> (ids, pixels): the frontend's keypoints at keyframe time

    def __post_init__(self):
        if self.lm_age is None:
            self.lm_age = np.full(len(self.lm_ids), 10 ** 6, dtype=np.int64)


class KeyframeTracker:
    """Watch the VIO's snapshots; yield finalised KeyframeRecords in marginalisation order."""

    def __init__(self):
        self.pending = {}   # t_ns -> dict(images, pose7, lm_ids, lm_pos)
        self.seen = set()

    def update(self, t_ns, images, snap, flow=None):
        """Call after every `track()` with the snapshot and the flow frame. Returns the records
        finalised by this frame."""
        out = []
        if snap is None:
            return out
        kf_ids = set(int(k) for k in snap.kf_ids)
        if t_ns in kf_ids and t_ns not in self.pending and t_ns not in self.seen:
            obs = None
            if flow is not None and int(flow.t_ns) == int(t_ns):
                obs = {c: (np.asarray(flow.ids(c)).astype(np.int64), np.asarray(flow.positions(c), dtype=np.float64))
                       for c in range(flow.camera_count)}
            self.pending[t_ns] = dict(images=[im.copy() for im in images], pose7=None, lm={}, obs=obs)
            self.seen.add(t_ns)
        if self.pending:
            wt = np.asarray(snap.window_t_ns)
            wp = np.asarray(snap.window_poses)
            hosts = np.asarray(snap.landmark_hosts)
            lids = np.asarray(snap.landmark_ids)
            lpos = np.asarray(snap.landmark_positions)
            for tk, rec in self.pending.items():
                w = np.nonzero(wt == tk)[0]
                if not len(w):
                    continue          # out of the window: nothing in this snapshot is consistent with it
                rec["pose7"] = wp[w[0]].copy()
                T_iw = inv(pose7_to_T(rec["pose7"]))
                # every landmark this keyframe ever hosted (the VIO marginalises lost landmarks long
                # before it marginalises the keyframe, so the set alive at the end is a remnant),
                # stored in the HOST RIG FRAME at this same snapshot: the window re-optimises poses
                # and landmarks together, so a world position frozen at an earlier snapshot is
                # inconsistent with the keyframe's later pose (run5: 5.7 px median, 0.6 px once fixed)
                for i in np.nonzero(hosts == tk)[0]:
                    p = lpos[i]
                    if not np.all(np.isfinite(p)):
                        continue
                    e = rec["lm"].get(int(lids[i]))
                    rec["lm"][int(lids[i])] = (T_iw[:3, :3] @ p + T_iw[:3, 3], (e[1] + 1) if e else 1)
            for tm in (int(x) for x in snap.marginalized):
                if tm in self.pending:
                    out.append(self._finalise(tm))
        return out

    def flush(self):
        """End of log: finalise what is still in the window, oldest first."""
        return [self._finalise(t) for t in sorted(self.pending)]

    def _finalise(self, t):
        rec = self.pending.pop(t)
        if rec["pose7"] is None:
            raise RuntimeError(f"keyframe {t} never had a window pose")
        ids = np.array(sorted(rec["lm"]), dtype=np.int64)
        T = pose7_to_T(rec["pose7"])
        pos_rig = np.array([rec["lm"][i][0] for i in ids]).reshape(-1, 3)
        pos = pos_rig @ T[:3, :3].T + T[:3, 3]          # world frame, consistent with this final pose
        age = np.array([rec["lm"][i][1] for i in ids], dtype=np.int64)
        return KeyframeRecord(t, T, rec["images"], ids, pos, lm_age=age, lm_obs=rec["obs"])


def export_keyframe(out_dir, kf):
    """Write one keyframe for inspection: PNGs, pose and landmarks. Deterministic file names."""
    d = Path(out_dir) / f"kf_{kf.t_ns}"
    d.mkdir(parents=True, exist_ok=True)
    for k, im in enumerate(kf.images):
        cv2.imwrite(str(d / f"cam{k + 1}.png"), im)
    np.savetxt(d / "landmarks.csv", np.column_stack([kf.lm_ids, kf.lm_pos, kf.lm_age]), delimiter=",",
               header="id,x,y,z,age", comments="", fmt=["%d", "%.6f", "%.6f", "%.6f", "%d"])
    json.dump({"t_ns": kf.t_ns, "world_from_imu": T_to_pose7(kf.T_vio).tolist()}, open(d / "pose.json", "w"), indent=1)
    with open(d / "observations.csv", "w") as f:
        f.write("cam,id,u,v\n")
        for c, (ids, uv) in (kf.lm_obs or {}).items():
            for i, (u, v) in zip(ids, uv):
                f.write(f"{c},{i},{u:.3f},{v:.3f}\n")


def load_keyframes(out_dir):
    """Read an export back, oldest first."""
    out = []
    for d in sorted(Path(out_dir).glob("kf_*"), key=lambda p: int(p.name[3:])):
        meta = json.load(open(d / "pose.json"))
        with open(d / "landmarks.csv") as f:
            body = f.read().splitlines()[1:]
        lm = np.loadtxt(body, delimiter=",", ndmin=2) if body else np.zeros((0, 5))
        ims = [cv2.imread(str(d / f"cam{k + 1}.png"), cv2.IMREAD_GRAYSCALE) for k in range(4)]
        obs = None
        if (d / "observations.csv").exists():
            with open(d / "observations.csv") as f:
                ob = f.read().splitlines()[1:]
            o = np.loadtxt(ob, delimiter=",", ndmin=2) if ob else np.zeros((0, 4))
            obs = {c: (o[o[:, 0] == c, 1].astype(np.int64), o[o[:, 0] == c, 2:4]) for c in range(4)}
        out.append(KeyframeRecord(int(meta["t_ns"]), pose7_to_T(meta["world_from_imu"]), ims,
                                  lm[:, 0].astype(np.int64) if len(lm) else np.zeros(0, np.int64),
                                  lm[:, 1:4] if len(lm) else np.zeros((0, 3)),
                                  lm_age=lm[:, 4].astype(np.int64) if len(lm) else np.zeros(0, np.int64), lm_obs=obs))
    return out
