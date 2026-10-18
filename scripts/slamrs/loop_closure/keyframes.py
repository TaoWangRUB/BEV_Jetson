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

from .camera import T_to_pose7, pose7_to_T


@dataclass
class KeyframeRecord:
    t_ns: int
    T_vio: np.ndarray                 # world_from_imu at finalisation, 4x4
    images: list                      # 4 uint8 arrays (h, w), the frames the VIO saw; None once consumed
    lm_ids: np.ndarray                # landmark ids hosted by this keyframe
    lm_pos: np.ndarray                # their world positions (n, 3) at finalisation
    index: int = -1                   # position in the backend's keyframe list, set by the backend
    features: object = field(default=None, repr=False)


class KeyframeTracker:
    """Watch the VIO's snapshots; yield finalised KeyframeRecords in marginalisation order."""

    def __init__(self):
        self.pending = {}   # t_ns -> dict(images, pose7, lm_ids, lm_pos)
        self.seen = set()

    def update(self, t_ns, images, snap):
        """Call after every `track()`. Returns the records finalised by this frame."""
        out = []
        if snap is None:
            return out
        kf_ids = set(int(k) for k in snap.kf_ids)
        if t_ns in kf_ids and t_ns not in self.pending and t_ns not in self.seen:
            self.pending[t_ns] = dict(images=[im.copy() for im in images], pose7=None,
                                      lm_ids=np.zeros(0, np.int64), lm_pos=np.zeros((0, 3)))
            self.seen.add(t_ns)
        if self.pending:
            wt = np.asarray(snap.window_t_ns)
            wp = np.asarray(snap.window_poses)
            hosts = np.asarray(snap.landmark_hosts)
            lids = np.asarray(snap.landmark_ids)
            lpos = np.asarray(snap.landmark_positions)
            for tk, rec in self.pending.items():
                w = np.nonzero(wt == tk)[0]
                if len(w):
                    rec["pose7"] = wp[w[0]].copy()
                m = hosts == tk
                if m.any():
                    rec["lm_ids"], rec["lm_pos"] = lids[m].copy(), lpos[m].copy()
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
        return KeyframeRecord(t, pose7_to_T(rec["pose7"]), rec["images"], rec["lm_ids"], rec["lm_pos"])


def export_keyframe(out_dir, kf):
    """Write one keyframe for inspection: PNGs, pose and landmarks. Deterministic file names."""
    d = Path(out_dir) / f"kf_{kf.t_ns}"
    d.mkdir(parents=True, exist_ok=True)
    for k, im in enumerate(kf.images):
        cv2.imwrite(str(d / f"cam{k + 1}.png"), im)
    np.savetxt(d / "landmarks.csv", np.column_stack([kf.lm_ids, kf.lm_pos]), delimiter=",",
               header="id,x,y,z", comments="", fmt=["%d", "%.6f", "%.6f", "%.6f"])
    json.dump({"t_ns": kf.t_ns, "world_from_imu": T_to_pose7(kf.T_vio).tolist()}, open(d / "pose.json", "w"), indent=1)


def load_keyframes(out_dir):
    """Read an export back, oldest first."""
    out = []
    for d in sorted(Path(out_dir).glob("kf_*"), key=lambda p: int(p.name[3:])):
        meta = json.load(open(d / "pose.json"))
        lm = np.loadtxt(d / "landmarks.csv", delimiter=",", skiprows=1, ndmin=2)
        ims = [cv2.imread(str(d / f"cam{k + 1}.png"), cv2.IMREAD_GRAYSCALE) for k in range(4)]
        out.append(KeyframeRecord(int(meta["t_ns"]), pose7_to_T(meta["world_from_imu"]), ims,
                                  lm[:, 0].astype(np.int64) if len(lm) else np.zeros(0, np.int64),
                                  lm[:, 1:4] if len(lm) else np.zeros((0, 3))))
    return out
