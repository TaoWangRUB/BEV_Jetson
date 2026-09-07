#!/usr/bin/env python3
"""What a gain setting did to the images: clipping, and whether any texture survives.

  exposure_report.py <raw_log_dir> [--label TEXT]

Reports per camera, over the middle of the clip:

  mean          overall level. Not the thing to optimise - see below.
  % at white    the clipped fraction. THIS is what kills tracking: a clipped region is one
                constant value, and no amount of post-processing recovers a constant.
                Detected as the modal value when the mode sits high, NOT as ">= 250" -
                this pipeline emits limited range and never reaches 250. On run1 the white
                level was 227 and the >=250 test reported 0.0% saturation on a frame that
                was 89% clipped.
  local std     median 16x16 block standard deviation - the gradients cuVSLAM actually
                tracks. A well-exposed indoor frame runs 4-6; the blown frame ran 0.00.
  dead blocks   fraction of 16x16 blocks with std < 2, i.e. nothing to track. 16% is
                normal indoors, 88% was the failure.

PICK THE GAIN with % at white ~ 0 in the BRIGHT room and local std still healthy in the
DIM one. Prefer the darker option when they conflict: an underexposed frame with gradients
still tracks, a clipped one cannot.
"""
import sys, argparse, pathlib, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("log_dir")
ap.add_argument("--label", default="")
ap.add_argument("--frames", type=int, default=5)
a = ap.parse_args()

d = pathlib.Path(a.log_dir)
g = dict(l.split() for l in (d / "geometry.txt").read_text().splitlines() if len(l.split()) == 2)
W, H = int(g["width"]), int(g["height"])
cams = sorted(p.stem for p in d.glob("cam?.raw"))
print(f"  {a.label or d.name}")
print(f"    {'cam':6s} {'mean':>7s} {'% at white':>11s} {'local std':>10s} {'dead blk':>9s}")
worst_clip, worst_flat = 0.0, 0.0
for c in cams:
    m = np.memmap(d / f"{c}.raw", dtype="u1", mode="r").reshape(-1, H, W)
    n = m.shape[0]
    idx = np.linspace(n * 0.3, n * 0.7, a.frames).astype(int)   # skip start/stop transients
    mus, clips, stds, deads = [], [], [], []
    for i in idx:
        f = np.asarray(m[i])
        hist = np.bincount(f.ravel(), minlength=256)
        mode = int(hist.argmax())
        # Only call it clipping when the pile-up sits in the top of the range; a dark scene
        # legitimately has its mode low and is not clipped.
        clip = 100.0 * hist[mode] / f.size if mode >= 200 else 0.0
        b = f[:H // 16 * 16, :W // 16 * 16].reshape(H // 16, 16, W // 16, 16).std(axis=(1, 3))
        mus.append(f.mean()); clips.append(clip)
        stds.append(np.median(b)); deads.append(100.0 * (b < 2).mean())
    print(f"    {c:6s} {np.mean(mus):7.1f} {np.mean(clips):10.1f}% {np.mean(stds):10.2f} "
          f"{np.mean(deads):8.0f}%")
    worst_clip = max(worst_clip, float(np.mean(clips)))
    worst_flat = max(worst_flat, float(np.mean(deads)))
verdict = ("CLIPPED - unusable" if worst_clip > 30 else
           "clipping starts" if worst_clip > 5 else
           "no clipping")
print(f"    -> worst clip {worst_clip:.0f}%, worst dead {worst_flat:.0f}%  [{verdict}]")
