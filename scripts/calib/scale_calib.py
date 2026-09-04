#!/usr/bin/env python3
"""Scale KANNALA_BRANDT camN.yaml intrinsics for a downscaled output resolution.

For a pure isotropic downscale by `scale` (e.g. 0.5), focal + principal point + image
size scale linearly; the equidistant distortion coeffs (k2..k5) are angle-based and
UNCHANGED. So no re-calibration is needed when the Argus ISP downscales the frame.

    python3 scripts/calib/scale_calib.py --in scripts/config/1640x1232 \
        --out scripts/config/820x616 --scale 0.5
"""

# ---------------------------------------------------------------------------
# ⚠ IMX219 / KANNALA-BRANDT LINEAGE — NOT PORTED TO THE IMX296 RIG (2026-09-04)
#
# This tool reads equidistant (KANNALA_BRANDT) intrinsics and the board_center
# rig format. Both belonged to the retired 4x IMX219 rig. The IMX219 intrinsics
# under scripts/config/ have been deleted and the rig files moved to
# config/rig/archive/imx219/, so this script has no valid input any more.
#
# The IMX296 rig is calibrated in omni/Mei (config/calib/imx296_1456x1088) with
# extrinsics in config/rig/rig_extrinsics_imx296.yaml. Porting means teaching this
# tool the Mei projection - bev_panorama_node.cpp has a reference implementation
# in mei_project(). Until then it exits rather than produce a wrong answer.
# ---------------------------------------------------------------------------
import sys as _sys
if "--i-know-this-is-imx219" not in _sys.argv:
    _sys.exit(__file__ + ": IMX219/KB lineage, not ported to the IMX296 rig. "
              "See the banner at the top of this file.")

import argparse
import os
import re


def scale_file(src, dst, sx, sy, to_w=None, to_h=None):
    with open(src) as f:
        txt = f.read()
    if to_w and to_h:  # target resolution: derive per-axis ratios from the source size
        srcw = int(re.search(r"image_width:\s*(\d+)", txt).group(1))
        srch = int(re.search(r"image_height:\s*(\d+)", txt).group(1))
        sx, sy = to_w / srcw, to_h / srch
    # x-axis params: mu, u0, image_width ; y-axis params: mv, v0, image_height ; k* unchanged
    txt = re.sub(r"image_width:\s*(\d+)",  lambda m: f"image_width: {to_w or int(round(int(m.group(1))*sx))}", txt)
    txt = re.sub(r"image_height:\s*(\d+)", lambda m: f"image_height: {to_h or int(round(int(m.group(1))*sy))}", txt)
    txt = re.sub(r"(mu|u0):\s*([-\d.]+)", lambda m: f"{m.group(1)}: {float(m.group(2))*sx:.10f}", txt)
    txt = re.sub(r"(mv|v0):\s*([-\d.]+)", lambda m: f"{m.group(1)}: {float(m.group(2))*sy:.10f}", txt)
    with open(dst, "w") as f:
        f.write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--scale", type=float, default=0.5, help="uniform scale (ignored if --to-width/-height given)")
    ap.add_argument("--to-width", type=int, default=None, help="target width (anisotropic ratios derived from source)")
    ap.add_argument("--to-height", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(a.indir)):
        if re.fullmatch(r"cam\d+\.yaml", fn):
            scale_file(os.path.join(a.indir, fn), os.path.join(a.outdir, fn),
                       a.scale, a.scale, a.to_width, a.to_height)
            n += 1
    tag = f"->{a.to_width}x{a.to_height}" if a.to_width else f"x{a.scale}"
    print(f"scaled {n} files from {a.indir} -> {a.outdir} ({tag})")


if __name__ == "__main__":
    main()
