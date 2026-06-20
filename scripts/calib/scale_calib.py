#!/usr/bin/env python3
"""Scale KANNALA_BRANDT camN.yaml intrinsics for a downscaled output resolution.

For a pure isotropic downscale by `scale` (e.g. 0.5), focal + principal point + image
size scale linearly; the equidistant distortion coeffs (k2..k5) are angle-based and
UNCHANGED. So no re-calibration is needed when the Argus ISP downscales the frame.

    python3 scripts/calib/scale_calib.py --in scripts/config/1640x1232 \
        --out scripts/config/820x616 --scale 0.5
"""
import argparse
import os
import re


def scale_file(src, dst, s):
    with open(src) as f:
        txt = f.read()
    def mul_int(m):  return f"{m.group(1)}: {int(round(int(m.group(2)) * s))}"
    def mul_flt(m):  return f"{m.group(1)}: {float(m.group(2)) * s:.10f}"
    txt = re.sub(r"(image_width|image_height):\s*(\d+)", mul_int, txt)
    txt = re.sub(r"(mu|mv|u0|v0):\s*([-\d.]+)", mul_flt, txt)  # k2..k5 left as-is
    with open(dst, "w") as f:
        f.write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(a.indir)):
        if re.fullmatch(r"cam\d+\.yaml", fn):
            scale_file(os.path.join(a.indir, fn), os.path.join(a.outdir, fn), a.scale)
            n += 1
    print(f"scaled {n} files from {a.indir} -> {a.outdir} (x{a.scale})")


if __name__ == "__main__":
    main()
