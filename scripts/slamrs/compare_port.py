#!/usr/bin/env python3
"""Hold the Rust port (rust/slamrs-lc) to the Python backend's outputs on the same log.

    python3 scripts/slamrs/compare_port.py datasets/replay_out/lc_run6 datasets/replay_out/rs_lc_run6

Both directories come from the same raw log: the Python one from run_slamrs_lc.py, the Rust one
from `slamrs-lc replay`. The VIO rows must be byte-identical (same core, same inputs, pinned
dependency versions); the backend is held to edges and poses within tolerance, since its KLT
is not OpenCV's.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def read(path, **kw):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8", ndmin=1, **kw)


def first_to_last(lc):
    tr = lc[lc["status"] == 1]
    p = np.stack([tr["tx"], tr["ty"], tr["tz"]], axis=1)
    return float(np.linalg.norm(p[-1] - p[0])), p


def main(py_dir, rs_dir):
    py, rs = Path(py_dir), Path(rs_dir)
    print(f"python: {py}\nrust:   {rs}")
    # VIO rows
    a = (py / "slamrs_vio.csv").read_text().splitlines()
    b = (rs / "slamrs_vio.csv").read_text().splitlines()
    # the last five columns are per-run timings; compare the pose columns only
    cut = lambda lines: [",".join(l.split(",")[:18]) for l in lines]
    ca, cb = cut(a), cut(b)
    n = min(len(ca), len(cb))
    diff = [i for i in range(n) if ca[i] != cb[i]]
    print(f"VIO rows: python {len(a) - 1}, rust {len(b) - 1}; pose columns identical on {n - len(diff)}/{n} rows"
          + (f"; first difference at row {diff[0]}" if diff else ""))
    # keyframes
    ka, kb = read(py / "keyframes_lc.csv"), read(rs / "keyframes_lc.csv")
    same_t = len(ka) == len(kb) and np.array_equal(ka["t_ns"], kb["t_ns"])
    print(f"keyframes: python {len(ka)}, rust {len(kb)}, same stamps: {same_t}")
    if same_t:
        va = np.stack([ka["vio_tx"], ka["vio_ty"], ka["vio_tz"]], 1)
        vb = np.stack([kb["vio_tx"], kb["vio_ty"], kb["vio_tz"]], 1)
        la = np.stack([ka["lc_tx"], ka["lc_ty"], ka["lc_tz"]], 1)
        lb = np.stack([kb["lc_tx"], kb["lc_ty"], kb["lc_tz"]], 1)
        dv = np.linalg.norm(va - vb, axis=1)
        dl = np.linalg.norm(la - lb, axis=1)
        print(f"  VIO keyframe positions: max |diff| {dv.max():.2e} m")
        print(f"  loop-closed keyframe positions: median |diff| {np.median(dl):.4f} m, max {dl.max():.4f} m")
        print(f"  n_assoc: python median {np.median(ka['n_assoc']):.0f}, rust {np.median(kb['n_assoc']):.0f}")
    # closed trajectories
    fa, pa = first_to_last(read(py / "slamrs_lc.csv"))
    fb, pb = first_to_last(read(rs / "slamrs_lc.csv"))
    m = min(len(pa), len(pb))
    d = np.linalg.norm(pa[:m] - pb[:m], axis=1)
    print(f"closed trajectory: first-to-last python {fa:.3f} m, rust {fb:.3f} m; position |diff| median {np.median(d):.4f} m, max {d.max():.4f} m")
    # edges
    def edges(d):
        try:
            e = read(d / "loop_edges.csv")
        except (ValueError, IndexError):
            return {}
        return {(int(r["cur"]), int(r["cand"])): r for r in e}
    ea, eb = edges(py), edges(rs)
    long = lambda e: sum(1 for (c, k), r in e.items() if (r["t_cur_ns"] - r["t_cand_ns"]) > 60e9)
    common = sorted(set(ea) & set(eb))
    print(f"loop edges: python {len(ea)} ({long(ea)} over 60 s), rust {len(eb)} ({long(eb)} over 60 s), common {len(common)}")
    only_a, only_b = sorted(set(ea) - set(eb)), sorted(set(eb) - set(ea))
    if only_a:
        print(f"  python only: {only_a}")
    if only_b:
        print(f"  rust only:   {only_b}")
    if common:
        di = [int(eb[k]["inliers"]) - int(ea[k]["inliers"]) for k in common]
        dt = [int(eb[k]["tracked"]) - int(ea[k]["tracked"]) for k in common]
        print(f"  common edges: inliers rust-python median {np.median(di):+.0f}, tracked {np.median(dt):+.0f}")
    # hypotheses: reasons and KLT stage counts
    ha, hb = read(py / "hypotheses.csv"), read(rs / "hypotheses.csv")
    ra, rb = Counter(ha["reason"]), Counter(hb["reason"])
    grp = lambda c: Counter("low_ratio" if k.startswith("low_ratio") else k for k in c.elements())
    ga, gb = grp(ra), grp(rb)
    print(f"hypotheses: python {len(ha)}, rust {len(hb)}")
    for k in sorted(set(ga) | set(gb)):
        print(f"  {k:20s} python {ga.get(k, 0):4d}  rust {gb.get(k, 0):4d}")
    print("  KLT stages (median per hypothesis):")
    for col in ("n_sel", "n_proj", "n_klt", "n_fb", "n_err", "n_ncc", "n_weak", "tracked", "inliers2"):
        print(f"    {col:9s} python {np.median(ha[col]):7.1f}  rust {np.median(hb[col]):7.1f}")
    # timings
    ta, tb = read(py / "timings.csv"), read(rs / "timings.csv")
    va_, vb_ = ta[ta["candidates"] > 0]["verify_ms"], tb[tb["candidates"] > 0]["verify_ms"]
    print(f"verify ms median: python {np.median(va_):.1f}, rust {np.median(vb_):.1f}; total verify python {ta['verify_ms'].sum() / 1e3:.1f} s, rust {tb['verify_ms'].sum() / 1e3:.1f} s")
    sa, sb = json.loads((py / "summary.json").read_text()), json.loads((rs / "summary.json").read_text())
    print(f"summary accepted: python {sa['accepted']}, rust {sb['accepted']}; classes python {sa['landmark_classes']}, rust {sb['landmark_classes']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
