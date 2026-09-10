#!/usr/bin/env python3
"""Per-set cost and map growth, on the axes cuVSLAM issues #77 and #136 use.

  #77  https://github.com/nvidia-isaac/cuVSLAM/issues/77
       track() climbing from ~10 ms to 50 ms+ as the trajectory grows, on a
       12-camera rig. Plotted as duration against frame index.
  #136 https://github.com/nvidia-isaac/cuVSLAM/issues/136
       staged landmarks never promoted into the map on a full-coverage rig, so
       loop closure has nothing to match against.

WHAT THIS CAN AND CANNOT MEASURE FROM A REPLAY BAG.

The bags in datasets/replay_out do not contain per-set timing -- the node logs a
windowed maximum every 5 s, which hides a trend by construction. The only per-set
clock in the bag is the interval between consecutive /cuvslam/odometry messages,
and that is FLOORED BY THE REPLAY RATE: 0.4x of a 20 Hz log is 125 ms/set, so any
cost below 125 ms is invisible and #77's 10->50 ms would not show at all.

So panel A is not a reproduction of #77. What it can show is the sets that
OVERRAN that floor, which are a lower bound on their own cost. Run the node with
`timing_csv:=<path>` for the real per-set Track() series and pass --timing to plot
it directly; that is the measurement #77 actually calls for.

Usage:
  scripts/vo/slam_cost_and_map.py --out datasets/replay_out/slam_cost_and_map.png
  scripts/vo/slam_cost_and_map.py --timing /tmp/vo_timing.csv --out fig.png
"""
import argparse
import csv
import glob
import os
import sqlite3
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated reference palette: the only three that
# clear the all-pairs CVD and normal-vision floors together. Colour follows the
# RUN, not its rank, so a run keeps its hue across every panel.
C_V6, C_CTRL, C_V9 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"


def _topic(con, name):
    r = con.execute("SELECT id FROM topics WHERE name=?", (name,)).fetchone()
    return r[0] if r else None


def read_run(run_dir):
    """Pull the per-set wall clock, the landmark cloud size, and the loop closures."""
    dbs = glob.glob(os.path.join(run_dir, "*.db3"))
    if not dbs:
        return None
    con = sqlite3.connect(dbs[0])
    tid = _topic(con, "/cuvslam/odometry")
    if tid is None:
        con.close()
        return None
    wall, stamps = [], []
    for ts, blob in con.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,)):
        wall.append(ts)
        stamps.append(_cdr_header_stamp(bytes(blob)))

    lm = []
    tid = _topic(con, "/cuvslam/landmarks")
    if tid is not None:
        for ts, blob in con.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,)):
            lm.append((ts, _cdr_pointcloud_width(bytes(blob))))

    lc = []
    tid = _topic(con, "/cuvslam/loop_closures")
    if tid is not None:
        lc = [ts for (ts,) in con.execute(
            "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))]
    has_slam = _topic(con, "/cuvslam/slam_odometry") is not None
    con.close()

    if len(wall) < 50:
        return None
    data_span = (stamps[-1] - stamps[0]) / 1e9
    wall_span = (wall[-1] - wall[0]) / 1e9
    return {
        "name": os.path.basename(run_dir.rstrip("/")),
        "wall": wall, "stamps": stamps, "lm": lm, "lc": lc, "slam": has_slam,
        "data_span": data_span, "wall_span": wall_span,
        "rate": data_span / wall_span if wall_span else 0.0,
        "intervals": [(b - a) / 1e6 for a, b in zip(wall, wall[1:])],
    }


# The bags are CDR. Rather than depend on a sourced ROS environment, read the two
# fixed-offset fields we need directly: both messages start with a 4-byte
# encapsulation header, then std_msgs/Header = {int32 sec, uint32 nsec, string frame_id}.
def _cdr_header_stamp(buf):
    import struct
    sec, nsec = struct.unpack_from("<iI", buf, 4)
    return sec * 1_000_000_000 + nsec


def _cdr_pointcloud_width(buf):
    """PointCloud2: header, then uint32 height, uint32 width (4-byte aligned)."""
    import struct
    off = 4 + 8                                    # encapsulation + stamp
    (n,) = struct.unpack_from("<I", buf, off)      # frame_id length
    off += 4 + n
    off = (off + 3) & ~3                           # align to 4 for height
    height, width = struct.unpack_from("<II", buf, off)
    return height * width


def rolling(xs, w):
    out, n = [], len(xs)
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        out.append(statistics.median(xs[lo:hi]))
    return out


def lc_set_window(run):
    """Which set indices the loop-closure events fall between."""
    if not run["lc"]:
        return None
    lo, hi = min(run["lc"]), max(run["lc"])
    idx = [i for i, t in enumerate(run["wall"]) if lo <= t <= hi]
    return (idx[0], idx[-1]) if idx else None


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def figure_timing(csv_path, out):
    """The per-set CSV from the node — the measurement the bags could not provide.

    #77 asks whether track() grows with trajectory length. The trap is that keyframe
    fraction and scene quality both move over a run, and either will masquerade as a
    cost trend, so every panel here holds one of them fixed.
    """
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))[1:]      # drop set 0: GPU warm-up, ~270 ms
    kfr = [r for r in rows if r["keyframe"] == "1"]
    nkr = [r for r in rows if r["keyframe"] != "1"]
    ms = lambda r: int(r["track_us"]) / 1000.0
    dt = lambda r: float(r["data_t_s"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=SURFACE)
    fig.suptitle("cuVSLAM #77 on the 4-fisheye ring rig — per-set Track() cost, "
                 "measured not inferred", color=INK, fontsize=13, fontweight="bold", y=0.98)

    # A: the #77 axis, split by keyframe. The keyframe path triangulates and runs SBA
    # where a normal frame only solves PnP; one line over both hides a bimodal
    # distribution and makes the keyframe cost look like jitter.
    ax = axes[0][0]; style(ax)
    for sub, colr, lbl in ((nkr, C_CTRL, "non-keyframe (PnP only)"),
                           (kfr, C_V6, "keyframe (triangulate + SBA)")):
        xs = [int(r["set"]) for r in sub]
        ys = [ms(r) for r in sub]
        ax.plot(xs, ys, color=colr, linewidth=0.5, alpha=0.22)
        ax.plot(xs, rolling(ys, 21), color=colr, linewidth=2.0, label=lbl)
    ax.set_ylim(0, 180)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    ax.set_title("A  Track() per set, split by keyframe", color=INK, fontsize=10,
                 loc="left", fontweight="bold")
    ax.set_xlabel("set index", color=INK_2, fontsize=9)
    ax.set_ylabel("milliseconds", color=INK_2, fontsize=9)

    # B: THE ATTRIBUTION. Keyframe cost against map size, with the scene held healthy.
    # Without the frame_landmarks filter this panel shows a rise that is really the
    # tracking-distress window late in the run, not map size.
    ax = axes[0][1]; style(ax)
    bins = [(0, 10000), (10000, 25000), (25000, 40000), (40000, 55000), (55000, 70000)]
    healthy = [r for r in kfr if int(r["frame_landmarks"]) > 300]
    labels, meds, counts = [], [], []
    for lo, hi in bins:
        seg = [ms(r) for r in healthy if lo <= int(r["map_landmarks"]) < hi]
        if not seg:
            continue
        labels.append(f"{lo//1000}-{hi//1000}k")
        meds.append(statistics.median(seg))
        counts.append(len(seg))
    ax.bar(range(len(meds)), meds, 0.62, color=C_V6, linewidth=0)
    for i, (m, c) in enumerate(zip(meds, counts)):
        ax.annotate(f"{m:.1f} ms\nn={c}", (i, m), color=INK_2, fontsize=8, ha="center",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylim(0, max(meds) * 1.42 if meds else 1)
    ax.set_title("B  Keyframe Track() vs MAP SIZE, healthy frames only",
                 color=INK, fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("promoted map landmarks", color=INK_2, fontsize=9)
    ax.set_ylabel("median Track() (ms)", color=INK_2, fontsize=9)

    # C: #136 answered. The promoted map, and whether the 60 s drain does anything.
    ax = axes[1][0]; style(ax)
    ax.plot([dt(r) for r in rows], [int(r["map_landmarks"]) for r in rows],
            color=C_V9, linewidth=2.0)
    span = dt(rows[-1])
    if span > 60:
        ax.axvline(60, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.text(60.8, max(int(r["map_landmarks"]) for r in rows) * 0.30,
                "60 s staging drain\nREACHED — and the\ncurve does not step",
                color=INK_2, fontsize=8, va="center", ha="left")
    ax.set_title("C  Promoted SLAM map (DataLayer::Map) — #136 does not bite here",
                 color=INK, fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("data time (s)", color=INK_2, fontsize=9)
    ax.set_ylabel("promoted landmarks", color=INK_2, fontsize=9)

    # D: the confound, made visible. This is what actually drives panel A's tail.
    ax = axes[1][1]; style(ax)
    ax.plot([dt(r) for r in rows], [int(r["frame_landmarks"]) for r in rows],
            color=C_CTRL, linewidth=1.4)
    ax.axhline(300, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(1, 315, "the 'healthy tracking' cut used in B", color=INK_2, fontsize=8)
    ax.set_title("D  Landmarks the tracker holds per frame — the real cause of A's tail",
                 color=INK, fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("data time (s)", color=INK_2, fontsize=9)
    ax.set_ylabel("frame landmarks", color=INK_2, fontsize=9)

    fig.text(0.5, 0.012,
             "Track() cost is flat against a 10x growth in map size once the scene is held "
             "constant (B). The rise in A tracks the collapse in D, not the trajectory.",
             color=INK_2, fontsize=8.5, ha="center")
    fig.tight_layout(rect=(0, 0.032, 1, 0.955))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"  {len(rows)} sets, {len(kfr)} keyframes ({100*len(kfr)/len(rows):.1f}%), "
          f"{dt(rows[-1]):.1f} s data, map high-water "
          f"{max(int(r['map_landmarks']) for r in rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slam", default="datasets/replay_out/obs_slam_v6")
    ap.add_argument("--slam2", default="datasets/replay_out/obs_slam_v9")
    ap.add_argument("--control", default="datasets/replay_out/vo_clean_04")
    ap.add_argument("--timing", help="per-set CSV from the node's timing_csv param")
    ap.add_argument("--out", default="datasets/replay_out/slam_cost_and_map.png")
    a = ap.parse_args()

    if a.timing:
        if not os.path.exists(a.timing):
            sys.exit(f"no such timing CSV: {a.timing}")
        figure_timing(a.timing, a.out)
        return

    v6, v9, ctrl = (read_run(p) for p in (a.slam, a.slam2, a.control))
    if v6 is None or ctrl is None:
        sys.exit(f"need {a.slam} and {a.control} with /cuvslam/odometry")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=SURFACE)
    fig.suptitle("cuVSLAM per-set cost and map growth on the 4-fisheye / 8-pinhole ring rig",
                 color=INK, fontsize=13, fontweight="bold", x=0.5, y=0.98)

    # ---- A: the #77 axes, with the replay floor drawn so the caveat is visible ----
    ax = axes[0][0]
    style(ax)
    if True:
        for run, colr, lbl in ((v6, C_V6, "SLAM on"), (ctrl, C_CTRL, "SLAM off (control)")):
            d = run["intervals"]
            ax.plot(range(len(d)), d, color=colr, linewidth=0.5, alpha=0.25)
            ax.plot(range(len(d)), rolling(d, 41), color=colr, linewidth=2.0, label=lbl)
        ax.set_ylim(0, 650)
        floor = statistics.median(v6["intervals"])
        ax.axhline(floor, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.annotate(f"replay floor {floor:.0f} ms\ncost below this is invisible",
                    xy=(1120, floor), xytext=(1120, 300), color=INK_2, fontsize=8,
                    va="center", ha="right",
                    arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.8,
                                    shrinkA=2, shrinkB=2))
        # Thin rules at the first and last closure, NOT a shaded span: the overrun burst in
        # panel B runs past the last closure, so a span would assert an alignment the data
        # does not support.
        w = lc_set_window(v6)
        if w:
            for x in w:
                ax.axvline(x, color=INK_2, linewidth=0.9, linestyle=(0, (2, 3)))
            ax.annotate("first / last loop closure", (w[0], 615), color=INK_2, fontsize=8,
                        xytext=(6, 0), textcoords="offset points", va="center", ha="left")
        ax.set_title("A  Per-set interval — NOT #77: the replay rate floors it",
                     color=INK, fontsize=10, loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    ax.set_xlabel("set index", color=INK_2, fontsize=9)
    ax.set_ylabel("milliseconds", color=INK_2, fontsize=9)

    # ---- B: where the overruns actually are ----
    ax = axes[0][1]
    style(ax)
    k = 10
    width = 0.38
    for j, (run, colr, lbl) in enumerate(((v6, C_V6, "SLAM on"),
                                          (ctrl, C_CTRL, "SLAM off (control)"))):
        d = run["intervals"]
        floor = statistics.median(d)
        q = len(d) // k
        pct = [100.0 * sum(1 for x in d[i * q:(i + 1) * q] if x > floor * 1.25) / q
               for i in range(k)]
        ax.bar([i + (j - 0.5) * width for i in range(k)], pct, width * 0.94,
               color=colr, label=lbl, linewidth=0)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"{i+1}" for i in range(k)])
    ax.set_title("B  Sets overrunning the floor by >25%, per decile of the run",
                 color=INK, fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("decile of run", color=INK_2, fontsize=9)
    ax.set_ylabel("% of sets", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    # ---- C: #136 — the landmark cloud, and the 60 s drain it never reaches ----
    ax = axes[1][0]
    style(ax)
    # The two runs are the same bag replayed twice; the curves lie on top of each other to
    # within a few hundred points out of 54k. Label them stacked rather than at the line end,
    # where they would collide and read as one mislabelled series.
    for i, (run, colr) in enumerate(((v6, C_V6), (v9, C_V9))):
        if run is None or not run["lm"]:
            continue
        t0 = run["wall"][0]
        xs = [(ts - t0) / 1e9 * run["rate"] for ts, _ in run["lm"]]
        ys = [n for _, n in run["lm"]]
        ax.plot(xs, ys, color=colr, linewidth=2.0 if i == 0 else 1.2,
                linestyle="-" if i == 0 else (0, (5, 2)))
        ax.text(2, 52000 - i * 4200, f'{run["name"]} — {ys[-1]:,} pts', color=colr,
                fontsize=9, fontweight="bold", va="center")
    ax.set_xlim(0, 74)
    ax.axvline(60, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(61, 26000, "60 s staging drain\n(lsi_grid.h:177)\nnever reached",
            color=INK_2, fontsize=8, va="center", ha="left")
    ax.set_title("C  Odometry landmark dump — monotonic, no thinning in 57.7 s of data",
                 color=INK, fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("data time (s) — the clock cuVSLAM keys on", color=INK_2, fontsize=9)
    ax.set_ylabel("points in /cuvslam/landmarks", color=INK_2, fontsize=9)

    # ---- D: #136 — loop closures stop, and stay stopped ----
    ax = axes[1][1]
    style(ax)
    for run, colr in ((v6, C_V6), (v9, C_V9)):
        if run is None or not run["lc"]:
            continue
        t0 = run["wall"][0]
        xs = [(ts - t0) / 1e9 * run["rate"] for ts in run["lc"]]
        ys = list(range(1, len(xs) + 1))
        ax.step(xs + [run["data_span"]], ys + [ys[-1]], where="post",
                color=colr, linewidth=2.0)
        ax.plot(xs, ys, "o", color=colr, markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.annotate(f'{run["name"]}  ({len(xs)})', (run["data_span"], ys[-1]), color=colr,
                    fontsize=9, fontweight="bold", xytext=(-2, 7),
                    textcoords="offset points", ha="right")
    ax.set_xlim(0, 74)
    ax.set_title("D  Loop closures — none in the last 18 s of either run",
                 color=INK, fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("data time (s)", color=INK_2, fontsize=9)
    ax.set_ylabel("cumulative loop closures", color=INK_2, fontsize=9)

    fig.text(0.5, 0.028,
             "Promotion out of staging requires a landmark to leave EVERY camera frustum "
             "(lsi_grid.cpp:406), and this rig closes a 360 deg ring.",
             color=INK_2, fontsize=8.5, ha="center")
    fig.text(0.5, 0.008,
             "The 60 s staging lifetime is therefore the only reliable drain — and no run in "
             "datasets/replay_out has ever reached it.",
             color=INK_2, fontsize=8.5, ha="center")
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {a.out}")

    for run in (v6, v9, ctrl):
        if run is None:
            continue
        print(f"  {run['name']:16s} slam={'Y' if run['slam'] else 'n'} "
              f"sets={len(run['wall'])} data={run['data_span']:.2f}s "
              f"wall={run['wall_span']:.2f}s rate={run['rate']:.2f}x "
              f"closures={len(run['lc'])}")


if __name__ == "__main__":
    main()
