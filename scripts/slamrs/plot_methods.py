"""Accuracy and compute of the two methods, slam-rs + loop closure against cuVSLAM SLAM, as one image.

  .venv/bin/python scripts/slamrs/plot_methods.py OUT.png \\
      --walk "walk 1" datasets/replay_out/rs_lc_run6 datasets/replay_out/raw_run6_slam0125_performance \\
      --walk "walk 2" datasets/replay_out/rs_lc_run5 datasets/replay_out/raw_run5_slam0125_performance

Each --walk names a slamrs-lc replay directory and a scripts/vo/replay_raw.sh SLAM directory of the
same raw log, so both methods saw the same frames, every one of them.

  accuracy  distance between the first and the last pose (cam1), for the odometry alone (light) and
            with loop closure (dark): slam-rs's VIO and its corrected frames (through the Kalibr
            T_imu_cam1); cuVSLAM's odometry and its final optimised path from the same run. The walks
            end near where they started; there is no tape ground truth, so this is an indicator.
  compute   per frame set, mean over the walk, stacked: tracking (light) + loop closure (dark), all
            run inline.
            slam-rs: imu + track + snapshot + keyframe tracker + backend from the replay's stage
            timers (replay.log in its directory; without it, track_ms + the backend's per-keyframe
            timings). cuVSLAM: Track() + Slam::Track() from odometry.csv. Image preprocessing is left
            out of both (cuVSLAM's fisheye-to-pinhole remap, slam-rs's 2x downscale).
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from compare import slamrs_cam1  # noqa: E402

# The two methods' colours are the video's: purple slam-rs loop-closed, green cuVSLAM SLAM
# (validated: CVD dE 25, normal-vision dE 38; green is 2.2:1 on white, so every bar carries its value).
METHODS = [("slam-rs + loop closure", "Rust, one CPU core", "#B65CE8"),
           ("cuVSLAM SLAM", "v17, on the GPU", "#52C46B")]
SURFACE, INK, MUTED, GRID = "#FFFFFF", "#1A1D24", "#5B6270", "#E6E8EE"


def first_to_last(p):
    return float(np.linalg.norm(p[-1] - p[0]))


def slamrs_numbers(d):
    """(odometry error, loop-closed error) in m and (tracking, loop closure) in ms per frame set."""
    s = np.genfromtxt(d / "slamrs_vio.csv", delimiter=",", names=True)
    l = np.genfromtxt(d / "slamrs_lc.csv", delimiter=",", names=True)
    err = (first_to_last(slamrs_cam1(s)[s["status"] == 1]), first_to_last(slamrs_cam1(l)[l["status"] == 1]))
    log = d / "replay.log"
    m = re.search(r"imu ([\d.]+)\s+track ([\d.]+)\s+snapshot ([\d.]+)\s+keyframe tracker ([\d.]+)\s+backend ([\d.]+)",
                  log.read_text()) if log.exists() else None
    if m:
        imu, track, snap, kft, backend = (float(x) for x in m.groups())
        tracking_s, lc_s = imu + track + snap + kft, backend
    else:
        t = np.genfromtxt(d / "timings.csv", delimiter=",", names=True)
        tracking_s = s["track_ms"].sum() / 1e3
        lc_s = sum(t[k].sum() for k in ("feat_ms", "retrieve_ms", "verify_ms", "optimize_ms")) / 1e3
    return err, (tracking_s * 1e3 / len(s), lc_s * 1e3 / len(s))


def cuvslam_numbers(d):
    """The same for a replay_raw.sh SLAM run: its odometry, its final optimised path, its timers."""
    o = np.genfromtxt(d / "odometry.csv", delimiter=",", names=True)
    ok = o["status"] == 1
    sp = np.genfromtxt(d / "slam_path.csv", delimiter=",", names=True)
    err = (first_to_last(np.stack([o["tx"], o["ty"], o["tz"]], -1)[ok]),
           first_to_last(np.stack([sp["tx"], sp["ty"], sp["tz"]], -1)))
    return err, (float(np.mean(o["track_us"])) / 1e3, float(np.mean(o["slam_us"])) / 1e3)


def tint(hex_colour, f=0.55):
    """The method's hue, mixed toward white: the lighter step of the same colour."""
    c = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)])
    return "#" + "".join(f"{int(round(v)):02X}" for v in c + (255 - c) * f)


def rounded_bar(ax, x, h, w, color, r_px=7):
    """A bar with a rounded data end and a square baseline (bars here are never zero)."""
    fig = ax.figure
    bbox = ax.get_window_extent()
    dx = np.diff(ax.get_xlim())[0] / bbox.width
    dy = np.diff(ax.get_ylim())[0] / bbox.height
    r = min(r_px * dx, w / 2)
    ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=dy / dx, facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x - w / 2, 0), w, min(h, r * dy / dx), facecolor=color, edgecolor="none", zorder=3))
    del fig


def rounded_bar_at(ax, x, y0, h, w, color, r_px=7):
    """A stacked segment with a rounded top, starting at y0."""
    bbox = ax.get_window_extent()
    dx = np.diff(ax.get_xlim())[0] / bbox.width
    dy = np.diff(ax.get_ylim())[0] / bbox.height
    r = min(r_px * dx, w / 2)
    ax.add_patch(FancyBboxPatch((x - w / 2, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=dy / dx, facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x - w / 2, y0), w, min(h, r * dy / dx), facecolor=color, edgecolor="none", zorder=3))


def panel(fig, rect, title, unit, walks, values, ymax, mode):
    ax = fig.add_axes(rect)
    ax.set_facecolor(SURFACE)
    n = len(walks)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, ymax)
    fig.canvas.draw()                        # so rounded_bar can convert pixels to data units
    w, off = 0.28, 0.17
    gap = 2 * np.diff(ax.get_ylim())[0] / ax.get_window_extent().height     # 2 px of surface between segments
    for i in range(n):
        for k, (_, _, col) in enumerate(METHODS):
            light, dark = values[i][k]
            x = i + (k - 0.5) * 2 * off
            if mode == "overlay":       # odometry alone behind, with loop closure in front
                rounded_bar(ax, x, light, w, tint(col))
                rounded_bar(ax, x, dark, w, col)
                # anchored on the bar's outer edge, so a wide label grows away from its neighbour
                ax.text(x + w / 2 if k == 0 else x - w / 2, light + ymax * 0.02, f"{light:.2f} \u2192 {dark:.2f} m",
                        ha="right" if k == 0 else "left", va="bottom", fontsize=22, color=INK, fontweight="semibold",
                        zorder=5)
            else:                       # tracking at the base, loop closure stacked on it
                ax.add_patch(Rectangle((x - w / 2, 0), w, light, facecolor=tint(col), edgecolor="none", zorder=3))
                rounded_bar_at(ax, x, light + gap, dark - gap, w, col)
                for y0, h, v in ((0, light, light), (light, dark, dark)):
                    if h > ymax * 0.07:
                        ax.text(x, y0 + h / 2, f"{v:.0f}", ha="center", va="center", fontsize=19, color=INK, zorder=4)
                ax.text(x, light + dark + ymax * 0.02, f"{light + dark:.0f} ms", ha="center", va="bottom", fontsize=24,
                        color=INK, fontweight="semibold")
    ax.set_xticks(range(n), walks, fontsize=21, color=INK)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", labelsize=17, colors=MUTED, length=0, pad=8)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#C9CED8")
    fig.text(rect[0], rect[1] + rect[3] + 0.035, title, fontsize=25, color=INK, fontweight="semibold", ha="left")
    fig.text(rect[0], rect[1] + rect[3] + 0.008, unit, fontsize=18, color=MUTED, ha="left")
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--walk", nargs=3, action="append", metavar=("LABEL", "SLAMRS_DIR", "CUVSLAM_DIR"), required=True)
    a = ap.parse_args()

    walks, acc, cost = [], [], []
    for label, sd, cd in a.walk:
        e_s, c_s = slamrs_numbers(Path(sd))
        e_c, c_c = cuvslam_numbers(Path(cd))
        walks.append(label)
        acc.append((e_s, e_c))
        cost.append((c_s, c_c))
        print(f"{label}: first-to-last slam-rs VIO {e_s[0]:.3f} -> LC {e_s[1]:.3f} m, cuVSLAM VO {e_c[0]:.3f} -> SLAM "
              f"{e_c[1]:.3f} m; per set slam-rs {c_s[0]:.1f} + {c_s[1]:.1f} = {sum(c_s):.1f} ms, cuVSLAM "
              f"{c_c[0]:.1f} + {c_c[1]:.1f} = {sum(c_c):.1f} ms")

    plt.rcParams["font.family"] = ["Noto Sans", "DejaVu Sans"]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=SURFACE)
    fig.text(0.05, 0.915, "slam-rs + loop closure vs cuVSLAM SLAM, same frames", fontsize=40, color=INK,
             fontweight="semibold")
    fig.text(0.05, 0.865, "Two indoor walks that end near where they started, every frame replayed on one laptop.",
             fontsize=21, color=MUTED)
    # legend: the colour sits beside ink-coloured text, never on it
    x = 0.05
    for name, note, col in METHODS:
        fig.patches.append(FancyBboxPatch((x, 0.792), 0.014, 0.025, boxstyle="round,pad=0,rounding_size=0.003",
                                          transform=fig.transFigure, facecolor=col, edgecolor="none"))
        t = fig.text(x + 0.022, 0.795, name, fontsize=21, color=INK, fontweight="semibold")
        fig.canvas.draw()
        w = t.get_window_extent().width / fig.bbox.width
        t2 = fig.text(x + 0.022 + w + 0.007, 0.795, note, fontsize=19, color=MUTED)
        fig.canvas.draw()
        x += 0.022 + w + 0.007 + t2.get_window_extent().width / fig.bbox.width + 0.04
    fig.text(0.05, 0.752, "In both charts the lighter part is the odometry alone, the darker part the loop closure.",
             fontsize=19, color=MUTED)
    panel(fig, [0.05, 0.215, 0.41, 0.455], "Distance between first and last pose",
          "metres, lower is better: odometry alone \u2192 with loop closure",
          walks, acc, max(max(e[0] for e in v) for v in acc) * 1.2, "overlay")
    panel(fig, [0.55, 0.215, 0.41, 0.455], "Compute per frame set",
          "milliseconds, mean over the walk, lower is better: tracking + loop closure",
          walks, cost, max(max(sum(c) for c in v) for v in cost) * 1.2, "stack")
    fig.text(0.05, 0.03,
             "Laptop: Intel i7-11850H, RTX A2000. slam-rs: Basalt visual-inertial odometry on all four fisheyes and the IMU, "
             "with a loop-closure backend. cuVSLAM 17: multi-camera visual odometry\n"
             "on 8 virtual pinhole views, with its SLAM layer. Compute is tracking plus loop closure, run inline; "
             "image preprocessing excluded. No tape ground truth: first-to-last is an indicator.\n"
             "Compute is one replay per walk with the GPU clock unlocked: cuVSLAM's timings moved by up to 2x between two "
             "replays of the same walk, slam-rs's by about 10%.",
             fontsize=15, color=MUTED, linespacing=1.5)
    fig.savefig(a.out, facecolor=SURFACE)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
