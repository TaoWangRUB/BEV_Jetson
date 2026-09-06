#!/usr/bin/env python3
"""Convert a raw 4-camera image log into a rosbag2, offline on the host.

WHY THIS EXISTS. Recording straight to a bag ON THE BOARD tops out at 6-7 Hz per camera:
measured 2026-09-02, rosbag2's writer plateaus at 38-46 MB/s (46 with --max-cache-size 200)
while DDS carries >=142 MB/s and the eMMC takes 136 MB/s. The image topics are best_effort,
so the shortfall is DROPPED, not queued - the capture node held a clean 30 Hz throughout
while the bag got 6.1 Hz. Foxy cannot compress its way out either: --compression-mode has
only {none,file}, and file mode compresses AFTER the write path, so it saves disk and not
throughput; message mode arrives in Galactic.

So: capture raw at the full rate (argus_capture_node -p image_log_dir:=, 29.4-29.7 fps to
tmpfs, 20.95 fps to eMMC - storage-bound, nothing dropped), and convert here, where 46 MB/s
does not matter because nothing is real time. The result replays with `ros2 bag play` under
sim time like any other bag, which the raw files cannot do.

The timestamps are the originals: camN_index.csv carries the exposure-midpoint stamp the
capture node computed per frame (SOF - exposure/2), so the bag is faithful to the rig and
NOT re-stamped at conversion time.

  raw_log_to_bag.py <raw_log_dir> -o out.bag [--compress] [--max-frames N]
"""
import argparse
import pathlib
import sqlite3
import sys

import numpy as np
from rosbags.typesys import Stores, get_typestore


def read_geometry(d):
    g = {}
    for line in (d / "geometry.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            g[parts[0]] = parts[1]
    return int(g["width"]), int(g["height"]), int(g["bytes_per_frame"])


def read_index(path):
    """(stamp_ns, byte offset) per frame, skipping the comment header."""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("stamp"):
            continue
        a, b = line.split(",")
        out.append((int(a), int(b)))
    return out


def write_metadata(out, dbname, topics, counts, t0, span_ns, comp_fmt="", comp_mode=""):
    """rosbag2 BAG FORMAT VERSION 4 - what Foxy writes and, more to the point, what Foxy
    READS. The rosbags library only writes versions 8 and 9, so a bag produced with its
    Writer cannot be opened by `ros2 bag play` on this board at all. The schema is two
    tables and this file, so writing v4 directly is both simpler and actually usable."""
    lines = ["rosbag2_bagfile_information:",
             "  version: 4",
             "  storage_identifier: sqlite3",
             "  relative_file_paths:",
             "    - %s" % dbname,
             "  duration:",
             "    nanoseconds: %d" % span_ns,
             "  starting_time:",
             "    nanoseconds_since_epoch: %d" % t0,
             "  message_count: %d" % sum(counts.values()),
             "  topics_with_message_count:"]
    for t in topics:
        lines += ["    - topic_metadata:",
                  "        name: %s" % t,
                  "        type: sensor_msgs/msg/Image",
                  "        serialization_format: cdr",
                  # empty = rosbag2 defaults (reliable). A reliable publisher still serves a
                  # best_effort subscriber, so this replays into the VO node fine, and it
                  # avoids emitting a QoS string that a different distro might reject.
                  '        offered_qos_profiles: ""',
                  "      message_count: %d" % counts[t]]
    lines += ['  compression_format: "%s"' % comp_fmt,
              '  compression_mode: "%s"' % comp_mode, ""]
    (out / "metadata.yaml").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir")
    ap.add_argument("-o", "--out", required=True, help="output bag directory (must not exist)")
    ap.add_argument("--compress", action="store_true",
                    help="zstd the .db3, as rosbag2 --compression-mode file does. Free here: "
                         "nothing is real time at conversion, which is exactly why the board "
                         "could not do it.")
    # --max-frames TAKES THE FIRST N, which is almost never the N you want. The rig is
    # stationary while the operator starts the run and reaches the rig, so the first frames
    # are the least useful in the log: on imglog_vio1_20260903_103102 a `--max-frames 600`
    # subset (30 s, used for the 5.0c replay) spent its first 5.7 s on a stationary rig and
    # threw away 21 s of motion that came later. Use --motion instead and let the IMU pick.
    ap.add_argument("--max-frames", type=int, default=0,
                    help="per camera, 0 = all. Takes the FIRST N - see --motion.")
    ap.add_argument("--motion", action="store_true",
                    help="convert only the window where the IMU says the rig was moving, "
                         "read from imu0.csv in the log dir. This is normally what you want: "
                         "a stationary rig makes a flawless log with nothing in it.")
    ap.add_argument("--pad-s", type=float, default=1.0,
                    help="seconds to keep either side of the motion window (default 1.0), "
                         "so VO has a moment to initialise before anything moves")
    a = ap.parse_args()

    d = pathlib.Path(a.log_dir)
    w, h, nbytes = read_geometry(d)
    print("geometry: %dx%d, %d bytes/frame" % (w, h, nbytes))

    cams = sorted(p.name[: -len("_index.csv")] for p in d.glob("cam*_index.csv"))
    if not cams:
        sys.exit("no cam*_index.csv in %s" % d)

    lo = hi = None
    if a.motion:
        # Shared with check_log_sets.py rather than reimplemented: two thresholds that drift
        # apart would mean the tool that reports the motion window and the tool that cuts on
        # it disagree about where it is.
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from check_log_sets import motion_window
        mw = motion_window([str(d)])
        if not mw:
            sys.exit("--motion needs imu0.csv in %s" % d)
        lo, hi = mw[0], mw[1]
        if lo is None:
            sys.exit("--motion: the gyro never crossed %.4f rad/s - the rig never moved in "
                     "this log, so there is nothing worth converting." % mw[3])
        pad = int(a.pad_s * 1e9)
        lo, hi = lo - pad, hi + pad
        print("motion window: %.2f s (+/- %.1f s pad)" % ((hi - lo) / 1e9, a.pad_s))

    # Merge all cameras into ONE time-ordered stream. A bag whose messages are not sorted by
    # timestamp still plays, but every consumer that assumes monotonic time gets it wrong, and
    # the whole point of these logs is debugging timing.
    entries = []
    for cam in cams:
        idx = read_index(d / (cam + "_index.csv"))
        have = (d / (cam + ".raw")).stat().st_size // nbytes
        # Integrity FIRST, on the whole index, and only then the motion cut - checking a
        # filtered index against the full .raw would report every --motion run as damaged.
        if len(idx) != have:
            # The capture node stops the log if either stream fails, so a mismatch means the
            # run was killed mid-write. Trust the index: those frames are known-complete.
            print("  %s: index %d vs raw %d frames - using %d" % (cam, len(idx), have, min(len(idx), have)))
        idx = idx[:min(len(idx), have)]
        if lo is not None:
            kept = [(s, o) for s, o in idx if lo <= s <= hi]
            print("  %s: %d of %d frames inside the motion window" % (cam, len(kept), len(idx)))
            idx = kept
        n = len(idx)
        if a.max_frames:
            n = min(n, a.max_frames)
        entries += [(stamp, cam, off) for stamp, off in idx[:n]]
    entries.sort(key=lambda e: e[0])
    if not entries:
        sys.exit("no frames found")
    print("%d frames from %s" % (len(entries), ", ".join(cams)))

    mm = {c: np.memmap(d / (c + ".raw"), dtype=np.uint8, mode="r") for c in cams}
    ts = get_typestore(Stores.ROS2_FOXY)
    Image = ts.types["sensor_msgs/msg/Image"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dbname = out.name + "_0.db3"
    con = sqlite3.connect(out / dbname)
    con.execute("CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT NOT NULL,type TEXT NOT NULL,"
                "serialization_format TEXT NOT NULL,offered_qos_profiles TEXT NOT NULL)")
    con.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER NOT NULL,"
                "timestamp INTEGER NOT NULL, data BLOB NOT NULL)")
    topic_of = {}
    for i, cam in enumerate(cams, start=1):
        topic = "/%s/image_raw" % cam
        topic_of[cam] = (i, topic)
        con.execute("INSERT INTO topics VALUES (?,?,?,?,?)",
                    (i, topic, "sensor_msgs/msg/Image", "cdr", ""))

    counts = {topic_of[c][1]: 0 for c in cams}
    for i, (stamp, cam, off) in enumerate(entries):
        msg = Image(header=Header(stamp=Time(sec=stamp // 1_000_000_000,
                                             nanosec=stamp % 1_000_000_000),
                                  frame_id=cam),
                    height=h, width=w, encoding="mono8", is_bigendian=0, step=w,
                    data=np.asarray(mm[cam][off:off + nbytes]))
        con.execute("INSERT INTO messages(topic_id,timestamp,data) VALUES (?,?,?)",
                    (topic_of[cam][0], stamp, ts.serialize_cdr(msg, Image.__msgtype__)))
        counts[topic_of[cam][1]] += 1
        if i % 200 == 0:
            con.commit()
            print("  %d/%d" % (i, len(entries)), end="\r", flush=True)
    con.commit()
    con.close()

    span = entries[-1][0] - entries[0][0]
    comp_fmt = comp_mode = ""
    if a.compress:
        import zstandard
        raw = (out / dbname).read_bytes()
        (out / (dbname + ".zstd")).write_bytes(zstandard.ZstdCompressor(level=3).compress(raw))
        (out / dbname).unlink()
        dbname += ".zstd"
        comp_fmt, comp_mode = "zstd", "FILE"
        print("  compressed %.0f -> %.0f MB" % (len(raw) / 1e6, (out / dbname).stat().st_size / 1e6))

    write_metadata(out, dbname, [topic_of[c][1] for c in cams], counts,
                   entries[0][0], span, comp_fmt, comp_mode)
    print("\nwrote %s: %d msgs over %.2f s (%.1f Hz per camera)"
          % (out, len(entries), span / 1e9, len(entries) / (span / 1e9) / len(cams) if span else 0))
    print("  ros2 bag play %s --clock     # then run nodes with use_sim_time:=true" % out)


if __name__ == "__main__":
    main()
