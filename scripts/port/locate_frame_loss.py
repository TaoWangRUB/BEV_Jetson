#!/usr/bin/env python3
"""WHERE does a triggered rig lose a frame? Set completeness says how many, not where.

check_log_sets.py reports that N trigger edges are missing a camera. That number alone
cannot be acted on: an edge the SENSOR never captured and a frame the CONSUMER failed to
collect need opposite fixes, and both look identical in the index CSV.

The capture node writes two independent counters per frame for exactly this reason:

  capture_id   session-side  - what the Argus capture session produced
  seq          consumer-side - what was actually delivered to acquireFrame()

So, per gap in the SOF timeline (a missing trigger edge):

  capture_id jumps AND seq jumps    the frame was never produced. The sensor or the driver
                                    missed the trigger edge - look at the pulse, the
                                    polarity, or the sensor's own frame timing. Nothing the
                                    capture loop does can recover it.
  capture_id contiguous, seq jumps  the frame WAS produced and we failed to acquire it in
                                    time; Argus dropped it at the EGLStream. This is the
                                    capture loop being too slow for the trigger period -
                                    fixable on our side.
  neither jumps, SOF still gaps     the counters disagree with time: a stalled session, or
                                    a period estimate that is wrong for this run.

  locate_frame_loss.py <dir> [<dir> ...]        # dirs holding camN.csv
  locate_frame_loss.py --period-us 50000 <dir>  # override the inferred trigger period
"""
import argparse
import glob
import os
import statistics
import sys


def read_csv(path):
    """-> list of (stamp_ns, seq, capture_id, sof_ns). Comment lines start with '#'."""
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) < 4:
            continue
        try:
            rows.append((int(f[0]), int(f[1]), int(f[2]), int(f[3])))
        except ValueError:
            continue
    return rows


def analyse(cam, rows, period_ns):
    print("\n%s: %d frames" % (cam, len(rows)))
    if len(rows) < 2:
        print("  too few frames to analyse")
        return

    sof = [r[3] for r in rows]
    if period_ns is None:
        # Median inter-frame gap IS the period: a handful of missed edges cannot move a
        # median, while a mean or a min/max would be dragged by exactly the events of
        # interest.
        period_ns = statistics.median(sof[i + 1] - sof[i] for i in range(len(sof) - 1))
    print("  trigger period %.3f ms (%.2f Hz)" % (period_ns / 1e6, 1e9 / period_ns))

    never_produced = 0   # capture_id gapped too: the frame does not exist
    lost_in_stream = 0   # capture_id contiguous but seq gapped: produced, not collected
    unexplained = 0
    total_missing = 0
    examples = []

    for i in range(len(rows) - 1):
        t0, seq0, cid0, sof0 = rows[i]
        t1, seq1, cid1, sof1 = rows[i + 1]
        # How many trigger edges does this gap span? 1 = no loss.
        edges = int(round((sof1 - sof0) / period_ns))
        if edges <= 1:
            continue
        missing = edges - 1
        total_missing += missing
        d_seq = seq1 - seq0
        d_cid = cid1 - cid0
        if d_cid > 1 and d_seq > 1:
            kind = "never produced (sensor/driver missed the edge)"
            never_produced += missing
        elif d_cid <= 1 and d_seq > 1:
            kind = "LOST IN STREAM (produced, not acquired in time)"
            lost_in_stream += missing
        else:
            kind = "unexplained (counters contiguous, time is not)"
            unexplained += missing
        if len(examples) < 6:
            examples.append("    at %.3f s: %d edge(s), d_seq=%d d_capture_id=%d -> %s"
                            % ((sof0 - sof[0]) / 1e9, missing, d_seq, d_cid, kind))

    print("  missing trigger edges: %d" % total_missing)
    if examples:
        print("\n".join(examples))
    if total_missing:
        print("    never produced : %d" % never_produced)
        print("    lost in stream : %d" % lost_in_stream)
        if unexplained:
            print("    unexplained    : %d" % unexplained)

    # A gap in seq with NO gap in the SOF timeline is the other half of the same story:
    # Argus renumbered without losing time, which still means a frame went missing.
    seq_gaps = sum((rows[i + 1][1] - rows[i][1] - 1)
                   for i in range(len(rows) - 1) if rows[i + 1][1] - rows[i][1] > 1)
    cid_gaps = sum((rows[i + 1][2] - rows[i][2] - 1)
                   for i in range(len(rows) - 1) if rows[i + 1][2] - rows[i][2] > 1)
    print("  total gaps: seq %d, capture_id %d" % (seq_gaps, cid_gaps))
    return dict(missing=total_missing, never=never_produced,
                stream=lost_in_stream, unexplained=unexplained)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--period-us", type=float, default=None,
                    help="trigger period in us; default = median inter-frame gap")
    a = ap.parse_args()

    period_ns = a.period_us * 1000 if a.period_us else None
    files = []
    for d in a.dirs:
        files.extend(sorted(glob.glob(os.path.join(d, "cam*.csv"))))
    files = [f for f in files if not f.endswith("_index.csv")]
    if not files:
        sys.exit("no camN.csv found in: %s" % ", ".join(a.dirs))

    totals = dict(missing=0, never=0, stream=0, unexplained=0)
    for f in files:
        r = analyse(os.path.basename(f)[:-4], read_csv(f), period_ns)
        if r:
            for k in totals:
                totals[k] += r[k]

    print("\n=== all cameras ===")
    print("missing trigger edges : %d" % totals["missing"])
    print("  never produced      : %d   (sensor/driver - not the capture loop)" % totals["never"])
    print("  lost in stream      : %d   (produced but not acquired - the capture loop)"
          % totals["stream"])
    if totals["unexplained"]:
        print("  unexplained         : %d" % totals["unexplained"])


if __name__ == "__main__":
    main()
