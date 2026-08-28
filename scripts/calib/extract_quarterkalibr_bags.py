#!/usr/bin/env python3
"""Split one recording into the staged bags quarterKalibr's pipeline expects.

quarterKalibr calibrates a quadcam in STAGES rather than solving all four fisheyes at
once, and the stage boundaries are not a flag you pass — they are performed in front of
the rig and recovered here from which cameras can see the AprilGrid. Its own
`BagExtractor.py` recovers them from an ASSEMBLED image that it splits into four; our
capture node publishes four separate topics, so this does the same job from those.

THE RECORDING PROTOCOL (this is the part that is nowhere in quarterKalibr's README —
it is encoded in its `step_dict`). In ONE continuous recording, show the target to:

    1. cam1 alone          5. cam4 + cam1 together      <- the four overlaps, in
    2. cam2 alone          6. cam1 + cam2 together         this order, each pair
    3. cam3 alone          7. cam2 + cam3 together         adjacent around the ring
    4. cam4 alone          8. cam3 + cam4 together

Stages 1-4 give per-camera intrinsics; 5-8 give the adjacent-pair extrinsics that get
composed around the ring. The order matters: a stage is entered only when the set of
cameras seeing tags matches the NEXT expected pattern, so skipping one stalls the rest.
Move slowly between stages and let each settle.

  ros2 bag record /cam1/image_raw /cam2/image_raw /cam3/image_raw /cam4/image_raw \
                  /cam1/frame_meta ... /imu0
  # convert to ROS1 (rosbags-convert), then on the host:
  python3 scripts/calib/extract_quarterkalibr_bags.py --bag rec.bag --out stages/

Outputs one bag per stage, named as quarterKalibr expects (CAM_A .. CAM_C-CAM_D), the
IMU passed through into every bag, and the target yaml beside them.
"""
import argparse
import os
import shutil
import sys

import cv2 as cv
import rosbag
import tqdm
from cv_bridge import CvBridge

# quarterKalibr's naming: its notebook looks for these bag names in this order.
STAGE_NAMES = ["CAM_A", "CAM_B", "CAM_C", "CAM_D",
               "CAM_D-CAM_A", "CAM_A-CAM_B", "CAM_B-CAM_C", "CAM_C-CAM_D"]
# Which cameras must be seeing tags for each stage, as a bitmask over cam1..cam4.
STAGE_MASK = [0b0001, 0b0010, 0b0100, 0b1000,
              0b1001, 0b0011, 0b0110, 0b1100]
CAMERA_TOPICS = ["CAM_A", "CAM_B", "CAM_C", "CAM_D"]   # topic names inside the output bags


def make_detector():
    params = cv.aruco.DetectorParameters()
    params.markerBorderBits = 2
    params.adaptiveThreshWinSizeStep = 1
    params.adaptiveThreshWinSizeMin = 3
    return cv.aruco.ArucoDetector(
        cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11), params)


def group_by_timestamp(bag, topics, slop_ns):
    """Yield {topic: (msg, t)} sets whose stamps agree within slop_ns.

    Grouping is by the frames' OWN timestamps, never by arrival order: these are four
    separate subscriptions and their order in the bag says nothing about which trigger
    edge a frame came from. The rig is hardware-triggered, so a real set agrees to
    microseconds and anything else is not a set.
    """
    pending = {t: [] for t in topics}
    for topic, msg, t in bag.read_messages(topics=topics):
        stamp = msg.header.stamp.to_nsec()
        pending[topic].append((stamp, msg, t))
        # A set is complete once every topic has something at least as new as the oldest
        # head, so drive off the topic with the oldest head.
        while all(pending[x] for x in topics):
            heads = {x: pending[x][0] for x in topics}
            t0 = min(h[0] for h in heads.values())
            group, ok = {}, True
            for x in topics:
                stamp_x, msg_x, rt_x = heads[x]
                if abs(stamp_x - t0) > slop_ns:
                    ok = False
                    break
                group[x] = (msg_x, rt_x)
            if ok:
                for x in topics:
                    pending[x].pop(0)
                yield group
            else:
                # drop the straggler and try again
                oldest = min(topics, key=lambda x: heads[x][0])
                pending[oldest].pop(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="ROS1 bag (converted from ros2 bag)")
    ap.add_argument("--out", required=True, help="output directory for the staged bags")
    ap.add_argument("--image-topics", nargs=4,
                    default=["/cam1/image_raw", "/cam2/image_raw",
                             "/cam3/image_raw", "/cam4/image_raw"])
    ap.add_argument("--imu-topic", default="/imu0")
    ap.add_argument("--target", default="config/calib/april_6x6.yaml",
                    help="AprilGrid target yaml, copied beside the staged bags")
    ap.add_argument("--slop-us", type=int, default=1000,
                    help="max spread within one 4-camera set (cuVSLAM's own gate is 1 ms; "
                         "the triggered rig measures ~1 us)")
    ap.add_argument("--min-tags", type=int, default=4,
                    help="tags a camera must see before it counts as seeing the target")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    detector = make_detector()
    bridge = CvBridge()
    bag = rosbag.Bag(a.bag)

    total = min(bag.get_message_count(t) for t in a.image_topics)
    print("%d sets to scan across %s" % (total, ", ".join(a.image_topics)))

    stage = 0
    out_bag = rosbag.Bag(os.path.join(a.out, STAGE_NAMES[0] + ".bag"), "w")
    written = [0] * len(STAGE_NAMES)
    pbar = tqdm.tqdm(total=total, colour="green")

    for group in group_by_timestamp(bag, a.image_topics, a.slop_us * 1000):
        pbar.update(1)
        mask, images = 0, []
        for i, topic in enumerate(a.image_topics):
            msg, _ = group[topic]
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            images.append(img)
            gray = img if img.ndim == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, ids, _ = detector.detectMarkers(gray)
            if ids is not None and len(ids) >= a.min_tags:
                mask |= 1 << i

        # Advance only into the NEXT stage. Unlike quarterKalibr's step_dict lookup, an
        # unexpected pattern (three cameras seeing the target at once, say) is ignored
        # rather than raising — mid-recording is exactly where that happens.
        if stage + 1 < len(STAGE_NAMES) and mask == STAGE_MASK[stage + 1]:
            out_bag.close()
            print("\nstage %d/%d complete: %s (%d sets)"
                  % (stage + 1, len(STAGE_NAMES), STAGE_NAMES[stage], written[stage]))
            stage += 1
            out_bag = rosbag.Bag(os.path.join(a.out, STAGE_NAMES[stage] + ".bag"), "w")
            continue

        if mask != STAGE_MASK[stage]:
            continue                       # between stages: nothing useful to record

        for i, name in enumerate(CAMERA_TOPICS):
            msg, rt = group[a.image_topics[i]]
            out = bridge.cv2_to_imgmsg(images[i], encoding="mono8" if images[i].ndim == 2 else "bgr8")
            out.header = msg.header
            out_bag.write("/" + name, out, rt)
        written[stage] += 1

    out_bag.close()
    pbar.close()

    # The IMU has to be in whichever bag the camera-IMU stage uses, and passing it into
    # all of them costs little and removes a way to get it wrong.
    for name in STAGE_NAMES:
        path = os.path.join(a.out, name + ".bag")
        if not os.path.exists(path):
            continue
    with rosbag.Bag(os.path.join(a.out, "imu.bag"), "w") as imu_out:
        n = 0
        for topic, msg, t in bag.read_messages(topics=[a.imu_topic]):
            imu_out.write(topic, msg, t)
            n += 1
    print("imu.bag: %d samples from %s" % (n, a.imu_topic))

    if os.path.exists(a.target):
        shutil.copy(a.target, a.out)

    print("\nstages written to %s:" % a.out)
    for i, name in enumerate(STAGE_NAMES):
        flag = "" if written[i] else "   <-- EMPTY: that stage was never performed"
        print("  %-12s %6d sets%s" % (name, written[i], flag))
    if not all(written):
        print("\nRe-record the missing stages: each one is a separate solve, and the "
              "pairs are what make the ring close.")
        sys.exit(1)


if __name__ == "__main__":
    main()
