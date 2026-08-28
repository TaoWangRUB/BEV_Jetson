#!/usr/bin/env python3
"""Host side of the calibration capture: receive MJPEG from the TX2, look, and record.

Everything that costs CPU lives here, because the host has 16 cores and the TX2 has 6
that are already running the ISP. The board only captures and hardware-encodes
(scripts/stream/calib_sender.sh); this end decodes, detects the AprilGrid, draws the
coverage that tells you whether the sweep was any good, and writes the frames.

Live feedback is the reason this exists. The first stage recorded blind looked fine and
turned out to have left 22 of 36 image cells untouched - all at the periphery, which is
where fisheye distortion lives and where a calibration that never saw the target is
confidently wrong. The coverage grid fills in as you sweep, so you can see it happen.

  # on the TX2
  ./calib_sender.sh c 4
  # on the host - preview only, records nothing
  python3 scripts/stream/calib_receiver.py --cams c
  # ...and when the sweep is worth keeping
  python3 scripts/stream/calib_receiver.py --cams c --record datasets/calib/CAM_A

Frames are written as <index>_<host_ns>.jpg exactly as received, with no re-encoding:
what the calibrator sees is what the sensor sent, minus the one transport JPEG.

⚠ There is NO capture timestamp in this path - MJPEG over TCP carries none, and the
host arrival time is not one. That is fine for intrinsics and for the pairwise
extrinsics, which use no time at all. The camera-IMU stage is different: the offset is
the measurement, so it needs the ROS path (README 4.7).
"""
import argparse
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

PORTS = {"c": (5000, "cam1 front-left"), "d": (5001, "cam2 front-right"),
         "e": (5002, "cam3 back-left"),  "f": (5003, "cam4 back-right")}
GRID = 6

_params = cv2.aruco.DetectorParameters()
_params.markerBorderBits = 2
_params.adaptiveThreshWinSizeStep = 1
_params.adaptiveThreshWinSizeMin = 3
_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), _params)

state = {}                      # port -> dict(img, polys, ntags, cov, n, saved)
lock = threading.Lock()

# What a calibration is actually short of is DIVERSITY, not frames. A hundred views
# taken centred and flat-on constrain less than thirty spread across the image, across
# tilt and across distance - flat-on views in particular cannot separate focal length
# from radial distortion, because both just scale the target. So novelty is judged in
# three axes at once and each bin has a quota; the operator is told what is still empty
# rather than left to guess when to stop.
POS_BINS = 6            # target centre, across the image
TILT_BINS = 3           # how oblique the board is (its quad's squareness)
SCALE_BINS = 3          # apparent size, i.e. distance


def pose_bin(pts_list, shape):
    """(position, tilt, scale) bin for the target in this frame.

    Tilt is estimated from the tag quads themselves: a square-on tag projects to a
    square, an oblique one to a trapezoid, so the ratio of its diagonals is a cheap,
    calibration-free measure of obliquity that needs no pose solve.
    """
    cx = np.mean([p[:, 0].mean() for p in pts_list])
    cy = np.mean([p[:, 1].mean() for p in pts_list])
    pos = (min(POS_BINS - 1, int(cy / shape[0] * POS_BINS)),
           min(POS_BINS - 1, int(cx / shape[1] * POS_BINS)))

    areas, skews = [], []
    for p in pts_list:
        areas.append(abs(cv2.contourArea(p.astype(np.float32))))
        d1 = np.linalg.norm(p[0] - p[2]); d2 = np.linalg.norm(p[1] - p[3])
        skews.append(min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 1.0)
    frac = np.mean(areas) / (shape[0] * shape[1])
    scale = 0 if frac < 0.002 else (1 if frac < 0.008 else 2)
    sk = float(np.mean(skews))                      # 1.0 = square-on, lower = oblique
    tilt = 0 if sk > 0.92 else (1 if sk > 0.8 else 2)
    return pos, tilt, scale


def receive(host, port_letter, record_dir, detect, every=1, sel=None):
    """One camera: read the multipart JPEG stream, decode, detect, optionally save."""
    tcp, label = PORTS[port_letter]
    st = {"img": None, "polys": [], "ntags": 0, "n": 0, "saved": 0,
          "cov": np.zeros((GRID, GRID), int), "label": label, "err": None,
          "bins": {}, "accepted": False, "why": ""}
    with lock:
        state[port_letter] = st

    while True:
        try:
            s = socket.create_connection((host, tcp), timeout=10)
            s.settimeout(10)
            st["err"] = None
            buf = b""
            while True:
                chunk = s.recv(1 << 16)
                if not chunk:
                    raise ConnectionError("stream closed")
                buf += chunk
                # multipartmux frames are JPEGs back to back; find them by marker rather
                # than parsing MIME headers, which gstreamer formats loosely.
                while True:
                    a = buf.find(b"\xff\xd8")
                    b = buf.find(b"\xff\xd9", a + 2)
                    if a < 0 or b < 0:
                        break
                    jpg, buf = buf[a:b + 2], buf[b + 2:]
                    handle(st, jpg, record_dir, detect, every, sel)
        except Exception as e:                      # a dropped link must not end the run
            st["err"] = str(e)
            time.sleep(1.0)


def handle(st, jpg, record_dir, detect, every=1, sel=None):
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    polys = []
    if detect:
        corners, ids, _ = _detector.detectMarkers(img)
        if ids is not None:
            for c in corners:
                pts = c[0]
                polys.append(pts)
                gy = min(GRID - 1, int(pts[:, 1].mean() / img.shape[0] * GRID))
                gx = min(GRID - 1, int(pts[:, 0].mean() / img.shape[1] * GRID))
                st["cov"][gy, gx] += 1
    with lock:
        st["img"], st["polys"], st["ntags"], st["n"] = img, polys, len(polys), st["n"] + 1
    # Online selection: keep a frame only if it teaches the solver something new. Doing
    # this live rather than afterwards is the difference between a set that is uniform
    # by construction and one that is uniform by luck - offline selection can only pick
    # from what was swept, and cannot conjure a corner that was never visited.
    if sel is not None:
        st["accepted"], st["why"] = False, ""
        if not polys or len(polys) < sel["min_tags"]:
            st["why"] = "%d tags" % len(polys)
        else:
            xs = [p[:, 0].mean() for p in polys]; ys = [p[:, 1].mean() for p in polys]
            x0, x1 = int(max(0, min(xs) - 40)), int(min(img.shape[1], max(xs) + 40))
            y0, y1 = int(max(0, min(ys) - 40)), int(min(img.shape[0], max(ys) + 40))
            roi = img[y0:y1, x0:x1]
            sharp = cv2.Laplacian(roi, cv2.CV_64F).var() if roi.size else 0.0
            if sharp < sel["min_sharp"]:
                st["why"] = "blurred (%.0f)" % sharp
            else:
                key = pose_bin(polys, img.shape)
                have = st["bins"].get(key, 0)
                if have >= sel["per_bin"]:
                    st["why"] = "bin full"
                else:
                    st["bins"][key] = have + 1
                    st["accepted"] = True
        if not st["accepted"]:
            return

    if record_dir and st["n"] % every == 0:
        # Written as received: re-encoding would add a second generation of JPEG loss to
        # the images a calibration depends on.
        path = os.path.join(record_dir, "%06d_%d.jpg" % (st["saved"], time.time_ns()))
        with open(path, "wb") as f:
            f.write(jpg)
        st["saved"] += 1


def render(scale):
    with lock:
        cams = sorted(state)
        snap = [(p, dict(state[p]), state[p]["cov"].copy()) for p in cams]
    imgs = [(p, s, c) for p, s, c in snap if s["img"] is not None]
    if not imgs:
        return None
    h, w = imgs[0][1]["img"].shape
    h, w = int(h * scale), int(w * scale)
    cols = 1 if len(imgs) == 1 else 2
    rows = (len(imgs) + cols - 1) // cols
    canvas = np.zeros((h * rows, w * cols), np.uint8)

    for i, (p, s, cov) in enumerate(imgs):
        y, x = (i // cols) * h, (i % cols) * w
        canvas[y:y + h, x:x + w] = cv2.resize(s["img"], (w, h), interpolation=cv2.INTER_AREA)
        for pts in s["polys"]:
            cv2.polylines(canvas, [(pts * scale).astype(np.int32) + [x, y]], True, 255, 2)
        filled = int((cov > 0).sum())
        text = "%s  %d tags  coverage %d/%d  saved %d" % (
            s["label"], s["ntags"], filled, GRID * GRID, s["saved"])
        if s.get("bins"):
            tilts = len({k[1] for k in s["bins"]}); scales = len({k[2] for k in s["bins"]})
            text += "  tilt %d/%d  scale %d/%d" % (tilts, TILT_BINS, scales, SCALE_BINS)
        if s.get("why"):
            text += "   [skipped: %s]" % s["why"]
        if s["err"]:
            text += "  [%s]" % s["err"]
        for col, th in ((0, 4), (255, 1)):
            cv2.putText(canvas, text, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th)
        # Coverage: filled square = the target has been in that part of the frame. Sweep
        # until the outer ring is filled, not just the middle.
        sq = max(8, h // 36)
        for gy in range(GRID):
            for gx in range(GRID):
                x0, y0 = x + 10 + gx * sq, y + h - 14 - (GRID - gy) * sq
                cv2.rectangle(canvas, (x0, y0), (x0 + sq - 2, y0 + sq - 2),
                              255 if cov[gy, gx] else 110, -1 if cov[gy, gx] else 1)
    return canvas


PAGE = (b"<html><body style='margin:0;background:#111;color:#ccc;font:13px system-ui'>"
        b"<div style='padding:6px'>"
        b"<a style='color:#6cf' href='/?scale=0.5'>half</a> &middot; "
        b"<a style='color:#6cf' href='/?scale=1.0'>full</a></div>"
        b"<img src='/stream%s' style='width:100%%'></body></html>")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        scale = max(0.1, min(1.0, float(q.get("scale", ["0.5"])[0])))
        if u.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE % (("?" + u.query).encode() if u.query else b""))
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        try:
            while True:
                m = render(scale)
                if m is None:
                    time.sleep(0.05)
                    continue
                ok, jpg = cv2.imencode(".jpg", m, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                                     + jpg.tobytes() + b"\r\n")
                time.sleep(0.15)
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(ThreadingHTTPServer):
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.42.0.157", help="the TX2")
    ap.add_argument("--cams", nargs="+", default=["c"], choices=list(PORTS),
                    help="port letters to receive: c=front-left d=front-right "
                         "e=back-left f=back-right")
    ap.add_argument("--record", default="", help="directory to write frames into")
    ap.add_argument("--record-every", type=int, default=1,
                    help="keep every Nth frame (default 1 = everything). Recording all of "
                         "it is usually right: selection is better done offline, where a "
                         "blurred or redundant frame can be rejected on evidence")
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="save only frames that add something: a new (position, tilt, "
                         "scale) bin, sharp enough, with enough tags")
    ap.add_argument("--per-bin", type=int, default=3, help="frames to keep per bin")
    ap.add_argument("--min-tags", type=int, default=6)
    ap.add_argument("--min-sharp", type=float, default=40.0,
                    help="Laplacian variance over the target; blurred tags detect fine "
                         "but land their corners a pixel or two off")
    ap.add_argument("--port", type=int, default=8090, help="local preview port")
    a = ap.parse_args()

    if a.record:
        os.makedirs(a.record, exist_ok=True)
    for p in a.cams:
        rec = os.path.join(a.record, PORTS[p][1].split()[0]) if a.record else ""
        if rec:
            os.makedirs(rec, exist_ok=True)
        sel = None if not a.auto else {"per_bin": a.per_bin, "min_tags": a.min_tags,
                                       "min_sharp": a.min_sharp}
        threading.Thread(target=receive,
                         args=(a.host, p, rec, not a.no_detect, a.record_every, sel),
                         daemon=True).start()

    print("receiving %s from %s; preview http://localhost:%d/%s"
          % (",".join(a.cams), a.host, a.port,
             ("  RECORDING to " + a.record) if a.record else "  (not recording)"))
    Server(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
