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


def receive(host, port_letter, record_dir, detect):
    """One camera: read the multipart JPEG stream, decode, detect, optionally save."""
    tcp, label = PORTS[port_letter]
    st = {"img": None, "polys": [], "ntags": 0, "n": 0, "saved": 0,
          "cov": np.zeros((GRID, GRID), int), "label": label, "err": None}
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
                    handle(st, jpg, record_dir, detect)
        except Exception as e:                      # a dropped link must not end the run
            st["err"] = str(e)
            time.sleep(1.0)


def handle(st, jpg, record_dir, detect):
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
    if record_dir:
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
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--port", type=int, default=8090, help="local preview port")
    a = ap.parse_args()

    if a.record:
        os.makedirs(a.record, exist_ok=True)
    for p in a.cams:
        rec = os.path.join(a.record, PORTS[p][1].split()[0]) if a.record else ""
        if rec:
            os.makedirs(rec, exist_ok=True)
        threading.Thread(target=receive, args=(a.host, p, rec, not a.no_detect),
                         daemon=True).start()

    print("receiving %s from %s; preview http://localhost:%d/%s"
          % (",".join(a.cams), a.host, a.port,
             ("  RECORDING to " + a.record) if a.record else "  (not recording)"))
    Server(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
