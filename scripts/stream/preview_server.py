#!/usr/bin/env python3
"""Live 2x2 preview of the four cameras, served as MJPEG over HTTP.

For aiming a calibration target when the board has no display and the link cannot carry
raw frames: a 1456x1088 mono frame is 1.58 MB, so four at 3 Hz is ~19 MB/s. Downscaled
and JPEG-encoded on the board it is a few hundred KB/s, which any link carries.

It also detects the AprilGrid live on the cameras of the current stage and draws a
coverage grid that fills in as the target is swept. That feedback is the point: the
first stage recorded here looked fine and turned out to have left 22 of 36 image cells
untouched, all of them at the periphery - which is exactly where fisheye distortion
lives and where a calibration that never saw the target goes confidently wrong.

Cells match the rig seen from above (config/rig/rig_layout.yaml):

        front-left (cam1/c) | front-right (cam2/d)
        back-left  (cam3/e) | back-right  (cam4/f)

  python3 preview_server.py --port 8080 --detect cam1
  # then open http://<board>:8080/   (links there switch quality)
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

GRID = 6                 # coverage grid, matching the offline check's 6x6 report
DETECT_SCALE = 0.5       # detect at half size; full res costs CPU that capture needs

LAYOUT = {"cam1": (0, 0, "cam1 c FRONT-LEFT"),
          "cam2": (0, 1, "cam2 d FRONT-RIGHT"),
          "cam3": (1, 0, "cam3 e BACK-LEFT"),
          "cam4": (1, 1, "cam4 f BACK-RIGHT")}

SHOW = set(LAYOUT)
state = {"seq": 0}
latest, tags, coverage = {}, {}, {}
lock = threading.Lock()

_params = cv2.aruco.DetectorParameters_create()
_params.markerBorderBits = 2
_params.adaptiveThreshWinSizeStep = 1
_params.adaptiveThreshWinSizeMin = 3
_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)


class Preview(Node):
    def __init__(self, detect, show):
        super().__init__("preview_server")
        self.detect = detect
        # Subscribe ONLY to what is displayed. On a 6-core TX2 already running capture,
        # the IMU and a bag recorder, deserialising and encoding four 1.58 MB streams
        # pushed load to 8 and starved the stream until the page appeared dead. One
        # camera is a quarter of the pixels and of the DDS traffic.
        for cam in show:
            self.create_subscription(Image, "/%s/image_raw" % cam,
                                     lambda m, c=cam: self.on_image(c, m),
                                     qos_profile_sensor_data)

    def on_image(self, cam, msg):
        try:
            # Full resolution is kept: the downscale happens per request, so quality is
            # a URL parameter rather than a restart.
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            polys = []
            if cam in self.detect:
                small = cv2.resize(img, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                                   interpolation=cv2.INTER_AREA)
                corners, ids, _ = cv2.aruco.detectMarkers(small, _dict, parameters=_params)
                cov = coverage.setdefault(cam, np.zeros((GRID, GRID), int))
                if ids is not None:
                    for c in corners:
                        pts = c[0] / DETECT_SCALE
                        polys.append(pts)
                        cy = int(pts[:, 1].mean() / img.shape[0] * GRID)
                        cx = int(pts[:, 0].mean() / img.shape[1] * GRID)
                        cov[min(GRID - 1, cy), min(GRID - 1, cx)] += 1
            with lock:
                latest[cam] = img
                tags[cam] = polys
                state["seq"] += 1
        except Exception as e:
            # A raised callback kills the executor thread and the preview goes black with
            # nothing in the log, which is a miserable way to lose a calibration session.
            self.get_logger().error("preview %s: %s" % (cam, e))


def montage(scale, show):
    with lock:
        frames = {k: v for k, v in latest.items()}
        polys = {k: list(v) for k, v in tags.items()}
        cov = {k: v.copy() for k, v in coverage.items()}
        stamp = state["seq"]
    if not frames:
        return None
    ref = next(iter(frames.values()))
    h, w = int(ref.shape[0] * scale), int(ref.shape[1] * scale)
    cols = 1 if len(show) == 1 else 2
    rows = (len(show) + cols - 1) // cols
    canvas = np.zeros((h * rows, w * cols), np.uint8)

    for i, cam in enumerate(sorted(show)):
        label = LAYOUT[cam][2]
        y, x = (i // cols) * h, (i % cols) * w
        img = frames.get(cam)
        if img is None:
            cv2.putText(canvas, "%s: no frames" % cam, (x + 20, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            continue
        canvas[y:y + h, x:x + w] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        for p in polys.get(cam, []):
            cv2.polylines(canvas, [(p * scale).astype(np.int32) + [x, y]], True, 255, 2)

        text = "%s  %d tags" % (label, len(polys.get(cam, [])))
        for col, th in ((0, 4), (255, 1)):
            cv2.putText(canvas, text, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th)

        # Coverage so far: filled square = the target has been in that part of the frame.
        # Sweep until the edges and corners are filled, not just the middle.
        cg = cov.get(cam)
        if cg is not None:
            s = max(6, h // 40)
            for gy in range(GRID):
                for gx in range(GRID):
                    x0, y0 = x + 10 + gx * s, y + h - 12 - (GRID - gy) * s
                    if cg[gy, gx]:
                        cv2.rectangle(canvas, (x0, y0), (x0 + s - 2, y0 + s - 2), 255, -1)
                    else:
                        cv2.rectangle(canvas, (x0, y0), (x0 + s - 2, y0 + s - 2), 110, 1)

    if cols > 1:
        cv2.line(canvas, (w, 0), (w, h * rows), 200, 1)
    if rows > 1:
        cv2.line(canvas, (0, h), (w * cols, h), 200, 1)
    return canvas, stamp


PAGE = (b"<html><body style='margin:0;background:#111;color:#ccc;font:13px system-ui'>"
        b"<div style='padding:6px'>quality: "
        b"<a style='color:#6cf' href='/?scale=0.35&fps=6&q=70'>wifi</a> &middot; "
        b"<a style='color:#6cf' href='/?scale=0.6&fps=10&q=80'>ethernet</a> &middot; "
        b"<a style='color:#6cf' href='/?scale=1.0&fps=10&q=90'>full</a></div>"
        b"<img src='/stream%s' style='width:100%%'></body></html>")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        f = lambda k, d: float(q.get(k, [d])[0])
        scale = max(0.1, min(1.0, f("scale", 0.4)))
        fps = max(1.0, min(15.0, f("fps", 6)))
        quality = int(max(30, min(95, f("q", 75))))

        if urlparse(self.path).path == "/":
            qs = ("?" + urlparse(self.path).query) if urlparse(self.path).query else ""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE % qs.encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        last_seq, last_t = -1, 0.0
        try:
            while True:
                got = montage(scale, SHOW)
                now = time.time()
                if got is None or got[1] == last_seq or (now - last_t) < 1.0 / fps:
                    time.sleep(0.02)
                    continue
                m, last_seq = got
                last_t = now
                ok, jpg = cv2.imencode(".jpg", m, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok:
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                                     + jpg.tobytes() + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


# THREADING, not the default single-threaded HTTPServer: an MJPEG response never ends,
# so on a single-threaded server the first open stream blocks every later request - the
# page then appears simply dead when reloaded, with nothing wrong in the log.
class Server(ThreadingHTTPServer):
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--detect", nargs="*", default=["cam1"],
                    help="cameras to run live tag detection on (the current stage's)")
    ap.add_argument("--show", nargs="*", default=None,
                    help="cameras to display; defaults to --detect. Fewer = less load")
    a = ap.parse_args()

    global SHOW
    SHOW = set(a.show if a.show else a.detect)
    rclpy.init()
    node = Preview(set(a.detect), SHOW)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print("preview on http://0.0.0.0:%d/ showing %s, detecting on %s"
          % (a.port, ",".join(sorted(SHOW)), ",".join(a.detect)), flush=True)
    Server(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
