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
import os
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

# OpenCV parallelises the aruco detector across every core it can find - a ONE-camera
# preview measured 329% CPU - and detection runs inside the DDS subscriber callback, so
# that contention directly delays draining the queue. It cost 17% of the frames in a
# two-camera recording (6.4/6.1 Hz against 7.5). Two threads is plenty for a preview and
# leaves the recorder alone. The real fix is to detect off the callback thread.
cv2.setNumThreads(2)

# 8x8 and quota-based, matching filter_bag.py exactly: the live grid and the offline
# report must mean the same thing, or "covered" at the rig is not "covered" in the solve.
GRID = 8
QUOTA = 12               # frames wanted per cell, as filter_bag.py --quota
DETECT_SCALE = 0.5       # detect at half size; full res costs CPU that capture needs

LAYOUT = {"cam1": (0, 0, "cam1 c FRONT-LEFT"),
          "cam2": (0, 1, "cam2 d FRONT-RIGHT"),
          "cam3": (1, 0, "cam3 e BACK-LEFT"),
          "cam4": (1, 1, "cam4 f BACK-RIGHT")}

SHOW = set(LAYOUT)
state = {"seq": 0}
latest, tags, coverage = {}, {}, {}
# Coverage the PREVIOUS recording already achieved, per camera. Without this a re-sweep
# starts with all 64 cells flagged and the overlay hides the scene, when in fact only a
# handful of cells are actually missing. Seeded coverage means the boxes you see are the
# work that is genuinely left, and /reset returns here rather than to zero - a corner
# re-sweep is COMBINED with the earlier bag at solve time, not a replacement for it.
SEED = {}
lock = threading.Lock()

# OpenCV renamed the whole aruco entry point at 4.7. This file has to run in BOTH
# places: the board's Foxy container is OpenCV 4.2 (old API) and the host is 4.13 (new),
# and the failure is an AttributeError at import, i.e. after you have already set the
# rig up. Bind once here and detect through _detect().
if hasattr(cv2.aruco, "DetectorParameters_create"):            # OpenCV < 4.7
    _params = cv2.aruco.DetectorParameters_create()
    _dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
    _detector = None
else:                                                          # OpenCV >= 4.7
    _params = cv2.aruco.DetectorParameters()
    _dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    _detector = None                                           # built after tuning below

# markerBorderBits=2 is the AprilGrid's border; the adaptive-threshold window settings
# are what let a tag at the dim periphery of a fisheye still threshold.
_params.markerBorderBits = 2
_params.adaptiveThreshWinSizeStep = 1
_params.adaptiveThreshWinSizeMin = 3

if hasattr(cv2.aruco, "ArucoDetector"):
    _detector = cv2.aruco.ArucoDetector(_dict, _params)


def _detect(img):
    """corners, ids — through whichever aruco API this OpenCV has."""
    if _detector is not None:
        return _detector.detectMarkers(img)[:2]
    return cv2.aruco.detectMarkers(img, _dict, parameters=_params)[:2]


class Preview(Node):
    def __init__(self, detect, show, detect_every=1, rotate180=False):
        # Unique node name per process: this script gets killed and restarted often, and
        # a fresh participant reusing a dead one's name is a way to end up with a node
        # that exists, subscribes, and silently receives nothing.
        super().__init__("preview_server_%d" % os.getpid())
        self.detect = detect
        self.rotate180 = rotate180
        self.detect_every = max(1, detect_every)
        self.frame_no = {}
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
            # DISPLAY ONLY. The modules are mounted inverted (config/rig/rig_layout.yaml,
            # camera_roll_deg: 180) and the capture node publishes the raw sensor
            # orientation - it does no flip, deliberately: the whole extrinsic chain is
            # calibrated in the raw frame and the roll is folded in afterwards by
            # fold_roll_for_vo.py. So this rotation must NEVER reach anything recorded.
            # It exists so the operator sweeping a target is not working upside-down.
            # Consequence to know: the coverage grid below is then in DISPLAY
            # orientation, which is the offline report's grid relabelled by 180 deg.
            if self.rotate180:
                img = np.rot90(img, 2)
            polys = []
            if cam in self.detect:
                small = cv2.resize(img, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                                   interpolation=cv2.INTER_AREA)
                corners, ids = _detect(small)
                cov = coverage.setdefault(
                    cam, np.array(SEED[cam], int) if cam in SEED
                    else np.zeros((GRID, GRID), int))
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
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = x + max(4, (w - tw) // 2)          # centred: the CORNER cells are the ones
        for col, th in ((0, 4), (255, 1)):      # being worked, so keep them unobstructed
            cv2.putText(canvas, text, (tx, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th)

        # Coverage drawn ON THE IMAGE, not as a legend in the corner. A legend tells you
        # THAT a cell is empty; it cannot tell you WHERE to hold the board, and the
        # operator should not have to map grid indices onto the scene in their head.
        # Under-quota cells are outlined with the shortfall printed in them; satisfied
        # cells are left clean so the remaining work is what stands out.
        cg = cov.get(cam)
        if cg is not None:
            cw, ch = w / float(GRID), h / float(GRID)
            for gy in range(GRID):
                for gx in range(GRID):
                    need = QUOTA - int(cg[gy, gx])
                    if need <= 0:
                        continue
                    x0, y0 = int(x + gx * cw), int(y + gy * ch)
                    x1, y1 = int(x + (gx + 1) * cw), int(y + (gy + 1) * ch)
                    # thicker box the emptier the cell, so a wholly untouched corner is
                    # visible at a glance from across the room
                    th = 3 if need >= QUOTA else (2 if need > QUOTA // 2 else 1)
                    cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), 0, th + 2)
                    cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), 255, th)
                    cv2.putText(canvas, str(need), (x0 + int(cw / 2) - 8, y0 + int(ch / 2) + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 3)
                    cv2.putText(canvas, str(need), (x0 + int(cw / 2) - 8, y0 + int(ch / 2) + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)


    if cols > 1:
        cv2.line(canvas, (w, 0), (w, h * rows), 200, 1)
    if rows > 1:
        cv2.line(canvas, (0, h), (w * cols, h), 200, 1)
    return canvas, stamp


PAGE = (b"<html><body style='margin:0;background:#111;color:#ccc;font:13px system-ui'>"
        b"<div style='padding:6px'>quality: "
        b"<a style='color:#6cf' href='/?scale=0.35&fps=6&q=70'>wifi</a> &middot; "
        b"<a style='color:#6cf' href='/?scale=0.6&fps=10&q=80'>ethernet</a> &middot; "
        b"<a style='color:#6cf' href='/?scale=1.0&fps=10&q=90'>full</a>"
        b" &nbsp;|&nbsp; <a style='color:#fc6' href='/reset'>reset coverage</a>"
        b" &nbsp;<span id='s' style='color:#888'></span></div>"
        b"<img id='v' style='height:93vh;max-width:100vw;object-fit:contain;"
        b"display:block;margin:0 auto'>"
        # Poll one frame at a time. Each tick is an independent request, so a server
        # restart or a dropped connection costs one frame rather than the whole preview,
        # and it recovers without a manual refresh. The next request is only issued after
        # the previous frame decodes, so a slow link throttles itself instead of queueing.
        b"<script>"
        b"var q=location.search,i=document.getElementById('v'),s=document.getElementById('s');"
        b"var p=new URLSearchParams(q),ms=1000/(parseFloat(p.get('fps'))||6),bad=0;"
        b"function tick(){var n=new Image();"
        b"n.onload=function(){i.src=n.src;bad=0;s.textContent='';setTimeout(tick,ms);};"
        b"n.onerror=function(){bad++;s.textContent='reconnecting ('+bad+')';"
        b"setTimeout(tick,Math.min(3000,500*bad));};"
        b"n.src='/snap'+(q?q+'&':'?')+'_='+Date.now();}"
        b"tick();</script>"
        b"</body></html>")


class Preview(Node):
    def __init__(self, detect, show, detect_every=1, rotate180=False):
        # Unique node name per process: this script gets killed and restarted often, and
        # a fresh participant reusing a dead one's name is a way to end up with a node
        # that exists, subscribes, and silently receives nothing.
        super().__init__("preview_server_%d" % os.getpid())
        self.detect = detect
        self.rotate180 = rotate180
        self.detect_every = max(1, detect_every)
        self.frame_no = {}
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
            # DISPLAY ONLY. The modules are mounted inverted (config/rig/rig_layout.yaml,
            # camera_roll_deg: 180) and the capture node publishes the raw sensor
            # orientation - it does no flip, deliberately: the whole extrinsic chain is
            # calibrated in the raw frame and the roll is folded in afterwards by
            # fold_roll_for_vo.py. So this rotation must NEVER reach anything recorded.
            # It exists so the operator sweeping a target is not working upside-down.
            # Consequence to know: the coverage grid below is then in DISPLAY
            # orientation, which is the offline report's grid relabelled by 180 deg.
            if self.rotate180:
                img = np.rot90(img, 2)
            polys = []
            if cam in self.detect:
                small = cv2.resize(img, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                                   interpolation=cv2.INTER_AREA)
                corners, ids = _detect(small)
                cov = coverage.setdefault(
                    cam, np.array(SEED[cam], int) if cam in SEED
                    else np.zeros((GRID, GRID), int))
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
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = x + max(4, (w - tw) // 2)          # centred: the CORNER cells are the ones
        for col, th in ((0, 4), (255, 1)):      # being worked, so keep them unobstructed
            cv2.putText(canvas, text, (tx, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th)

        # Coverage drawn ON THE IMAGE, not as a legend in the corner. A legend tells you
        # THAT a cell is empty; it cannot tell you WHERE to hold the board, and the
        # operator should not have to map grid indices onto the scene in their head.
        # Under-quota cells are outlined with the shortfall printed in them; satisfied
        # cells are left clean so the remaining work is what stands out.
        cg = cov.get(cam)
        if cg is not None:
            cw, ch = w / float(GRID), h / float(GRID)
            for gy in range(GRID):
                for gx in range(GRID):
                    need = QUOTA - int(cg[gy, gx])
                    if need <= 0:
                        continue
                    x0, y0 = int(x + gx * cw), int(y + gy * ch)
                    x1, y1 = int(x + (gx + 1) * cw), int(y + (gy + 1) * ch)
                    # thicker box the emptier the cell, so a wholly untouched corner is
                    # visible at a glance from across the room
                    th = 3 if need >= QUOTA else (2 if need > QUOTA // 2 else 1)
                    cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), 0, th + 2)
                    cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), 255, th)
                    cv2.putText(canvas, str(need), (x0 + int(cw / 2) - 8, y0 + int(ch / 2) + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 3)
                    cv2.putText(canvas, str(need), (x0 + int(cw / 2) - 8, y0 + int(ch / 2) + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)


    if cols > 1:
        cv2.line(canvas, (w, 0), (w, h * rows), 200, 1)
    if rows > 1:
        cv2.line(canvas, (0, h), (w * cols, h), 200, 1)
    return canvas, stamp


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        # /reset clears the coverage grid. Coverage accumulated across takes is
        # misleading: it shows cells filled by a sweep whose frames are in a bag that
        # was discarded, so the operator stops early believing a region is covered when
        # the CURRENT recording has never seen it.
        if self.path.split("?")[0] == "/debug":
            with lock:
                info = {k: v.shape for k, v in latest.items()}
                sq = state["seq"]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(("frames held: %s\nseq: %d\nshow: %s\n"
                              % (info, sq, sorted(SHOW))).encode())
            return
        if self.path.split("?")[0] == "/reset":
            with lock:
                for cam, v in coverage.items():
                    v[:] = np.array(SEED[cam]) if cam in SEED else 0
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"coverage cleared\n")
            return

        q = parse_qs(urlparse(self.path).query)
        f = lambda k, d: float(q.get(k, [d])[0])
        scale = max(0.1, min(1.0, f("scale", 0.4)))
        fps = max(1.0, min(15.0, f("fps", 6)))
        quality = int(max(30, min(95, f("q", 75))))

        if urlparse(self.path).path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)   # no %s any more: the page reads its own query string
            return

        # ONE frame per request. The page polls this instead of holding a
        # multipart/x-mixed-replace stream open: a long-lived stream dies whenever the
        # server restarts or the browser drops it, and when it ends CLEANLY no onerror
        # fires - the img simply stops and the dark page background shows through, with
        # nothing wrong server-side. Polling has no connection to lose: each frame is its
        # own short request, and a failed one just retries on the next tick.
        if urlparse(self.path).path == "/snap":
            got = montage(scale, SHOW)
            if got is None:
                self.send_response(503); self.end_headers(); return
            ok, jpg = cv2.imencode(".jpg", got[0], [cv2.IMWRITE_JPEG_QUALITY, quality])
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg.tobytes())
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
    ap.add_argument("--seed", default="",
                    help="JSON of coverage already achieved per camera (from the offline "
                         "report), so the overlay shows only what is still missing")
    ap.add_argument("--rotate180", action="store_true",
                    help="rotate the DISPLAY 180 deg (the modules are mounted inverted). "
                         "Affects nothing that is recorded - the bag comes off DDS untouched.")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--detect-every", type=int, default=1,
                    help="run tag detection on every Nth frame. Detection is the expensive "
                         "part; the target does not move far in 100 ms")
    ap.add_argument("--detect", nargs="*", default=["cam1"],
                    help="cameras to run live tag detection on (the current stage's)")
    ap.add_argument("--show", nargs="*", default=None,
                    help="cameras to display; defaults to --detect. Fewer = less load")
    a = ap.parse_args()

    global SHOW
    SHOW = set(a.show if a.show else a.detect)
    rclpy.init()
    if a.seed:
        import json
        SEED.update(json.load(open(a.seed)))
        print("seeded coverage for: %s" % ", ".join(sorted(SEED)))
    node = Preview(set(a.detect), SHOW, a.detect_every, a.rotate180)

    # ROS spins on the MAIN thread and HTTP serves on a background one, not the other
    # way round. With rclpy.spin in a side thread the subscriptions silently delivered
    # nothing - the node existed, the topics were publishing, another process in the
    # same container received them fine, and this one sat at seq 0 forever with no
    # error anywhere. Serving HTTP from a thread has no such problem.
    srv = Server(("0.0.0.0", a.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("preview on http://0.0.0.0:%d/ showing %s, detecting on %s"
          % (a.port, ",".join(sorted(SHOW)), ",".join(a.detect)), flush=True)
    # A spin_once LOOP, not rclpy.spin(). Measured in this container: rclpy.spin()
    # delivers zero callbacks while an identical node driven by spin_once receives
    # normally - no error, no warning, the subscription simply never fires. Not worth
    # root-causing mid-session when the loop is equivalent and works.
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
