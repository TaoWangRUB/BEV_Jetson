#!/usr/bin/env python3
"""Interactive panorama extrinsics tuner.

Hand-adjust each camera's rotation (yaw/pitch/roll, about its optical axes) and translation
(tx/ty/tz in the rig frame) plus an assumed scene depth, and watch the stitched 360 panorama
re-render live in the browser. Save the result to rig_extrinsics_tuned.yaml when the seams line up.

  python3 scripts/calib/pano_tuner.py            # then open http://localhost:8000
  python3 scripts/calib/pano_tuner.py --images scripts/calib/capture --port 8000

Reuses the exact stitch math from extrinsic_calib.render_panorama (R = camera->rig, 180 flip for the
upside-down mount). Needs the 4 RAW images (camN_image_raw.png) + the KB intrinsics for that res.
"""
import argparse
import io
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import yaml

import extrinsic_calib as ec  # same dir

CAMS = ["cam1", "cam2", "cam3", "cam4"]
LABELS = {"cam1": "cam1 / port c / -Y back", "cam2": "cam2 / port d / -X left",
          "cam3": "cam3 / port e / +Y front", "cam4": "cam4 / port f / +X right"}

STATE = {}        # filled in main(): per-cam yaw/pitch/roll(deg), tx/ty/tz(mm), + depth(m), quality


def current_rots_trans():
    """Apply the UI deltas to the nominal extrinsics -> (rots camera->rig, trans rig-frame t)."""
    rots, trans = {}, {}
    for c in CAMS:
        d = np.deg2rad([STATE[c]["pitch"], STATE[c]["yaw"], STATE[c]["roll"]])  # about cam x,y,z
        rots[c] = R_NOM[c] @ ec.so3_exp(d)
        trans[c] = T_NOM[c] + np.array([STATE[c]["tx"], STATE[c]["ty"], STATE[c]["tz"]]) / 1000.0
    return rots, trans


def render_png():
    rots, trans = current_rots_trans()
    q = STATE["quality"]
    pano = ec.render_panorama(IMGS, INTR, rots, out_w=q, out_h=q * 9 // 32,
                              trans=trans, depth=STATE["depth"])
    ok, buf = cv2.imencode(".png", pano)
    return buf.tobytes()


def save_yaml(path):
    rots, trans = current_rots_trans()
    out = {k: v for k, v in RIG.items()}
    for c in CAMS:
        out["cameras"][c] = dict(RIG["cameras"][c])
        out["cameras"][c]["q_wxyz"] = [round(float(x), 8) for x in ec.R_to_quat(rots[c])]
        out["cameras"][c]["t_xyz_m"] = [round(float(x), 5) for x in trans[c]]
    with open(path, "w") as f:
        f.write("# Hand-tuned by scripts/calib/pano_tuner.py.\n")
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)
    return path


PARAMS = [("yaw", -30, 30, 0.5, "deg"), ("pitch", -30, 30, 0.5, "deg"), ("roll", -30, 30, 0.5, "deg"),
          ("tx", -200, 200, 1, "mm"), ("ty", -200, 200, 1, "mm"), ("tz", -200, 200, 1, "mm")]


def html():
    rows = ""
    for c in CAMS:
        sliders = ""
        for name, lo, hi, step, unit in PARAMS:
            sliders += (f'<div class=s><label>{name} <span id="{c}_{name}_v">{STATE[c][name]}</span>{unit}</label>'
                        f'<input type=range min={lo} max={hi} step={step} value="{STATE[c][name]}" '
                        f'oninput="upd(\'{c}\',\'{name}\',this.value)"></div>')
        rows += f'<fieldset><legend>{LABELS[c]}</legend>{sliders}</fieldset>'
    return f"""<!doctype html><html><head><meta charset=utf-8><title>BEV panorama tuner</title>
<style>body{{font-family:sans-serif;margin:0;display:flex;height:100vh}}
#left{{width:360px;overflow:auto;padding:10px;background:#1c1c1c;color:#ddd}}
#right{{flex:1;background:#000;display:flex;align-items:center;justify-content:center}}
img{{max-width:100%}} fieldset{{border:1px solid #444;margin:6px 0}} legend{{color:#6f6;font-size:12px}}
.s{{display:flex;flex-direction:column;font-size:11px;margin:2px 0}} input[type=range]{{width:100%}}
button{{margin:4px 2px;padding:6px}} .g{{display:flex;align-items:center;gap:8px}}</style></head>
<body><div id=left>
<h3>Panorama extrinsics tuner</h3>
<div class=g><label>depth <span id=depth_v>{STATE['depth']}</span>m</label>
<input type=range min=0.3 max=20 step=0.1 value="{STATE['depth']}" oninput="upd('','depth',this.value)"></div>
<div class=g><label>quality</label>
<select onchange="STATE_q(this.value)"><option value=960>fast</option>
<option value=1280 selected>med</option><option value=1920>hi</option></select></div>
{rows}
<button onclick="save()">Save → rig_extrinsics_tuned.yaml</button>
<button onclick="location.href='/reset'">Reset</button>
<div id=msg></div></div>
<div id=right><img id=pano src="/render"></div>
<script>
let t=null;
function refresh(){{document.getElementById('pano').src='/render?_='+Date.now();}}
function upd(c,k,v){{document.getElementById((c?c+'_':'')+k+'_v').innerText=v;
 fetch('/set?cam='+c+'&key='+k+'&val='+v).then(()=>{{clearTimeout(t);t=setTimeout(refresh,60);}});}}
function STATE_q(v){{fetch('/set?cam=&key=quality&val='+v).then(refresh);}}
function save(){{fetch('/save').then(r=>r.text()).then(s=>document.getElementById('msg').innerText=s);}}
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="text/html"):
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            self._send(html())
        elif u.path == "/render":
            self._send(render_png(), "image/png")
        elif u.path == "/set":
            cam = q.get("cam", [""])[0]; key = q["key"][0]; val = float(q["val"][0])
            if key == "depth":
                STATE["depth"] = val
            elif key == "quality":
                STATE["quality"] = int(val)
            else:
                STATE[cam][key] = val
            self._send(b"ok", "text/plain")
        elif u.path == "/save":
            p = save_yaml(os.path.join(os.path.dirname(RIG_PATH), "rig_extrinsics_tuned.yaml"))
            self._send(("saved " + p).encode(), "text/plain")
        elif u.path == "/reset":
            for c in CAMS:
                for name, *_ in PARAMS:
                    STATE[c][name] = 0.0
            self.send_response(303); self.send_header("Location", "/"); self.end_headers()
        else:
            self.send_response(404); self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", default="scripts/config/1640x1232")
    ap.add_argument("--rig", default="config/rig/rig_extrinsics.yaml")
    ap.add_argument("--images", default="scripts/calib/capture")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    global IMGS, INTR, RIG, RIG_PATH, R_NOM, T_NOM
    INTR = {c: ec.load_intrinsics(args.calib_dir, c) for c in CAMS}
    IMGS = {}
    for c in CAMS:
        p = os.path.join(args.images, c + "_image_raw.png")
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise SystemExit("missing image: " + p)
        IMGS[c] = im
    RIG_PATH = args.rig
    with open(args.rig) as f:
        RIG = yaml.safe_load(f)
    R_NOM = {c: ec.quat_to_R(RIG["cameras"][c]["q_wxyz"]) for c in CAMS}
    T_NOM = {c: np.array(RIG["cameras"][c]["t_xyz_m"], float) for c in CAMS}
    for c in CAMS:
        STATE[c] = {name: 0.0 for name, *_ in PARAMS}
    STATE["depth"] = 3.0
    STATE["quality"] = 1280

    print(f"panorama tuner -> http://localhost:{args.port}  (Ctrl-C to stop)")
    HTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
