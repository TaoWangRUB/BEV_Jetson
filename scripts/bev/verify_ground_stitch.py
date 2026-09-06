#!/usr/bin/env python3
"""Check the BEV ground stitch without the rig, by giving it a floor it already knows.

WHY THIS EXISTS. The ground projection is exact by construction, so testing the node
against its own maps proves nothing. What can actually be wrong is the INPUT to that
construction: the rig frame, the handedness, the sign of the plane, which camera is
which. Every one of those produces a stitch that still looks like a picture of a floor.

So: render what each fisheye would see of a known ground texture -- written from the
physics (camera pose -> ray -> plane -> texture), independently of the node's mapping --
publish those four images as a synchronised set, and check the node reproduces the
texture it started from. A frame error cannot survive the round trip.

The texture is deliberately ASYMMETRIC: a bright bar along +forward and a dark bar along
+left. A 180-degree roll, a mirrored axis or a swapped pair moves them somewhere they
cannot be, which a checkerboard alone would hide.

Run inside a ROS 2 container with the node already running:
    ros2 run bev_ground bev_ground_stitch_node --ros-args ... &
    python3 scripts/bev/verify_ground_stitch.py --height 0.30 \
        --rig datasets/calib_20260901/closed.yaml \
        --calib datasets/calib_20260901/chains
"""
import argparse, os, sys, time
import numpy as np, yaml, cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy,
                       QoSHistoryPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import Image
from std_msgs.msg import String

CAMS = ['cam1', 'cam2', 'cam3', 'cam4']
TEX_M_PER_PX = 0.005
TEX_HALF = 6.0
CHECK_PITCH_M = 0.25


def ground_texture():
    n = int(2 * TEX_HALF / TEX_M_PER_PX)
    xs = TEX_HALF - (np.arange(n) + 0.5) * TEX_M_PER_PX     # forward; row 0 = most forward
    ys = TEX_HALF - (np.arange(n) + 0.5) * TEX_M_PER_PX     # left;    col 0 = most left
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    tex = (((np.floor(X / CHECK_PITCH_M) + np.floor(Y / CHECK_PITCH_M)) % 2) * 80 + 60).astype(np.float32)
    tex[(np.abs(Y) < 0.05) & (X > 0.3) & (X < 1.8)] = 245   # BRIGHT bar -> forward
    tex[(np.abs(X) < 0.05) & (Y > 0.3) & (Y < 1.8)] = 15    # DARK bar   -> left
    return tex


def sample_ground(tex, X, Y, valid):
    r = (TEX_HALF - X) / TEX_M_PER_PX - 0.5
    c = (TEX_HALF - Y) / TEX_M_PER_PX - 0.5
    ok = valid & (r >= 0) & (c >= 0) & (r < tex.shape[0] - 1) & (c < tex.shape[1] - 1)
    img = cv2.remap(tex, np.where(ok, c, -1).astype(np.float32),
                    np.where(ok, r, -1).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.where(ok, img, 0).astype(np.uint8), ok


def load_omni(path):
    y = yaml.safe_load(open(path))
    y = y.get('cam0', y)
    xi, fx, fy, cx, cy = y['intrinsics']
    k1, k2, p1, p2 = y['distortion_coeffs']
    w, h = y['resolution']
    return dict(xi=xi, fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2, w=w, h=h)


def unproject(o, u, v):
    """Pixel -> unit ray in the camera frame: undo radtan by fixed point, then Mei's
    closed-form lift to the unit sphere. The inverse of the node's ProjectOmni."""
    xd = (u - o['cx']) / o['fx']
    yd = (v - o['cy']) / o['fy']
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        r2 = xu * xu + yu * yu
        rad = 1 + o['k1'] * r2 + o['k2'] * r2 * r2
        dx = 2 * o['p1'] * xu * yu + o['p2'] * (r2 + 2 * xu * xu)
        dy = o['p1'] * (r2 + 2 * yu * yu) + 2 * o['p2'] * xu * yu
        xu, yu = (xd - dx) / rad, (yd - dy) / rad
    r2 = xu * xu + yu * yu
    disc = 1 + (1 - o['xi'] ** 2) * r2
    ok = disc >= 0
    f = (o['xi'] + np.sqrt(np.where(ok, disc, 0.0))) / (1 + r2)
    return np.stack([f * xu, f * yu, f - o['xi']], -1), ok


def render(repo, calib_dir, rig_path, plane_path, height):
    R_rig_c1 = np.array(yaml.safe_load(open(plane_path))['rig_frame']['R_rig_cam1'], float)
    rig = yaml.safe_load(open(rig_path))['rig_in_cam1']
    tex = ground_texture()
    n_plane = np.array([0.0, 0.0, 1.0])
    imgs = {}
    for name in CAMS:
        o = load_omni(os.path.join(calib_dir, f'{name}.yaml'))
        T = np.array(rig[name], float)
        R_rig_cam = R_rig_c1 @ T[:3, :3]
        t_rig_cam = R_rig_c1 @ T[:3, 3]
        vv, uu = np.mgrid[0:o['h'], 0:o['w']].astype(np.float64)
        d_cam, ok = unproject(o, uu, vv)
        d_rig = d_cam @ R_rig_cam.T
        denom = d_rig @ n_plane
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (-height - (t_rig_cam @ n_plane)) / denom
        # the ray must actually go DOWN and hit in front of the camera
        hit = ok & np.isfinite(t) & (t > 0.02) & (denom < -1e-9)
        P = t_rig_cam + d_rig * t[..., None]
        img, _ = sample_ground(tex, P[..., 0], P[..., 1], hit)
        imgs[name] = img
        print(f'  rendered {name}: {100 * hit.mean():5.1f}% of the sensor sees the floor, '
              f'at (F {t_rig_cam[0]:+.3f}, L {t_rig_cam[1]:+.3f}, U {t_rig_cam[2]:+.3f}) m')
    return imgs, tex


class Harness(Node):
    def __init__(self, imgs, tex, args):
        super().__init__('bev_ground_verify')
        self.imgs, self.tex, self.args = imgs, tex, args
        self.info, self.got = None, None
        # The node SUBSCRIBES best-effort (SensorDataQoS) because on the rig a late frame
        # is worthless. Here the four 1.5 MB images come from one Python process, so a
        # best-effort publisher simply loses some of them — and the node then sees three
        # cameras from one edge and one from an earlier edge, and correctly refuses the
        # set. Publish RELIABLE (compatible with a best-effort subscriber, which accepts a
        # stronger offer) and slowly, so the test exercises the geometry and not the
        # transport.
        pub_qos = QoSProfile(depth=20, history=QoSHistoryPolicy.KEEP_LAST,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.pubs = {c: self.create_publisher(Image, f'/{c}/image_raw', pub_qos)
                     for c in CAMS}
        self.create_subscription(Image, '/bev/ground', self.on_bev, qos_profile_sensor_data)
        self.create_subscription(String, '/bev/ground/info', self.on_info,
                                 QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                                            reliability=QoSReliabilityPolicy.RELIABLE,
                                            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.timer = self.create_timer(0.2, self.tick)
        self.n = 0

    def on_info(self, msg):
        self.info = yaml.safe_load(msg.data)

    def on_bev(self, msg):
        self.got = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width).copy()

    def tick(self):
        # One trigger edge: all four frames carry the SAME stamp, which is what the node's
        # skew gate expects of a hardware-triggered rig.
        stamp = self.get_clock().now().to_msg()
        for c in CAMS:
            im = self.imgs[c]
            m = Image()
            m.header.stamp = stamp
            m.header.frame_id = c
            m.height, m.width = im.shape
            m.encoding = 'mono8'
            m.step = im.shape[1]
            m.data = im.tobytes()
            self.pubs[c].publish(m)
        self.n += 1


def score(out, info, tex, args):
    res = float(info['resolution_m_per_px'])
    H, W = out.shape
    rows = np.arange(H); cols = np.arange(W)
    X = float(info['range_forward_m']) - (rows + 0.5) * res
    Y = float(info['range_left_m']) - (cols + 0.5) * res
    XX, YY = np.meshgrid(X, Y, indexing='ij')
    truth, _ = sample_ground(tex, XX, YY, np.ones_like(XX, bool))

    cov = out > 0
    err = np.abs(out[cov].astype(float) - truth[cov].astype(float))
    print(f'\n  covered cells          : {100 * cov.mean():.1f}%')
    print(f'  |output - truth|       : mean {err.mean():.1f}, median {np.median(err):.1f}, '
          f'p95 {np.percentile(err, 95):.1f} grey levels')
    print(f'  cells within 20 levels : {100 * (err < 20).mean():.1f}%')

    # Landmarks. A frame error puts these somewhere they cannot be.
    def centroid(mask):
        if mask.sum() < 50:
            return None
        return XX[mask].mean(), YY[mask].mean(), mask.sum()
    fwd = centroid(cov & (out > 225))
    left = centroid(cov & (out < 35))
    print('\n  landmark check (the test a checkerboard alone cannot do):')
    for label, c, exp in (('bright bar, expected along +forward at y=0', fwd, (1.05, 0.0)),
                          ('dark bar,   expected along +left    at x=0', left, (0.0, 1.05))):
        if c is None:
            print(f'    {label}: NOT FOUND  <-- the output does not contain it')
            continue
        d = np.hypot(c[0] - exp[0], c[1] - exp[1])
        verdict = 'OK' if d < 0.15 else 'WRONG PLACE'
        print(f'    {label}: centroid (F {c[0]:+.3f}, L {c[1]:+.3f}) m, '
              f'{c[2]} cells, {d * 1000:.0f} mm from expected -> {verdict}')

    # Scale, straight off the checkerboard and independently of the landmarks. Measured
    # as the spacing between checker edges rather than by an FFT: over a 4 m extent the
    # spectrum's largest peak is the coverage envelope, not the 0.25 m squares.
    row = int((float(info['range_forward_m']) - 0.8) / res)      # a band at x = +0.8 m,
    band = out[row - 10:row + 10, :].astype(float)               # clear of both landmarks
    keep = (np.abs(Y) > 0.12) & cov[row - 10:row + 10, :].all(0)
    prof = np.where(keep, band.mean(0), np.nan)
    sign = np.sign(prof - np.nanmean(prof))
    cross = [i for i in range(1, len(sign))
             if np.isfinite(sign[i]) and np.isfinite(sign[i - 1]) and sign[i] != sign[i - 1]]
    gaps = np.diff(cross)
    gaps = gaps[(gaps > 5) & (gaps < 200)]
    if len(gaps) >= 4:
        sq = np.median(gaps) * res
        print(f'\n  checker square measured at x=+0.8 m: {sq * 1000:.0f} mm from '
              f'{len(gaps)} edges (true {CHECK_PITCH_M * 1000:.0f} mm, '
              f'{100 * (sq / CHECK_PITCH_M - 1):+.1f}%)')

    # Per-seam error. The four seams sit on the bisectors between adjacent cameras, i.e.
    # straight forward, left, back and right of the rig. If the projection were wrong the
    # error would concentrate there, so quote those bands against the rest.
    az = np.degrees(np.arctan2(YY, XX))
    rad = np.hypot(XX, YY)
    far = cov & (rad > 0.5)
    diff = np.abs(out.astype(float) - truth.astype(float))
    print('\n  per-seam error (the bands where two cameras must agree):')
    for name, a in (('forward (cam1|cam2)', 0.0), ('left  (cam3|cam1)', 90.0),
                    ('back  (cam4|cam3)', 180.0), ('right (cam2|cam4)', -90.0)):
        band_m = far & (np.abs((az - a + 180) % 360 - 180) < 10)
        if band_m.sum() > 100:
            print(f'    {name:22s}: mean {diff[band_m].mean():5.2f}, '
                  f'p95 {np.percentile(diff[band_m], 95):5.1f} grey levels '
                  f'({band_m.sum()} cells)')
    rest = far & (np.abs((np.abs(az) % 90) - 45) < 25)      # away from every seam
    print(f'    {"away from any seam":22s}: mean {diff[rest].mean():5.2f}, '
          f'p95 {np.percentile(diff[rest], 95):5.1f} grey levels ({rest.sum()} cells)')
    return err


def main():
    ap = argparse.ArgumentParser()
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--repo', default=repo)
    ap.add_argument('--calib', default='config/calib/imx296_1456x1088')
    ap.add_argument('--rig', default='config/rig/rig_extrinsics_imx296.yaml')
    ap.add_argument('--plane', default='config/rig/ground_plane.yaml')
    ap.add_argument('--height', type=float, default=0.30)
    ap.add_argument('--timeout', type=float, default=60.0)
    ap.add_argument('--save', default='')
    # Keep publishing after the first result, so the node reaches steady state and its
    # own 5-second timing report means something. One set tells you nothing about rate.
    ap.add_argument('--hold', type=float, default=0.0)
    args = ap.parse_args()
    j = lambda p: p if os.path.isabs(p) else os.path.join(args.repo, p)

    print(f'rendering a known floor at h={args.height} m through {args.calib} / {args.rig}')
    imgs, tex = render(args.repo, j(args.calib), j(args.rig), j(args.plane), args.height)

    rclpy.init()
    node = Harness(imgs, tex, args)
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.got is not None and node.info is not None and node.n > 3:
            break
    if node.got is None or node.info is None:
        print('\nFAIL: no /bev/ground (or no /bev/ground/info) within the timeout. '
              'Is the node running, and did it refuse to start?')
        node.destroy_node(); rclpy.shutdown(); return 2
    print(f'\ngot a {node.got.shape[1]}x{node.got.shape[0]} stitch at '
          f'{float(node.info["resolution_m_per_px"]) * 1000:.0f} mm/px, plane '
          f'{node.info["plane_status"]}')
    err = score(node.got, node.info, tex, args)
    if args.hold > 0:
        print(f'\n  holding {args.hold:.0f} s so the node can report a steady-state rate...')
        t1 = time.time()
        while rclpy.ok() and time.time() - t1 < args.hold:
            rclpy.spin_once(node, timeout_sec=0.1)
    if args.save:
        cv2.imwrite(args.save, node.got)
        print(f'\n  wrote {args.save}')
    node.destroy_node(); rclpy.shutdown()
    return 0 if np.median(err) < 25 else 1


if __name__ == '__main__':
    sys.exit(main())
