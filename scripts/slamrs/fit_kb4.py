"""Fit Basalt KB4 intrinsics to the tartancalib Mei (omni + radtan) solves.

Pixels on a grid are unprojected through Mei+radtan, then KB4 is least-squares fitted
to map those rays back to the same pixels. Output at a chosen downscale.
"""
import json
import sys

import numpy as np
import yaml


def least_squares_lm(fun, p0, iters=200):
    p = np.asarray(p0, float); lam = 1e-3
    r = fun(p); cost = r @ r
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for i in range(p.size):
            h = 1e-6 * max(1.0, abs(p[i])); q = p.copy(); q[i] += h
            J[:, i] = (fun(q) - r) / h
        A = J.T @ J; g = J.T @ r
        while True:
            dp = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            rn = fun(p + dp); cn = rn @ rn
            if cn < cost:
                p, r, lam = p + dp, rn, lam * 0.3
                break
            lam *= 10
            if lam > 1e12:
                return p
        if abs(cost - cn) < 1e-12 * cost:
            cost = cn; break
        cost = cn
    return p

from pathlib import Path

CALIB = str(Path(__file__).resolve().parents[2] / "config/calib/imx296_1456x1088/cam{}.yaml")


def mei_unproject(u, v, xi, fx, fy, cx, cy, k1, k2, p1, p2):
    mx = (u - cx) / fx
    my = (v - cy) / fy
    # invert radtan by fixed-point iteration
    x, y = mx.copy(), my.copy()
    for _ in range(40):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (mx - dx) / rad
        y = (my - dy) / rad
    r2 = x * x + y * y
    disc = 1 + (1 - xi * xi) * r2
    ok = disc >= 0
    disc = np.clip(disc, 0, None)
    f = (xi + np.sqrt(disc)) / (r2 + 1)
    X = np.stack([f * x, f * y, f - xi], -1)
    X /= np.linalg.norm(X, axis=-1, keepdims=True)
    # round-trip check: re-distort must reproduce the pixel
    r2c = x * x + y * y
    radc = 1 + k1 * r2c + k2 * r2c * r2c
    ux = fx * (x * radc + 2 * p1 * x * y + p2 * (r2c + 2 * x * x)) + cx
    vy = fy * (y * radc + p1 * (r2c + 2 * y * y) + 2 * p2 * x * y) + cy
    ok &= np.hypot(ux - u, vy - v) < 1e-3
    return X, ok


def kb4_project(p, X):
    fx, fy, cx, cy, k1, k2, k3, k4 = p
    r = np.hypot(X[:, 0], X[:, 1])
    th = np.arctan2(r, X[:, 2])
    t2 = th * th
    d = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    s = np.where(r > 1e-9, d / np.maximum(r, 1e-12), 1.0)
    return np.stack([fx * s * X[:, 0] + cx, fy * s * X[:, 1] + cy], -1)


def fit(cam, scale):
    c = yaml.safe_load(open(CALIB.format(cam)))
    xi, fx, fy, cx, cy = c["intrinsics"]
    k1, k2, p1, p2 = c["distortion_coeffs"]
    W, H = c["resolution"]
    uu, vv = np.meshgrid(np.arange(0, W, 8.0), np.arange(0, H, 8.0))
    u, v = uu.ravel(), vv.ravel()
    X, ok = mei_unproject(u, v, xi, fx, fy, cx, cy, k1, k2, p1, p2)
    th = np.degrees(np.arccos(np.clip(X[:, 2], -1, 1)))
    ok &= th < 100.0
    X, uv = X[ok], np.stack([u[ok], v[ok]], -1)
    p0 = [fx / (1 + xi), fy / (1 + xi), cx, cy, 0, 0, 0, 0]
    px = least_squares_lm(lambda p: (kb4_project(p, X) - uv).ravel(), p0)
    e = np.linalg.norm(kb4_project(px, X) - uv, axis=1)
    thk = np.degrees(np.arccos(np.clip(X[:, 2], -1, 1)))
    p = px.copy()
    # downscale: pixel centres map as (u + 0.5) / s - 0.5
    p[0] /= scale
    p[1] /= scale
    p[2] = (p[2] + 0.5) / scale - 0.5
    p[3] = (p[3] + 0.5) / scale - 0.5
    stats = dict(cam=cam, n=len(e), max_theta_deg=float(thk.max()),
                 rms_px=float(np.sqrt((e ** 2).mean())), p99_px=float(np.percentile(e, 99)),
                 max_px=float(e.max()),
                 rms_px_theta_lt_80=float(np.sqrt((e[thk < 80] ** 2).mean())))
    return p, stats, (W // scale, H // scale)


if __name__ == "__main__":
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for cam in (1, 2, 3, 4):
        print(json.dumps(fit(cam, scale)[1]))
