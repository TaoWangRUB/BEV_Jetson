//! Pyramidal Lucas-Kanade for the verification step (openspec design D2): track a landmark's
//! stored appearance from its observation in the source keyframe to the current keyframe,
//! starting at the position the pose guess predicts. The same shape as OpenCV's
//! `calcOpticalFlowPyrLK` with `OPTFLOW_USE_INITIAL_FLOW`, which the Python reference used:
//! a 5-tap Gaussian pyramid, a square window, 30 iterations to 0.01 px per level, and the
//! residual `err` as the mean absolute difference over the window at the solution.
//! Numerically it is not OpenCV, so the Python fixtures hold the port to edges and poses within
//! tolerance, not to bytes.
use nalgebra::Vector2;

/// One pyramid level: 8-bit like OpenCV's, with a reflected (BORDER_REFLECT_101) border of
/// `pad` pixels on every side so a window may partly leave the image, as it may in
/// `calcOpticalFlowPyrLK`.
#[derive(Debug, Clone)]
pub struct Level {
    pub w: usize,
    pub h: usize,
    pub pad: usize,
    stride: usize,
    data: Vec<u8>,
}

#[inline]
fn refl(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    if i < 0 {
        i = -i;
    }
    if i >= n {
        i = 2 * n - 2 - i;
    }
    i.clamp(0, n - 1) as usize
}

impl Level {
    fn from_plain(plain: &[u8], w: usize, h: usize, pad: usize) -> Self {
        let stride = w + 2 * pad;
        let mut data = vec![0u8; stride * (h + 2 * pad)];
        for y in 0..h + 2 * pad {
            let sy = refl(y as isize - pad as isize, h);
            let row = &plain[sy * w..sy * w + w];
            let out = &mut data[y * stride..y * stride + stride];
            for (x, o) in out.iter_mut().enumerate() {
                *o = row[refl(x as isize - pad as isize, w)];
            }
        }
        Self {
            w,
            h,
            pad,
            stride,
            data,
        }
    }

    /// Pixel (x, y) in image coordinates; the border is addressable down to -pad.
    #[cfg(test)]
    #[inline]
    fn at(&self, x: isize, y: isize) -> f32 {
        f32::from(
            self.data
                [(y + self.pad as isize) as usize * self.stride + (x + self.pad as isize) as usize],
        )
    }

    #[inline]
    fn u8_at(&self, x: isize, y: isize) -> u32 {
        u32::from(
            self.data
                [(y + self.pad as isize) as usize * self.stride + (x + self.pad as isize) as usize],
        )
    }

    /// Bilinear sample; the caller keeps (x, y) inside the padded extent (see `inside`).
    #[cfg(test)]
    #[inline]
    fn interp(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let (x0, y0) = (x0 as isize, y0 as isize);
        let a = self.at(x0, y0);
        let b = self.at(x0 + 1, y0);
        let c = self.at(x0, y0 + 1);
        let d = self.at(x0 + 1, y0 + 1);
        (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy
    }

    /// Bilinear samples of the (2·half+1)² window around (x, y), row-major into `out`. Every
    /// sample shares the fractional offset, so the four weights are computed once and the rows
    /// are read contiguously. The caller keeps the window inside the padded extent.
    #[inline]
    fn window(&self, x: f32, y: f32, half: i32, out: &mut [f32]) {
        let xf = x.floor();
        let yf = y.floor();
        let fx = x - xf;
        let fy = y - yf;
        let (w00, w01, w10, w11) = (
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        );
        let n = (2 * half + 1) as usize;
        let x0 = (xf as isize - half as isize + self.pad as isize) as usize;
        let y0 = (yf as isize - half as isize + self.pad as isize) as usize;
        for r in 0..n {
            let s0 = (y0 + r) * self.stride + x0;
            let s1 = s0 + self.stride;
            let row0 = &self.data[s0..s0 + n + 1];
            let row1 = &self.data[s1..s1 + n + 1];
            let o = &mut out[r * n..r * n + n];
            for c in 0..n {
                o[c] = w00 * f32::from(row0[c])
                    + w01 * f32::from(row0[c + 1])
                    + w10 * f32::from(row1[c])
                    + w11 * f32::from(row1[c + 1]);
            }
        }
    }

    /// A window of half-size `half` around (x, y), plus one pixel of gradient support, is
    /// addressable. With pad = win + 2 this is OpenCV's limit: the centre within half a
    /// window of the image.
    #[inline]
    fn inside(&self, x: f32, y: f32, half: f32) -> bool {
        let lo = 1.0 - self.pad as f32;
        x - half - 2.0 >= lo
            && y - half - 2.0 >= lo
            && x + half + 2.0 < (self.w + self.pad - 2) as f32
            && y + half + 2.0 < (self.h + self.pad - 2) as f32
    }
}

pub struct Pyramid {
    pub levels: Vec<Level>,
}

impl Pyramid {
    /// `calcOpticalFlowPyrLK`'s pyramid for a window of `win`: up to `n_levels` levels, built
    /// only while the next level is larger than the window.
    pub fn build(data: &[u8], w: usize, h: usize, n_levels: usize, win: usize) -> Self {
        let pad = win + 2;
        let mut levels = Vec::with_capacity(n_levels);
        levels.push(Level::from_plain(data, w, h, pad));
        for _ in 1..n_levels {
            let prev = levels.last().unwrap();
            let (nw, nh) = (prev.w.div_ceil(2), prev.h.div_ceil(2));
            if nw <= win || nh <= win {
                break;
            }
            let plain = pyr_down(prev);
            levels.push(Level::from_plain(&plain, nw, nh, pad));
        }
        Self { levels }
    }
}

/// OpenCV's pyrDown on 8-bit: 5x5 Gaussian [1 4 6 4 1]^2 / 256 with reflect-101 borders, every
/// second pixel, rounded; bit-exact with cv2.pyrDown.
fn pyr_down(src: &Level) -> Vec<u8> {
    let (w, h) = (src.w, src.h);
    let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
    let k = [1u32, 4, 6, 4, 1];
    let mut tmp = vec![0u32; ow * h];
    for y in 0..h {
        for ox in 0..ow {
            let cx = 2 * ox as isize;
            let mut s = 0u32;
            for (t, kk) in k.iter().enumerate() {
                s += kk * src.u8_at(refl(cx + t as isize - 2, w) as isize, y as isize);
            }
            tmp[y * ow + ox] = s;
        }
    }
    let mut out = vec![0u8; ow * oh];
    for oy in 0..oh {
        let cy = 2 * oy as isize;
        for ox in 0..ow {
            let mut s = 0u32;
            for (t, kk) in k.iter().enumerate() {
                s += kk * tmp[refl(cy + t as isize - 2, h) * ow + ox];
            }
            out[oy * ow + ox] = ((s + 128) >> 8) as u8;
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub struct KltParams {
    pub win: usize,
    pub levels: usize,
    pub iters: usize,
    pub eps: f32,
}

/// Track `p0` of `prev` into `next`, starting from `init` at full resolution.
/// Returns the position in `next` and the mean absolute window residual, or None when the
/// window leaves the image or the system is singular.
pub fn track(
    prev: &Pyramid,
    next: &Pyramid,
    p0: Vector2<f64>,
    init: Vector2<f64>,
    prm: &KltParams,
) -> Option<(Vector2<f64>, f32)> {
    let half = (prm.win / 2) as i32;
    let halff = half as f32;
    // the pyramids hold levels only while they are larger than the window (OpenCV)
    let top = prm
        .levels
        .min(prev.levels.len() - 1)
        .min(next.levels.len() - 1);
    let scale_top = (1u32 << top) as f32;
    // flow is tracked as the displacement from p0's position at each level
    let mut d = Vector2::new((init.x - p0.x) as f32, (init.y - p0.y) as f32) / scale_top;
    let mut err = 0.0f32;
    for lvl in (0..=top).rev() {
        let s = (1u32 << lvl) as f32;
        let a = &prev.levels[lvl];
        let b = &next.levels[lvl];
        let px = p0.x as f32 / s;
        let py = p0.y as f32 / s;
        if !a.inside(px, py, halff + 1.0) {
            return None;
        }
        // template values and gradients (central differences) over the window, from one
        // sample of the window grown by a pixel on every side
        let n = prm.win * prm.win;
        let m = prm.win + 2;
        let mut wt = vec![0.0f32; m * m];
        a.window(px, py, half + 1, &mut wt);
        let mut tv = Vec::with_capacity(n);
        let mut gx = Vec::with_capacity(n);
        let mut gy = Vec::with_capacity(n);
        let (mut g00, mut g01, mut g11) = (0.0f32, 0.0f32, 0.0f32);
        for r in 0..prm.win {
            for c in 0..prm.win {
                let v = wt[(r + 1) * m + c + 1];
                let ix = (wt[(r + 1) * m + c + 2] - wt[(r + 1) * m + c]) * 0.5;
                let iy = (wt[(r + 2) * m + c + 1] - wt[r * m + c + 1]) * 0.5;
                tv.push(v);
                gx.push(ix);
                gy.push(iy);
                g00 += ix * ix;
                g01 += ix * iy;
                g11 += iy * iy;
            }
        }
        let det = g00 * g11 - g01 * g01;
        let min_eig = 0.5 * ((g00 + g11) - ((g00 - g11).powi(2) + 4.0 * g01 * g01).sqrt());
        if det < 1e-6 || min_eig / (n as f32) < 0.1024 {
            return None;
        }
        let inv = [g11 / det, -g01 / det, g00 / det];
        let mut wb = vec![0.0f32; n];
        for _ in 0..prm.iters {
            let qx = px + d.x;
            let qy = py + d.y;
            if !b.inside(qx, qy, halff) {
                return None;
            }
            b.window(qx, qy, half, &mut wb);
            let (mut b0, mut b1) = (0.0f32, 0.0f32);
            for k in 0..n {
                let diff = tv[k] - wb[k];
                b0 += diff * gx[k];
                b1 += diff * gy[k];
            }
            let step = Vector2::new(inv[0] * b0 + inv[1] * b1, inv[1] * b0 + inv[2] * b1);
            d += step;
            if step.norm_squared() < prm.eps * prm.eps {
                break;
            }
        }
        if lvl == 0 {
            let qx = px + d.x;
            let qy = py + d.y;
            if !b.inside(qx, qy, halff) {
                return None;
            }
            b.window(qx, qy, half, &mut wb);
            let e: f32 = tv.iter().zip(wb.iter()).map(|(t, v)| (t - v).abs()).sum();
            err = e / n as f32;
        } else {
            d *= 2.0;
        }
    }
    Some((
        Vector2::new(f64::from(p0.x as f32 + d.x), f64::from(p0.y as f32 + d.y)),
        err,
    ))
}

/// Normalised cross-correlation of two `size`x`size` bilinear patches, or None at the border.
pub fn ncc(a: &Level, pa: Vector2<f64>, b: &Level, pb: Vector2<f64>, size: usize) -> Option<f32> {
    let half = (size / 2) as i32;
    let hf = half as f32 + 1.0;
    let (ax, ay, bx, by) = (pa.x as f32, pa.y as f32, pb.x as f32, pb.y as f32);
    if !a.inside(ax, ay, hf) || !b.inside(bx, by, hf) {
        return None;
    }
    let n = (size * size) as f32;
    let mut va = vec![0.0f32; size * size];
    let mut vb = vec![0.0f32; size * size];
    a.window(ax, ay, half, &mut va);
    b.window(bx, by, half, &mut vb);
    let ma = va.iter().sum::<f32>() / n;
    let mb = vb.iter().sum::<f32>() / n;
    let (mut num, mut da, mut db) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..va.len() {
        let x = va[k] - ma;
        let y = vb[k] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    let d = (da * db).sqrt();
    if d > 1e-6 { Some(num / d) } else { Some(-1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(w: usize, h: usize, shift: (f32, f32)) -> (Vec<u8>, Vec<u8>) {
        // smooth random texture, and the same texture shifted by `shift` (bilinear)
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut noise = vec![0.0f32; (w + 16) * (h + 16)];
        for v in noise.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            *v = (rng % 256) as f32;
        }
        let noise8: Vec<u8> = noise.iter().map(|v| *v as u8).collect();
        let lvl = Level::from_plain(&noise8, w + 16, h + 16, 2);
        // blur a little so gradients are smooth
        let sample = |x: f32, y: f32| -> f32 {
            let mut s = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    s += lvl.interp(x + 8.0 + dx as f32, y + 8.0 + dy as f32);
                }
            }
            s / 9.0
        };
        let mut a = vec![0u8; w * h];
        let mut b = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                a[y * w + x] = sample(x as f32, y as f32) as u8;
                b[y * w + x] = sample(x as f32 - shift.0, y as f32 - shift.1) as u8;
            }
        }
        (a, b)
    }

    #[test]
    fn tracks_a_known_shift_from_a_rough_guess() {
        let (w, h) = (160, 120);
        let (a, b) = synthetic(w, h, (6.3, -3.7));
        let pa = Pyramid::build(&a, w, h, 5, 21);
        let pb = Pyramid::build(&b, w, h, 5, 21);
        let prm = KltParams {
            win: 21,
            levels: 4,
            iters: 30,
            eps: 0.01,
        };
        let mut ok = 0;
        for &(x, y) in &[(40.0, 40.0), (80.0, 60.0), (120.0, 90.0), (60.0, 30.0)] {
            let p0 = Vector2::new(x, y);
            let init = Vector2::new(x + 2.0, y - 1.0); // a guess ~5 px off
            let (p1, err) = track(&pa, &pb, p0, init, &prm).expect("tracked");
            let e = (p1 - Vector2::new(x + 6.3, y - 3.7)).norm();
            assert!(e < 0.5, "error {e} px at ({x},{y}), err {err}");
            ok += 1;
        }
        assert_eq!(ok, 4);
        let (p1, _) = track(
            &pa,
            &pb,
            Vector2::new(80.0, 60.0),
            Vector2::new(82.0, 59.0),
            &prm,
        )
        .unwrap();
        let (back, _) = track(&pb, &pa, p1, Vector2::new(80.0, 60.0), &prm).unwrap();
        assert!((back - Vector2::new(80.0, 60.0)).norm() < 0.5);
        let c = ncc(
            &pa.levels[0],
            Vector2::new(80.0, 60.0),
            &pb.levels[0],
            p1,
            15,
        )
        .unwrap();
        assert!(c > 0.9, "ncc {c}");
    }
}
