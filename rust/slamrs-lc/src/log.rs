//! The BEV raw log (scripts/log_rig.sh output): `camN.raw` + `camN_index.csv`, `imu0.csv`,
//! `geometry.txt`. The same reader as scripts/slamrs/run_slamrs.py, step for step, so the VIO
//! sees the same framesets and IMU samples and its output is comparable byte for byte:
//!
//! - a frameset is a cam1 frame plus the nearest frame of each other camera within 1 ms
//!   (binary search, nearest of the two neighbours); its stamp is the integer mean of the four;
//! - IMU stamps are shifted by the calibrated camera-IMU lag; the gyro bias and the along-gravity
//!   accelerometer offset are measured over the first two seconds, when every log holds the rig
//!   still; samples that do not advance time are dropped;
//! - frames are the raw 1456x1088 mono8 halved with OpenCV's INTER_AREA, which for an exact 2x
//!   is the rounded mean of each 2x2 block, `(a + b + c + d + 2) >> 2`.
use std::fs;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

/// `imu_mpu9250.yaml` timeshift_cam_imu: `t_imu = t_cam + shift`, DLPF 184 Hz.
pub const IMU_LAG_NS: i64 = 3_731_525;
/// Frames of the other cameras must sit within this of the cam1 stamp to form a set.
pub const SET_TOL_NS: i64 = 1_000_000;
pub const SCALE: usize = 2;
pub const CAMERAS: usize = 4;

#[derive(Debug, thiserror::Error)]
pub enum LogError {
    #[error("{0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("{0}: {1}")]
    Parse(PathBuf, String),
}

#[derive(Debug, Clone)]
pub struct Imu {
    pub t: Vec<i64>,
    pub gyr: Vec<[f64; 3]>,
    pub acc: Vec<[f64; 3]>,
    /// |a| at rest before the gravity correction (10.33–10.35 on this MPU-9250).
    pub a0_norm: f64,
}

#[allow(dead_code)]
pub struct RawLog {
    pub dir: PathBuf,
    pub width: usize,
    pub height: usize,
    pub bytes_per_frame: usize,
    raws: Vec<Mmap>,
    /// (stamp, byte offset per camera) of every complete set, in cam1 order.
    pub sets: Vec<(i64, [usize; CAMERAS])>,
    pub n_cam1: usize,
    pub imu: Imu,
}

fn read_text(path: &Path) -> Result<String, LogError> {
    fs::read_to_string(path).map_err(|e| LogError::Io(path.to_path_buf(), e))
}

fn parse_index(path: &Path) -> Result<Vec<(i64, usize)>, LogError> {
    let mut out = Vec::new();
    for line in read_text(path)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("stamp") {
            continue;
        }
        let mut it = line.split(',');
        let (Some(a), Some(b)) = (it.next(), it.next()) else {
            return Err(LogError::Parse(
                path.to_path_buf(),
                format!("bad row {line:?}"),
            ));
        };
        let t: i64 = a
            .trim()
            .parse()
            .map_err(|_| LogError::Parse(path.to_path_buf(), format!("bad stamp {a:?}")))?;
        let off: usize = b
            .trim()
            .parse()
            .map_err(|_| LogError::Parse(path.to_path_buf(), format!("bad offset {b:?}")))?;
        out.push((t, off));
    }
    Ok(out)
}

fn parse_imu(path: &Path) -> Result<Imu, LogError> {
    let mut t = Vec::new();
    let mut acc = Vec::new();
    let mut gyr = Vec::new();
    for line in read_text(path)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let v: Vec<f64> = line
            .split(',')
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|_| LogError::Parse(path.to_path_buf(), format!("bad row {line:?}")))?;
        if v.len() < 7 {
            return Err(LogError::Parse(
                path.to_path_buf(),
                format!("expected 7+ columns, got {}", v.len()),
            ));
        }
        // numpy loads the stamp as float64 and casts; 4e12 is exact either way
        t.push(v[0] as i64 - IMU_LAG_NS);
        acc.push([v[1], v[2], v[3]]);
        gyr.push([v[4], v[5], v[6]]);
    }
    if t.is_empty() {
        return Err(LogError::Parse(path.to_path_buf(), "no samples".into()));
    }
    // the rig is still for the first two seconds of every log
    let t_rest = t[0] + 2_000_000_000;
    let n_rest = t.iter().filter(|&&x| x < t_rest).count().max(1);
    let mean = |rows: &[[f64; 3]]| -> [f64; 3] {
        let mut s = [0.0; 3];
        for r in rows.iter().take(n_rest) {
            for k in 0..3 {
                s[k] += r[k];
            }
        }
        [
            s[0] / n_rest as f64,
            s[1] / n_rest as f64,
            s[2] / n_rest as f64,
        ]
    };
    let g0 = mean(&gyr);
    let a0 = mean(&acc);
    let a0n = (a0[0] * a0[0] + a0[1] * a0[1] + a0[2] * a0[2]).sqrt();
    // |a| at rest reads ~10.4 (uncalibrated zero-g offset). Only the component along gravity
    // is observable from one pose, so remove exactly that.
    let corr = [
        a0[0] - 9.80665 * a0[0] / a0n,
        a0[1] - 9.80665 * a0[1] / a0n,
        a0[2] - 9.80665 * a0[2] / a0n,
    ];
    let mut out = Imu {
        t: Vec::new(),
        gyr: Vec::new(),
        acc: Vec::new(),
        a0_norm: a0n,
    };
    for i in 0..t.len() {
        if i > 0 && t[i] <= t[i - 1] {
            continue; // keep only samples that advance time
        }
        out.t.push(t[i]);
        out.gyr
            .push([gyr[i][0] - g0[0], gyr[i][1] - g0[1], gyr[i][2] - g0[2]]);
        out.acc.push([
            acc[i][0] - corr[0],
            acc[i][1] - corr[1],
            acc[i][2] - corr[2],
        ]);
    }
    Ok(out)
}

impl RawLog {
    pub fn open(dir: &Path) -> Result<Self, LogError> {
        let geo_path = dir.join("geometry.txt");
        let mut width = 0;
        let mut height = 0;
        let mut bpf = 0;
        for line in read_text(&geo_path)?.lines() {
            let mut it = line.split_whitespace();
            match (it.next(), it.next()) {
                (Some("width"), Some(v)) => width = v.parse().unwrap_or(0),
                (Some("height"), Some(v)) => height = v.parse().unwrap_or(0),
                (Some("bytes_per_frame"), Some(v)) => bpf = v.parse().unwrap_or(0),
                _ => {}
            }
        }
        if width == 0 || height == 0 || bpf != width * height {
            return Err(LogError::Parse(
                geo_path,
                format!("width {width} height {height} bytes_per_frame {bpf}"),
            ));
        }
        let mut raws = Vec::with_capacity(CAMERAS);
        let mut indices = Vec::with_capacity(CAMERAS);
        for c in 1..=CAMERAS {
            let raw_path = dir.join(format!("cam{c}.raw"));
            let file = fs::File::open(&raw_path).map_err(|e| LogError::Io(raw_path.clone(), e))?;
            // SAFETY: the log is read-only data nothing writes to while we replay it.
            let map = unsafe { Mmap::map(&file) }.map_err(|e| LogError::Io(raw_path, e))?;
            raws.push(map);
            indices.push(parse_index(&dir.join(format!("cam{c}_index.csv")))?);
        }
        // framesets: cam1 drives; each other camera contributes its nearest frame within 1 ms
        let mut sets = Vec::with_capacity(indices[0].len());
        for &(t, off1) in &indices[0] {
            let mut offs = [off1, 0, 0, 0];
            let mut sum: i64 = t;
            let mut ok = true;
            for k in 1..CAMERAS {
                let idx = &indices[k];
                let j = idx.partition_point(|&(s, _)| s < t); // numpy searchsorted, side left
                let mut best: Option<(i64, usize, i64)> = None; // (|dt|, offset, stamp)
                for c in [j.wrapping_sub(1), j] {
                    if c < idx.len() {
                        let d = (idx[c].0 - t).abs();
                        if best.is_none_or(|b| d < b.0) {
                            best = Some((d, idx[c].1, idx[c].0));
                        }
                    }
                }
                match best {
                    Some((d, o, s)) if d <= SET_TOL_NS => {
                        offs[k] = o;
                        sum += s;
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                sets.push((sum.div_euclid(CAMERAS as i64), offs)); // int(mean) of positive stamps
            }
        }
        let imu = parse_imu(&dir.join("imu0.csv"))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            width,
            height,
            bytes_per_frame: bpf,
            raws,
            sets,
            n_cam1: indices[0].len(),
            imu,
        })
    }

    /// Width of the frames the VIO sees.
    pub fn out_width(&self) -> usize {
        self.width / SCALE
    }

    pub fn out_height(&self) -> usize {
        self.height / SCALE
    }

    /// The four frames of a set, halved as OpenCV's INTER_AREA halves them.
    pub fn frames(&self, offs: &[usize; CAMERAS]) -> Vec<Vec<u8>> {
        let (w, h) = (self.width, self.height);
        let (ow, oh) = (self.out_width(), self.out_height());
        offs.iter()
            .enumerate()
            .map(|(k, &o)| {
                let src = &self.raws[k][o..o + self.bytes_per_frame];
                let mut out = vec![0u8; ow * oh];
                for y in 0..oh {
                    let r0 = &src[(2 * y) * w..(2 * y + 1) * w];
                    let r1 = &src[(2 * y + 1) * w..(2 * y + 2) * w];
                    let row = &mut out[y * ow..(y + 1) * ow];
                    for x in 0..ow {
                        let s = r0[2 * x] as u32
                            + r0[2 * x + 1] as u32
                            + r1[2 * x] as u32
                            + r1[2 * x + 1] as u32;
                        row[x] = ((s + 2) >> 2) as u8;
                    }
                }
                let _ = h;
                out
            })
            .collect()
    }
}
