//! The slam-rs VIO behind the same calls run_slamrs.py makes through the PyO3 core, and the two
//! read-outs the backend lives on: the window snapshot and the frontend's keypoints.
use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::frontend::flow::FrontendOptions;
use slam_rs::{Backend, ImageView, VioError, VioResult, VioStatus};

/// The estimator's window, landmarks and last-frame statistics, as the Python `VioSnapshot`
/// flattens them: the 15-dof states then the pose-only blocks, each oldest first.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Snapshot {
    pub t_ns: i64,
    pub window_t_ns: Vec<i64>,
    /// `[tx, ty, tz, qx, qy, qz, qw]` per window frame.
    pub window_poses: Vec<[f64; 7]>,
    pub window_keyframe: Vec<bool>,
    pub kf_ids: Vec<i64>,
    pub marginalized: Vec<i64>,
    pub landmark_ids: Vec<i64>,
    pub landmark_hosts: Vec<i64>,
    pub landmark_positions: Vec<[f64; 3]>,
    pub num_observations: usize,
    /// The estimator's `measure` phase, milliseconds (the Python csv's `est_ms`).
    pub est_ms: f64,
}

/// The frontend's keypoints on the last accepted frameset: ids and pixel positions per camera.
#[derive(Debug, Clone)]
pub struct FlowObs {
    pub t_ns: i64,
    pub cams: Vec<(Vec<i64>, Vec<[f64; 2]>)>,
}

pub struct Vio {
    inner: slam_rs::Vio<f32>,
}

#[allow(dead_code)]
impl Vio {
    pub fn new(
        calib_json: &str,
        config_json: &str,
        threads: usize,
        gpu: bool,
    ) -> Result<Self, String> {
        let calibration: Calibration<f64> =
            Calibration::from_json_str(calib_json).map_err(|e| e.to_string())?;
        let config: VioConfig = VioConfig::from_json_str(config_json).map_err(|e| e.to_string())?;
        let defaults = FrontendOptions::default();
        let options = FrontendOptions {
            threads,
            max_keypoints: defaults.max_keypoints,
        };
        let backend = if gpu { Backend::Gpu } else { Backend::Cpu };
        let inner = slam_rs::Vio::with_backend(config, calibration, options, backend)
            .map_err(|e| e.to_string())?;
        Ok(Self { inner })
    }

    pub fn gpu(&self) -> bool {
        self.inner.backend() == Backend::Gpu
    }

    pub fn camera_count(&self) -> usize {
        self.inner.frontend().camera_count()
    }

    /// All or nothing, as the Python boundary: the batch is checked before any sample is pushed.
    pub fn push_imu_batch(
        &mut self,
        t: &[i64],
        gyr: &[[f64; 3]],
        acc: &[[f64; 3]],
    ) -> Result<(), VioError> {
        let mut previous = self.inner.last_imu_t_ns();
        for i in 0..t.len() {
            slam_rs::check_imu_sample(t[i], &gyr[i], &acc[i], previous)?;
            previous = Some(t[i]);
        }
        for i in 0..t.len() {
            self.inner.push_imu(t[i], gyr[i], acc[i])?;
        }
        Ok(())
    }

    pub fn track(
        &mut self,
        t_ns: i64,
        frames: &[Vec<u8>],
        width: usize,
        height: usize,
    ) -> Result<VioResult, VioError> {
        let views: Vec<ImageView<'_>> = frames
            .iter()
            .map(|f| ImageView {
                width,
                height,
                stride: width,
                data: f,
            })
            .collect();
        self.inner.track(t_ns, &views)
    }

    /// Milliseconds the frontend spent on the last frameset (the Python csv's `fe_ms`).
    pub fn frontend_ms(&self) -> f64 {
        let t = self.inner.frontend_timings();
        (t.pyramid_ns + t.detect_ns + t.track_ns + t.stereo_ns + t.imu_ns) as f64 / 1e6
    }

    pub fn snapshot(&self) -> Option<Snapshot> {
        let stats = self.inner.last_stats()?;
        let w = self.inner.estimator().snapshot();
        let n = w.states.len() + w.poses.len();
        let mut s = Snapshot {
            t_ns: w.t_ns,
            window_t_ns: Vec::with_capacity(n),
            window_poses: Vec::with_capacity(n),
            window_keyframe: Vec::with_capacity(n),
            kf_ids: stats.kf_ids.clone(),
            marginalized: w.marginalized.clone(),
            landmark_ids: Vec::with_capacity(w.landmarks.len()),
            landmark_hosts: Vec::with_capacity(w.landmarks.len()),
            landmark_positions: Vec::with_capacity(w.landmarks.len()),
            num_observations: stats.num_observations,
            est_ms: stats.timings.measure_ns as f64 / 1e6,
        };
        for st in w.states.iter().chain(w.poses.iter()) {
            let q = st.t_w_i.rotation.quaternion_xyzw();
            let t = st.t_w_i.translation;
            s.window_t_ns.push(st.t_ns);
            s.window_poses.push([
                f64::from(t.x),
                f64::from(t.y),
                f64::from(t.z),
                f64::from(q[0]),
                f64::from(q[1]),
                f64::from(q[2]),
                f64::from(q[3]),
            ]);
            s.window_keyframe.push(st.keyframe);
        }
        for lm in &w.landmarks {
            s.landmark_ids.push(lm.id.0 as i64);
            s.landmark_hosts.push(lm.host.frame_id);
            s.landmark_positions.push([
                f64::from(lm.position_w.x),
                f64::from(lm.position_w.y),
                f64::from(lm.position_w.z),
            ]);
        }
        Some(s)
    }

    pub fn flow_frame(&self) -> Option<FlowObs> {
        let t_ns = self.inner.frontend().t_ns()?;
        let frame = self.inner.frontend().frame();
        let cams = frame
            .cameras
            .iter()
            .map(|kps| {
                let ids: Vec<i64> = kps.ids.iter().map(|id| id.0 as i64).collect();
                let pos: Vec<[f64; 2]> = (0..kps.len())
                    .map(|slot| {
                        let w = kps.transforms.coefficients(slot); // [m00, m01, m10, m11, tx, ty]
                        [f64::from(w[4]), f64::from(w[5])]
                    })
                    .collect();
                (ids, pos)
            })
            .collect();
        Some(FlowObs { t_ns, cams })
    }
}

/// `int(r.status)` of the Python boundary.
pub fn status_int(s: VioStatus) -> i64 {
    match s {
        VioStatus::NeedMoreImu => 0,
        VioStatus::Tracking => 1,
    }
}
