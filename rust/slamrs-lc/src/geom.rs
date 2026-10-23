//! Poses, the rig and its cameras. Poses are `Isometry3<f64>`; a 7-vector is
//! `[tx, ty, tz, qx, qy, qz, qw]` as slam-rs reports it. Projection and unprojection are
//! slam-rs's own KB4 (Basalt's `atan2` form, which holds past 90° of incidence), with the
//! validity the Python backend used: inside the modelled field of view and inside the image.
use nalgebra::{Isometry3, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3, Vector4};
use slam_rs::calib::Calibration;
use slam_rs::camera::RigCamera;

pub type Pose = Isometry3<f64>;

/// Transform a point: nalgebra's `Isometry * Vector` applies the rotation only.
#[inline]
pub fn tf(t: &Pose, p: &Vector3<f64>) -> Vector3<f64> {
    t.rotation * p + t.translation.vector
}

pub fn pose7_to_iso(p: &[f64; 7]) -> Pose {
    let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(p[6], p[3], p[4], p[5]));
    Isometry3::from_parts(Translation3::new(p[0], p[1], p[2]), q)
}

pub fn iso_to_pose7(t: &Pose) -> [f64; 7] {
    let q = t.rotation.quaternion();
    [
        t.translation.x,
        t.translation.y,
        t.translation.z,
        q.i,
        q.j,
        q.k,
        q.w,
    ]
}

/// Heading about world +Z, the VIO's gravity axis.
pub fn yaw_of(r: &Rotation3<f64>) -> f64 {
    r[(1, 0)].atan2(r[(0, 0)])
}

pub fn rz(yaw: f64) -> Rotation3<f64> {
    Rotation3::from_axis_angle(&Vector3::z_axis(), yaw)
}

pub fn wrap_angle(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

pub fn rot_angle_deg(r: &Rotation3<f64>) -> f64 {
    r.angle().to_degrees()
}

pub fn from_rotvec_t(rv: &Vector3<f64>, t: &Vector3<f64>) -> Pose {
    Isometry3::from_parts(
        Translation3::from(*t),
        UnitQuaternion::from_scaled_axis(*rv),
    )
}

#[allow(dead_code)]
pub struct Rig {
    pub cams: Vec<RigCamera<f64>>,
    /// Camera pose in the rig (IMU) frame.
    pub t_imu_cam: Vec<Pose>,
    pub t_cam_imu: Vec<Pose>,
    pub theta_max: f64,
}

#[allow(dead_code)]
impl Rig {
    pub fn new(calibration: &Calibration<f64>, theta_max_deg: f64) -> Result<Self, String> {
        let cams = RigCamera::from_calibration(calibration).map_err(|e| e.to_string())?;
        let mut t_imu_cam = Vec::with_capacity(cams.len());
        for se3 in &calibration.t_i_c {
            let q: UnitQuaternion<f64> = *se3.rotation.quaternion();
            t_imu_cam.push(Isometry3::from_parts(
                Translation3::from(se3.translation),
                q,
            ));
        }
        let t_cam_imu = t_imu_cam.iter().map(|t| t.inverse()).collect();
        Ok(Self {
            cams,
            t_imu_cam,
            t_cam_imu,
            theta_max: theta_max_deg.to_radians(),
        })
    }

    pub fn n(&self) -> usize {
        self.cams.len()
    }

    pub fn width(&self, c: usize) -> f64 {
        f64::from(self.cams[c].width())
    }

    pub fn height(&self, c: usize) -> f64 {
        f64::from(self.cams[c].height())
    }

    /// Camera-frame point to a pixel, or None outside the modelled field of view or the image.
    pub fn project(&self, c: usize, p: &Vector3<f64>) -> Option<Vector2<f64>> {
        let r = p.xy().norm();
        let theta = r.atan2(p.z);
        if theta >= self.theta_max {
            return None;
        }
        let mut uv = Vector2::zeros();
        if !self.cams[c]
            .model
            .project(&Vector4::new(p.x, p.y, p.z, 1.0), &mut uv)
        {
            return None;
        }
        let (w, h) = (self.width(c), self.height(c));
        if uv.x >= 0.0 && uv.x < w - 1.0 && uv.y >= 0.0 && uv.y < h - 1.0 {
            Some(uv)
        } else {
            None
        }
    }

    /// Pixel to a unit bearing in the camera frame.
    pub fn unproject(&self, c: usize, uv: &Vector2<f64>) -> Vector3<f64> {
        let mut p = Vector4::zeros();
        self.cams[c].model.unproject(uv, &mut p);
        let b = Vector3::new(p.x, p.y, p.z);
        let n = b.norm();
        if n > 1e-12 { b / n } else { Vector3::z() }
    }
}

/// Normalised-plane coordinates of a bearing with z above `min_z`.
pub fn normalised(b: &Vector3<f64>, min_z: f64) -> Option<Vector2<f64>> {
    if b.z > min_z {
        Some(Vector2::new(b.x / b.z, b.y / b.z))
    } else {
        None
    }
}
