//! config/slamrs/bev_lc_config.json, the loop-closure parameters (openspec design D1–D6, D2c).
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // the reference's keys are all accepted; the appearance route is not ported (D12)
pub struct Features {
    #[serde(default)]
    pub orb_enabled: bool,
    pub min_landmark_age: i64,
    pub min_bearing_z: f64,
    pub theta_max_deg: f64,
    pub min_obs: i64,
    pub min_parallax_deg: f64,
    pub far_parallax_deg: f64,
    pub refresh_enabled: bool,
    pub refresh_max_px: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Retrieval {
    pub min_separation_s: f64,
    pub spatial_radius_m: f64,
    #[serde(default)]
    pub min_path_m: f64,
    pub max_candidates: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Verify {
    pub min_pair_matches: usize,
    pub neighbour_radius_m: f64,
    pub neighbour_max_kf: i64,
    pub klt_win: usize,
    pub klt_levels: usize,
    pub klt_patch: usize,
    pub klt_ncc_min: f64,
    pub klt_max_err: f64,
    pub klt_fb_max_px: f64,
    pub ransac_thresh_norm: f64,
    pub ransac_iters: usize,
    pub min_landmarks: usize,
    pub ratio_lo_at: f64,
    pub ratio_hi_at: f64,
    pub ratio_lo: f64,
    pub ratio_hi: f64,
    pub consistency_tol_m: f64,
    pub consistency_tol_deg: f64,
    pub base_sigma_t_m: f64,
    pub base_sigma_yaw_deg: f64,
    pub cov_scale_max: f64,
    pub seed: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // max_nodes: the node cap is open in the reference as well (D12)
pub struct Graph {
    pub seq_sigma_t0_m: f64,
    pub seq_sigma_t_per_m: f64,
    pub seq_sigma_yaw0_deg: f64,
    pub seq_sigma_yaw_per_m_deg: f64,
    pub huber_m: f64,
    pub edge_check_sigma: f64,
    #[serde(default)]
    pub max_nodes: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Backend {
    pub throttle_s: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LcConfig {
    pub features: Features,
    pub retrieval: Retrieval,
    pub verify: Verify,
    pub graph: Graph,
    pub backend: Backend,
}

impl LcConfig {
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}
