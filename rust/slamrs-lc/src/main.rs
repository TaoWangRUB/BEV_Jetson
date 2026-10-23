//! slamrs-lc: the slam-rs VIO with the loop-closing backend, as one native binary.
//!
//! `replay <log> <out>` mirrors scripts/slamrs/run_slamrs_lc.py: the same raw-log reader, the
//! same VIO call sequence and the same output files, so the Python comparison and Rerun tools
//! read its results unchanged and the Python outputs serve as the fixtures the port is held to.

mod backend;
mod config;
mod fmtg;
mod geom;
mod keyframes;
mod klt;
mod log;
mod pnp;
mod posegraph;
mod retrieval;
mod verify;
mod vio;

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};

use crate::fmtg::g9;

#[derive(Debug, Parser)]
#[command(
    name = "slamrs-lc",
    version,
    about = "slam-rs VIO with the BEV loop-closing backend"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the versions of this binary and of the slam-rs core it embeds.
    Version,
    /// Replay a raw log (log_rig.sh directory) through the VIO, with the backend beside it.
    Replay {
        /// The imglog directory: camN.raw, camN_index.csv, imu0.csv, geometry.txt.
        log: PathBuf,
        /// Output directory: slamrs_vio.csv and the loop-closure files.
        out: PathBuf,
        /// Stop after this many framesets (0 = all).
        #[arg(long, default_value_t = 0)]
        max_sets: usize,
        /// CPU frontend worker threads (the estimator is always one thread).
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// CubeCL/wgpu frontend lane (binary built with --features gpu-wgpu).
        #[arg(long)]
        gpu: bool,
        /// slam-rs calibration JSON (Basalt format).
        #[arg(long, default_value = "config/slamrs/bev_calib_s2.json")]
        calib: PathBuf,
        /// slam-rs VIO config JSON.
        #[arg(long, default_value = "config/slamrs/bev_vio_config.json")]
        config: PathBuf,
        /// Loop-closure parameters.
        #[arg(long, default_value = "config/slamrs/bev_lc_config.json")]
        lc_config: PathBuf,
        /// VIO only: write slamrs_vio.csv and stop.
        #[arg(long)]
        no_lc: bool,
    },
    /// Parity debugging: dump the prepared IMU samples and the halved frames of one set, so they
    /// can be compared with what run_slamrs.py feeds the same core.
    DebugDump {
        log: PathBuf,
        out: PathBuf,
        #[arg(long, default_value_t = 0)]
        set: usize,
    },
}

fn debug_dump(log_dir: &PathBuf, out: &PathBuf, set: usize) -> Result<(), String> {
    let log = log::RawLog::open(log_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut f = fs::File::create(out.join("imu_prepared.csv")).map_err(|e| e.to_string())?;
    for i in 0..log.imu.t.len() {
        let (g, a) = (log.imu.gyr[i], log.imu.acc[i]);
        writeln!(
            f,
            "{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            log.imu.t[i], g[0], g[1], g[2], a[0], a[1], a[2]
        )
        .map_err(|e| e.to_string())?;
    }
    let (t, offs) = log.sets[set];
    println!(
        "set {set}: stamp {t} offsets {offs:?} ({} sets)",
        log.sets.len()
    );
    for (k, frame) in log.frames(&offs).iter().enumerate() {
        fs::write(out.join(format!("cam{}.raw", k + 1)), frame).map_err(|e| e.to_string())?;
    }
    println!("frames {}x{} written", log.out_width(), log.out_height());
    Ok(())
}

const HDR: &str = "t_ns,status,tx,ty,tz,qx,qy,qz,qw,vx,vy,vz,bax,bay,baz,bgx,bgy,bgz,num_obs,num_landmarks,track_ms,fe_ms,est_ms";

fn replay(
    log_dir: &PathBuf,
    out: &PathBuf,
    max_sets: usize,
    threads: usize,
    gpu: bool,
    calib: &PathBuf,
    config: &PathBuf,
    lc_config: &PathBuf,
    no_lc: bool,
) -> Result<(), String> {
    let log = log::RawLog::open(log_dir).map_err(|e| e.to_string())?;
    println!(
        "{} complete sets of {} cam1 frames; IMU {} samples, |a|rest {:.3}",
        log.sets.len(),
        log.n_cam1,
        log.imu.t.len(),
        log.imu.a0_norm
    );
    let calib_text = fs::read_to_string(calib).map_err(|e| format!("{}: {e}", calib.display()))?;
    let config_text =
        fs::read_to_string(config).map_err(|e| format!("{}: {e}", config.display()))?;
    let lc = config::LcConfig::from_json(
        &fs::read_to_string(lc_config).map_err(|e| format!("{}: {e}", lc_config.display()))?,
    )
    .map_err(|e| format!("{}: {e}", lc_config.display()))?;
    let calibration: slam_rs::calib::Calibration<f64> =
        slam_rs::calib::Calibration::from_json_str(&calib_text).map_err(|e| e.to_string())?;
    let rig = geom::Rig::new(&calibration, lc.features.theta_max_deg)?;
    let mut vio = vio::Vio::new(&calib_text, &config_text, threads, gpu)?;
    println!(
        "slam-rs lane: {}  threads={threads}",
        if vio.gpu() { "gpu" } else { "cpu" }
    );
    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let n_sets = if max_sets > 0 {
        max_sets.min(log.sets.len())
    } else {
        log.sets.len()
    };
    let (w, h) = (log.out_width(), log.out_height());
    let mut tracker = keyframes::KeyframeTracker::new(&lc.features);
    let mut backend = if no_lc {
        None
    } else {
        Some(backend::LoopClosureBackend::new(lc.clone(), w, h))
    };
    let mut frames_out: Vec<(i64, geom::Pose, i64)> = Vec::with_capacity(n_sets);
    let mut n_kf = 0usize;
    let t0 = Instant::now();
    // wall time per stage of the loop, to see what surrounds track()
    let mut st = [0.0f64; 6]; // frames, imu, track, snapshot, tracker, backend
    let mut rows: Vec<String> = Vec::with_capacity(n_sets);
    let mut fe_rows: Vec<String> = Vec::with_capacity(n_sets);
    let mut ii = 0usize;
    let imu = &log.imu;
    // the frames are read and halved on their own thread, a few sets ahead: on the rig they come
    // from the cameras, so their cost is replay overhead, not VIO time
    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Vec<Vec<u8>>)>(4);
    let reader_dir = log_dir.clone();
    let reader = std::thread::spawn(move || -> Result<(), String> {
        let log2 = log::RawLog::open(&reader_dir).map_err(|e| e.to_string())?;
        for n in 0..n_sets {
            let (_, offs) = log2.sets[n];
            if tx.send((n, log2.frames(&offs))).is_err() {
                break;
            }
        }
        Ok(())
    });
    for n in 0..n_sets {
        let (t, offs) = log.sets[n];
        let _ = offs;
        // IMU samples up to 20 ms past the frame, as run_slamrs.py pushes them
        let ts = Instant::now();
        let j = imu.t.partition_point(|&x| x < t + 20_000_000);
        if j > ii {
            vio.push_imu_batch(&imu.t[ii..j], &imu.gyr[ii..j], &imu.acc[ii..j])
                .map_err(|e| e.to_string())?;
            ii = j;
        }
        st[1] += ts.elapsed().as_secs_f64();
        let ts = Instant::now();
        let (nn, frames) = rx.recv().map_err(|_| "frame reader stopped early".to_string())?;
        debug_assert_eq!(nn, n);
        st[0] += ts.elapsed().as_secs_f64();
        let seam0 = vio::Vio::seam();
        let tc = Instant::now();
        let r = vio.track(t, &frames, w, h).map_err(|e| e.to_string())?;
        let dt_ms = tc.elapsed().as_secs_f64() * 1e3;
        let seam1 = vio::Vio::seam();
        {
            let b = vio.frontend_breakdown_ms();
            let mut row = format!("{t},{dt_ms:.3},{:.3},{:.3},{:.3},{:.3},{:.3}", b[0], b[1], b[2], b[3], b[4]);
            for k in 0..4 {
                row.push_str(&format!(",{},{:.3}", seam1[k].0 - seam0[k].0, seam1[k].1 - seam0[k].1));
            }
            fe_rows.push(row);
        }
        st[2] += dt_ms * 1e-3;
        let ts = Instant::now();
        let snap = vio.snapshot();
        st[3] += ts.elapsed().as_secs_f64();
        let (nobs, nlm, est_ms) = snap.as_ref().map_or((0, 0, 0.0), |s| {
            (s.num_observations, s.landmark_ids.len(), s.est_ms)
        });
        let fe_ms = vio.frontend_ms();
        let mut row = format!("{},{}", t, vio::status_int(r.status));
        for v in r
            .world_from_rig
            .iter()
            .chain(r.velocity.iter())
            .chain(r.accel_bias.iter())
            .chain(r.gyro_bias.iter())
        {
            row.push(',');
            row.push_str(&g9(*v));
        }
        for v in [nobs as f64, nlm as f64, dt_ms, fe_ms, est_ms] {
            row.push(',');
            row.push_str(&g9(v));
        }
        rows.push(row);
        frames_out.push((
            t,
            geom::pose7_to_iso(&r.world_from_rig),
            vio::status_int(r.status),
        ));
        if let (Some(snap), Some(b)) = (snap.as_ref(), backend.as_mut()) {
            let ts = Instant::now();
            let flow = vio.flow_frame();
            let new_kf = tracker.update(t, &frames, snap, flow.as_ref(), &rig);
            st[4] += ts.elapsed().as_secs_f64();
            let ts = Instant::now();
            for idx in new_kf {
                n_kf += 1;
                if let Some(hy) = b.on_keyframe(&mut tracker.done, idx, &rig) {
                    println!(
                        "  LOOP kf {} -> {} ({}) inliers {}/{} ratio {:.2} best pair {}-{}",
                        hy.cur,
                        hy.cand,
                        hy.route,
                        hy.inliers2,
                        hy.tracked,
                        hy.ratio,
                        hy.best_pair.0 + 1,
                        hy.best_pair.1 + 1
                    );
                }
            }
            st[5] += ts.elapsed().as_secs_f64();
        }
        if n % 100 == 0 {
            let p = r.world_from_rig;
            println!(
                "{n:5} t={:6.2}s status={} pos=[{:.3} {:.3} {:.3}] lm={nlm} obs={nobs} kf={n_kf} track={dt_ms:.1}ms",
                (t - log.sets[0].0) as f64 / 1e9,
                vio::status_int(r.status),
                p[0],
                p[1],
                p[2]
            );
        }
    }
    drop(rx);
    reader.join().map_err(|_| "frame reader panicked".to_string())??;
    if let Some(b) = backend.as_mut() {
        let ts = Instant::now();
        for idx in tracker.flush() {
            n_kf += 1;
            b.on_keyframe(&mut tracker.done, idx, &rig);
        }
        st[5] += ts.elapsed().as_secs_f64();
    }
    println!(
        "stages (s): frames (wait) {:.1}  imu {:.1}  track {:.1}  snapshot {:.1}  keyframe tracker {:.1}  backend {:.1}  total {:.1}",
        st[0], st[1], st[2], st[3], st[4], st[5], t0.elapsed().as_secs_f64()
    );
    let path = out.join("slamrs_vio.csv");
    let mut f = fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    writeln!(f, "{HDR}").map_err(|e| e.to_string())?;
    for row in &rows {
        writeln!(f, "{row}").map_err(|e| e.to_string())?;
    }
    {
        let path = out.join("frontend_breakdown.csv");
        let mut f = fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        writeln!(f, "t_ns,track_ms,pyramid_ms,detect_ms,klt_ms,stereo_ms,imu_ms,launches,launch_ms,uploads,upload_ms,reads_track,read_track_ms,reads_detect,read_detect_ms").map_err(|e| e.to_string())?;
        for row in &fe_rows {
            writeln!(f, "{row}").map_err(|e| e.to_string())?;
        }
    }
    println!(
        "wrote {}: {} poses in {:.0} s; {n_kf} keyframes",
        path.display(),
        rows.len(),
        t0.elapsed().as_secs_f64()
    );
    let Some(mut b) = backend else {
        return Ok(());
    };
    b.finish();
    let s = b.write(out, &tracker.done, &frames_out)?;
    println!(
        "loop closure: {} keyframes, {} hypotheses, {} accepted, {} dropped; feat {:.1} ms, verify {:.1} ms median",
        s.keyframes,
        s.hypotheses,
        s.accepted,
        s.dropped.len(),
        s.feat_ms_median,
        s.verify_ms_median
    );
    Ok(())
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Command::Version => {
            println!(
                "slamrs-lc {} (slam-rs core {})",
                env!("CARGO_PKG_VERSION"),
                slam_rs::VERSION
            );
            ExitCode::SUCCESS
        }
        Command::Replay {
            log,
            out,
            max_sets,
            threads,
            gpu,
            calib,
            config,
            lc_config,
            no_lc,
        } => {
            match replay(
                &log, &out, max_sets, threads, gpu, &calib, &config, &lc_config, no_lc,
            ) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::DebugDump { log, out, set } => match debug_dump(&log, &out, set) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
    }
}
