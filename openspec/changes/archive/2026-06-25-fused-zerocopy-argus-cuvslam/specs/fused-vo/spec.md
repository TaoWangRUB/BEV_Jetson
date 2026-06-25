## ADDED Requirements

### Requirement: Single-process Argus → cuVSLAM with no CPU frame copy

The fused node SHALL capture from the 4 Argus cameras and run cuVSLAM `Track()` in one
process, feeding each frame's NVMM luma plane to cuVSLAM as GPU memory
(`cuvslam::Image.is_gpu_mem = true`) via the EGL→CUDA bridge, without copying pixels to host
memory or serializing them over DDS.

#### Scenario: Frame reaches cuVSLAM as a GPU pointer

- **WHEN** an Argus frame is acquired for camera i
- **THEN** the node obtains a CUDA device pointer to that frame's Y plane (via NVMM→EGLImage→CUDA)
- **AND** passes it to `Track()` with `is_gpu_mem = true` and the correct device-buffer `pitch`
- **AND** no `NvBufferMemMap`/CPU copy and no image publish occurs on the tracking path

#### Scenario: No image topics are published

- **WHEN** the fused node runs
- **THEN** it publishes `/cuvslam/odometry` and the `odom→base_link` TF only
- **AND** it does NOT publish `/camN/image_raw` (frames never leave the process)

### Requirement: Fused odometry matches the modular pipeline

The fused node SHALL produce odometry equivalent to the modular capture→VO pipeline (same
calibration, rig, bundler/staleness logic, unified per-set timestamp), at equal or better rate.

#### Scenario: Tracking parity

- **WHEN** the fused node runs with the same `scripts/config/calib` + `config/rig` inputs
- **THEN** `/cuvslam/odometry` publishes with live tracking (no "tracking lost") at ≥ the modular node's ~8.5 Hz
- **AND** the pose holds near origin when stationary and tracks under rig motion

#### Scenario: Lower CPU than the modular pipeline

- **WHEN** the fused node and the modular two-node pipeline are each run at the same camera rate
- **THEN** the fused node uses measurably less CPU (no per-frame NvBufferMemMap memcpy + DDS image serialization)

### Requirement: GPU resources are managed safely

The fused node SHALL manage the EGL/CUDA interop resources without leaks or use-after-free
across the streaming lifetime and on shutdown.

#### Scenario: Sustained streaming without leak or fault

- **WHEN** the fused node runs continuously for several minutes
- **THEN** registered EGL images / CUDA graphics resources are unregistered/unmapped per frame (or correctly cached) with no growth in GPU memory and no CUDA faults
- **AND** on Ctrl-C the Argus streams, EGL displays, and CUDA resources are released cleanly
