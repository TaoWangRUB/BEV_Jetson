# Scripts

Index of the helper scripts in this folder. "Runs on" = **TX2** (the board, usually inside the
`cuvslam-foxy:tx2` container) or **dev** (your workstation, needs python3 + opencv + numpy).

## Top level — build & run

| script | runs on | purpose |
|---|---|---|
| [setup_tx2_docker.sh](setup_tx2_docker.sh) | TX2 | one-time JetPack 4.6 / L4T R32.7 Docker prep for the BEV/cuVSLAM stack |
| [build_cuvslam_tx2gpu.sh](build_cuvslam_tx2gpu.sh) | TX2 | build cuVSLAM's GPU path on CUDA 10.2 with gcc-8 + C++14 (applies the port patch) |
| [run_vo_fused_tx2.sh](run_vo_fused_tx2.sh) | TX2 | wrapper: run the **fused** zero-copy Argus→cuVSLAM VO (`docker compose`) |
| [run_vo_tx2.sh](run_vo_tx2.sh) | TX2 | wrapper: run the **modular** capture + cuVSLAM VO (`docker compose`) |
| [capture_montage_tx2.sh](capture_montage_tx2.sh) | TX2 | capture the 4 fisheye views + stitched panorama, montage into one image |

## calib/ — camera calibration

| script | runs on | purpose |
|---|---|---|
| [calib/online_calib.py](calib/online_calib.py) | dev/TX2 | online interactive fisheye (Kannala-Brandt) **intrinsic** calibration |
| [calib/offline_calib.py](calib/offline_calib.py) | dev | offline outlier-rejection fisheye **intrinsic** calibration |
| [calib/scale_calib.py](calib/scale_calib.py) | dev | scale KB `camN.yaml` intrinsics to a downscaled output resolution |
| [calib/grid_view_tx2.sh](calib/grid_view_tx2.sh) | TX2 | live 2×3 IMX219 grid on the TX2 HDMI display (identify/label cameras) |
| [calib/capture_calib_sets.sh](calib/capture_calib_sets.sh) | TX2 | grab N raw 4-cam sets for **extrinsic** calibration (pan the rig) |
| [calib/extrinsic_calib.py](calib/extrinsic_calib.py) | dev | feature-based relative-rotation **extrinsic** calibration (joint solve + before/after render) |
| [calib/pano_tuner.py](calib/pano_tuner.py) | dev | interactive panorama extrinsics tuner (web UI; manual rot/trans) |

See [docs/extrinsic_calibration.md](../docs/extrinsic_calibration.md) for the full extrinsic-calibration procedure.

## port/ — cuVSLAM CUDA-10.2 port & measurement

| script | runs on | purpose |
|---|---|---|
| [port/build_and_validate.sh](port/build_and_validate.sh) | TX2 | one command: build the Foxy image, build the port, run the smoke test |
| [port/smoke_test.cpp](port/smoke_test.cpp) | TX2 | runtime smoke test for the CUDA-10.2 cuVSLAM port (WarmUpGPU) |
| [port/egl_cuda_spike.cpp](port/egl_cuda_spike.cpp) | TX2 | EGL→CUDA zero-copy bridge spike (validates the NVMM→CUDA device pointer) |
| [port/downgrade_cuvslam_cpp17.py](port/downgrade_cuvslam_cpp17.py) | TX2 | downgrade cuVSLAM C++17 device syntax to C++14 for nvcc 10.2 |
| [port/grab_views.py](port/grab_views.py) | TX2 | grab frames from ROS 2 image topics and/or montage 4 cams + panorama |
| [port/sync_check.py](port/sync_check.py) | TX2 | measure the timestamp spread across N camera topics |
| [port/topic_rate.py](port/topic_rate.py) | TX2 | count messages on topics over a window and print the rate |

## stream/ — quick live camera preview (no ROS/docker)

| script | runs on | purpose |
|---|---|---|
| [stream/csi_sender.sh](stream/csi_sender.sh) | TX2 | stream the 4 cameras as H.264/RTP/UDP to the host (`./csi_sender.sh [HOST_IP]`) |
| [stream/csi_receiver.sh](stream/csi_receiver.sh) | dev | receive the 4 UDP streams into a labelled 2×2 mosaic (`./csi_receiver.sh`) |

## rig/ — rig geometry

| script | runs on | purpose |
|---|---|---|
| [rig/gen_rig_extrinsics.py](rig/gen_rig_extrinsics.py) | dev | generate the cuVSLAM rig extrinsics (`rig_from_camera`/`imu`) from the physical layout |
