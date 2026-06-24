## ADDED Requirements

### Requirement: GPU equirectangular stitch of the 4 fisheye

The node SHALL stitch the 4 fisheye cameras into one equirectangular panorama on the GPU, using
a remap table precomputed from the KB intrinsics + rig extrinsics (no per-frame CPU pixel work
on the stitch path). Frames SHALL be obtained as CUDA device pointers (NVMM→CUDA bridge), and a
custom CUDA kernel SHALL bilinear-sample and weight-blend the in-FOV cameras per output pixel.

#### Scenario: Panorama is produced on the GPU

- **WHEN** the node runs with the 4 calibrated cameras
- **THEN** it computes per-camera equirect remap tables at startup (output azimuth×elevation)
- **AND** each frame is stitched by a CUDA kernel reading the cameras' device pointers (no host pixel copy on the stitch path)

#### Scenario: Overlaps are blended

- **WHEN** an output pixel is covered by more than one camera (the ~70° fisheye overlap)
- **THEN** the kernel weight-blends the contributing cameras (feather by angular distance from each FOV centre), not a hard cut

### Requirement: Published for rviz, optional video

The node SHALL publish the panorama as `sensor_msgs/Image` (mono8) on `/bev/panorama` viewable in
rviz, and SHALL optionally write it to a video file when enabled by a parameter.

#### Scenario: Topic visualizable in rviz

- **WHEN** the node is running
- **THEN** `/bev/panorama` publishes mono8 `sensor_msgs/Image` frames that display in rviz's Image view

#### Scenario: Optional video recording

- **WHEN** the `save_video` parameter is set
- **THEN** the node writes the panorama stream to the configured video file and finalizes it cleanly on shutdown

### Requirement: Configurable output and runs in the existing container

The panorama dimensions / elevation range / calibration path SHALL be parameters, and the node
SHALL build and run in the existing `cuvslam-foxy:tx2` image (CUDA `.cu` compiled for sm_62 with
host g++-8) with no new image.

#### Scenario: Configurable canvas

- **WHEN** the output width/height and elevation range are set via params
- **THEN** the panorama is produced at that resolution/coverage and the remap tables are sized accordingly
