# Per-frame timing metadata for one camera. Published alongside the image so that a
# recording carries what the image alone cannot: which trigger edge a frame came from,
# and whether any were lost. See README 4.7 for the timestamp convention.

# stamp = this frame's EXPOSURE MIDPOINT on CLOCK_MONOTONIC (not ROS system time);
# frame_id = the camera name (cam1..cam4).
std_msgs/Header header

# Consumer-side frame count (EGLStream::IFrame::getNumber). Counts what reached us.
uint64 frame_number

# Argus session-side capture id (ICaptureMetadata::getCaptureId). Counts what the
# session produced - so frame_number advancing slower than capture_id localises a
# loss to the delivery path rather than to the sensor or the trigger.
uint32 capture_id

# Kernel SOF: the time the first data of this capture left the sensor, i.e. the start
# of READOUT, which on a global shutter is after the exposure ended. CLOCK_MONOTONIC.
uint64 sof_ns

# False when the frame was acquired and timed but its pixels never made it out — the
# NVMM map or copy failed. The timing record still stands (the exposure happened), so
# this says "expect no image at this stamp" rather than leaving an unmatched record.
bool image_published

# The exposure used to derive the midpoint. One value for the whole rig: all cameras
# fire on the same trigger edge for the same pulse width.
uint64 exposure_ns
