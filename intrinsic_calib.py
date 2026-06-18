import argparse
try:
    import cv2
except Exception:
    print("OpenCV Python module 'cv2' is not installed.\n"
          "On Jetson TX2 install with: sudo apt-get update && sudo apt-get install -y python3-opencv")
    raise
import numpy as np
import os
import sys
import glob

# ======================
# SETTINGS
# ======================
CHECKERBOARD = (10, 8)  # Internal corners (WxH)
SQUARE_SIZE_MM = 30
PORT = 5005

# Command-line arguments (defaults above can be overridden)
parser = argparse.ArgumentParser(description="Calibrate IMX219 cameras (Jetson TX2 / J106)")
parser.add_argument("--mode", choices=["udp", "nvargus", "v4l2"], default="nvargus",
                    help="Capture mode: 'nvargus' for local Jetson cameras, 'v4l2' for /dev/video nodes, 'udp' for RTP H264 stream")
parser.add_argument("--port", type=int, default=PORT, help="UDP port for RTP H264 (udp mode)")
parser.add_argument("--sensor-id", type=int, default=0, help="Argus sensor-id (0..5) for nvarguscamerasrc")
parser.add_argument("--device", default="/dev/video0", help="v4l2 device path for v4l2 mode")
parser.add_argument("--width", type=int, default=1920, help="Capture width for nvargus/v4l2")
parser.add_argument("--height", type=int, default=1080, help="Capture height for nvargus/v4l2")
parser.add_argument("--fps", type=int, default=20, help="Frame rate for nvargus")
parser.add_argument("--flip", type=int, default=0, help="nvvidconv flip-method (0..7)")
parser.add_argument("--checkerboard", type=str, default=f"{CHECKERBOARD[0]}x{CHECKERBOARD[1]}",
                    help="Checkerboard internal corners WxH, e.g. 10x8")
parser.add_argument("--square", type=float, default=SQUARE_SIZE_MM, help="Checkerboard square size in mm")
args = parser.parse_args()

# override defaults with parsed values
parts = args.checkerboard.lower().split('x')
try:
    CHECKERBOARD = (int(parts[0]), int(parts[1]))
except Exception:
    print("Invalid --checkerboard format, expected WxH like 10x8")
    sys.exit(1)
SQUARE_SIZE_MM = float(args.square)
PORT = int(args.port)
# Normalize flip argument: accept degrees (0,90,180,270) or nvvidconv codes 0..7
if args.flip in (0, 90, 180, 270):
    _flip_map = {0: 0, 90: 1, 180: 2, 270: 3}
    args.flip = _flip_map[args.flip]
else:
    if not (0 <= args.flip <= 7):
        print("Invalid --flip value. Use 0..7 (nvvidconv flip-method) or degrees 0/90/180/270.")
        sys.exit(1)

# Normalize flip: allow degrees (0,90,180,270) or nvvidconv flip-method (0..7)
try:
    raw_flip = int(args.flip)
except Exception:
    print("Invalid --flip value, must be integer (0..7) or degrees 0/90/180/270")
    sys.exit(1)
deg_map = {0: 0, 90: 1, 180: 2, 270: 3}
if raw_flip in deg_map:
    flip_method = deg_map[raw_flip]
elif 0 <= raw_flip <= 7:
    flip_method = raw_flip
else:
    print("Invalid --flip value, must be 0..7 or one of [0,90,180,270]")
    sys.exit(1)
args.flip = flip_method

# If running headless over SSH, try to target the local display :0 so imshow shows
# on the TX2 screen. Also attempt to find a valid XAUTHORITY file under /home.
if 'DISPLAY' not in os.environ or not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'
if 'XAUTHORITY' not in os.environ or not os.environ.get('XAUTHORITY'):
    auth_candidates = glob.glob('/home/*/.Xauthority')
    if auth_candidates:
        os.environ['XAUTHORITY'] = auth_candidates[0]

# ======================
# GStreamer PIPELINE
# ======================
PIPE_TX2 = (
    f"udpsrc port={PORT} buffer-size=8388608 "
    "caps=application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
    "rtpjitterbuffer latency=5 ! "
    "rtph264depay ! h264parse ! "
    "nvv4l2decoder enable-max-performance=1 ! "
    "nvvidconv flip-method=2 ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true sync=false max-buffers=1"
)

# ======================
# DATA STORAGE
# ======================
obj_points = []  # 3d points in real world space
img_points = []  # 2d points in image plane

# Prepare object points (0,0,0), (square,0,0) ... in mm
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

# ======================
# INITIALIZATION
# ======================
# Build capture source depending on selected mode
if args.mode == 'udp':
    gst = PIPE_TX2.replace(f"port={PORT}", f"port={args.port}")
    capture_source = gst
    cap = cv2.VideoCapture(capture_source, cv2.CAP_GSTREAMER)
elif args.mode == 'nvargus':
    capture_source = (
        f"nvarguscamerasrc sensor-id={args.sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={args.width},height={args.height},format=NV12,framerate={args.fps}/1 ! "
        f"nvvidconv flip-method={args.flip} ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false max-buffers=1"
    )
    cap = cv2.VideoCapture(capture_source, cv2.CAP_GSTREAMER)
else:
    # v4l2 device path
    capture_source = args.device
    cap = cv2.VideoCapture(capture_source)

if not cap.isOpened():
    print(f"❌ Error: Could not open capture source: {capture_source}")
    sys.exit(1)

print("--- Camera Calibration ---")
print("SPACE: Capture Frame | C: Calibrate & Preview | S: Save & Exit | Q: Quit")
print(f"Capture source: {args.mode} -> {capture_source}")

calibrated = False
K = np.zeros((3, 3))
D = np.zeros((4, 1))
map1, map2 = None, None

warning_shown = False
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            if not warning_shown:
                print("WARNING: Video is not available yet!")
                warning_shown = True
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        display = frame.copy()
        if ret_corners:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret_corners)
            cv2.putText(display, "READY TO CAPTURE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        title = f"Calibration ({args.mode})"
        cv2.imshow(title, display)
        key = cv2.waitKey(1)

        if key == ord(' ') and ret_corners:
            img_points.append(corners)
            obj_points.append(objp)
            print(f"Captured {len(img_points)} samples...")

        elif key == ord('c'):
            if len(img_points) < 10:
                print("Need at least 10 images!")
            else:
                print("Calculating... please wait...")
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
                print(f"FINAL REPROJECTION ERROR: {ret:.4f}")
                if ret < 0.5:
                    # Choose sensible filename based on capture mode
                    if args.mode == 'nvargus':
                        fname = f"calib_nvargus_cam{args.sensor_id}_{args.width}x{args.height}.npz"
                    elif args.mode == 'v4l2':
                        dev = os.path.basename(args.device)
                        fname = f"calib_v4l2_{dev}_{args.width}x{args.height}.npz"
                    else:
                        fname = f"calib_port_{args.port}.npz"
                    np.savez(fname, mtx=mtx, dist=dist)
                    print(f"SUCCESS! Data saved to {fname}")
                else:
                    print("ERROR TOO HIGH. Try capturing more angles, especially edges.")

        elif key == ord('q'):
            break
except KeyboardInterrupt:
    print("\nStopped by user.")

cap.release()
cv2.destroyAllWindows()
