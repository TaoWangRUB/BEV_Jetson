import cv2
import numpy as np
import os

# ======================
# SETTINGS
# ======================
CHECKERBOARD = (10, 8)  # Internal corners
SQUARE_SIZE_MM = 30
PORT = 5005

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

# Prepare object points (0,0,0), (30,0,0) ...
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

# ======================
# INITIALIZATION
# ======================
cap = cv2.VideoCapture(PIPE_TX2, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ Error: Could not open GStreamer stream.")
    exit()

print("--- Fisheye Calibration ---")
print("SPACE: Capture Frame | C: Calibrate & Preview | S: Save & Exit | Q: Quit")

calibrated = False
K = np.zeros((3, 3))
D = np.zeros((4, 1))
map1, map2 = None, None

while True:
    ret, frame = cap.read()
    if not ret: continue

    h, w = frame.shape[:2]
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not calibrated:
        # Detect corners
        # Using SB (Sector Based) + EXHAUSTIVE for wide angle lenses
        found, corners = cv2.findChessboardCornersSB(
            gray, CHECKERBOARD, 
            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )

        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)
            cv2.putText(display, f"Captured: {len(img_points)}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # LIVE PREVIEW OF UNDISTORTED VIEW
        display = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.putText(display, "PREVIEW MODE (Check if lines are straight)", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Fisheye Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    # CAPTURE FRAME
    if key == ord(' ') and found and not calibrated:
        # Fisheye calibration requires points in shape (1, N, 2)
        img_points.append(corners.reshape(1, -1, 2))
        obj_points.append(objp)
        print(f"✅ Captured frame {len(img_points)}")

    # RUN CALIBRATION
    elif key == ord('c'):
        if len(img_points) < 20:
            print(f"⚠ Not enough data. Need ~20, have {len(img_points)}")
            continue
            
        print("⏳ Calculating Fisheye Parameters...")
        N_OK = len(obj_points)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        # Fisheye specific calibration
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj_points, img_points, gray.shape[::-1], K, D, rvecs, tvecs,
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
            print(f"🎯 Done! RMS Error: {rms:.4f}")

            # Prepare Undistortion Maps for preview
            # Note: balance=0.0 crops to valid pixels, balance=1.0 keeps all pixels
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
            calibrated = True
        except Exception as e:
            print(f"❌ Calibration failed: {e}")

    # SAVE AND EXIT
    elif key == ord('s') and calibrated:
        filename = f"calib_fisheye_port_{PORT}.npz"
        np.savez(filename, K=K, D=D)
        print(f"💾 Saved to {filename}. Exiting.")
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
