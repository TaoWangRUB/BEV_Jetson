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

        cv2.imshow(f"Calibration Port {PORT}", display)
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
                    np.savez(f"calib_port_{PORT}.npz", mtx=mtx, dist=dist)
                    print(f"SUCCESS! Data saved to calib_port_{PORT}.npz")
                else:
                    print("ERROR TOO HIGH. Try capturing more angles, especially edges.")

        elif key == ord('q'):
            break
except KeyboardInterrupt:
    print("\nStopped by user.")

cap.release()
cv2.destroyAllWindows()
