import cv2
import numpy as np

# SETTINGS
CHECKERBOARD = (12, 8) # For your A3 board (internal corners)
PORT = 5000            # Change for each camera
# GStreamer pipeline for TX2
PIPE = f"udpsrc port={PORT} ! application/x-rtp,encoding-name=H264 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"

obj_points = [] 
img_points = [] 
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * 30 # 30mm

cap = cv2.VideoCapture(PIPE, cv2.CAP_GSTREAMER)

print(f"--- Camera Calibration Tool (Port {PORT}) ---")
print("1. Move board until corners are green.")
print("2. Press 'SPACE' to capture (aim for 20 images).")
print("3. Press 'C' to calculate final score.")

warning_shown = False
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

cap.release()
cv2.destroyAllWindows()