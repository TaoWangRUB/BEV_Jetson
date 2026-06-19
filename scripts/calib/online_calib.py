#!/usr/bin/env python3
"""
Thread-Safe Online interactive fisheye (Kannala-Brandt) intrinsic calibration for Jetson TX2.
Fixes the shared memory buffer mutation bug and automatically captures raw images to disk.

Usage:
    python3 online_calib.py --id 0 --board 11x9 --square 30 --flip 180 --width 1920 --height 1080
"""

import argparse
import os
import sys
import cv2
import numpy as np
import threading

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0, help="Camera sensor ID")
    ap.add_argument("--board", default="11x9", help="Checkerboard SQUARES (e.g., 11x9)")
    ap.add_argument("--square", type=float, default=30.0, help="Square size in mm")
    ap.add_argument("--width", type=int, default=1280, help="Image width")
    ap.add_argument("--height", type=int, default=720, help="Image height")
    ap.add_argument("--fps", type=int, default=30, help="Preview framerate")
    ap.add_argument("--flip", type=int, default=0, help="nvvidconv flip-method (0..7) or degrees (0,90,180,270)")
    ap.add_argument("--min-views", type=int, default=15, help="Minimum views required to calibrate")
    ap.add_argument("--out", default="config/calib", help="Output directory")
    return ap.parse_args()

def get_gstreamer_pipeline(sensor_id, w, h, fps, flip):
    flip_map = {0: 0, 90: 1, 180: 2, 270: 3}
    if flip in flip_map:
        flip = flip_map[flip]
        
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false max-buffers=1"
    )

def detection_worker(gray_small, full_frame_snapshot, board_corners, find_flags, state_lock, shared_state):
    img_found, img_corners = cv2.findChessboardCorners(gray_small, board_corners, find_flags)
    
    with state_lock:
        shared_state["found"] = img_found
        if img_found:
            shared_state["corners"] = img_corners * 2.0
            shared_state["match_frame"] = full_frame_snapshot
            shared_state["match_gray"] = cv2.cvtColor(full_frame_snapshot, cv2.COLOR_BGR2GRAY)
        else:
            shared_state["corners"] = None
        shared_state["processing"] = False

def main():
    args = parse_args()
    
    sx, sy = (int(v) for v in args.board.lower().split("x"))
    board_corners = (sx - 1, sy - 1)
    
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    
    objp = np.zeros((1, board_corners[0] * board_corners[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:board_corners[0], 0:board_corners[1]].T.reshape(-1, 2)
    objp *= args.square

    objpoints, imgpoints, captured_frames = [], [], []
    cov = np.zeros((3, 3), int)
    
    # Setup paths for storage up front
    os.makedirs(args.out, exist_ok=True)
    raw_img_dir = os.path.join(args.out, f"cam{args.id}_raw")
    os.makedirs(raw_img_dir, exist_ok=True)
    
    pipeline = get_gstreamer_pipeline(args.id, args.width, args.height, args.fps, args.flip)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        sys.exit("Error: Failed to open nvarguscamerasrc pipeline.")
        
    print("\n--- Threaded Live 1080p Calibration (Auto-Save Enabled) ---")
    print(f" Raw image frames will save to: {raw_img_dir}/")
    print(" [SPACE] -> Lock-in current frame and save raw image to disk")
    print(" [C]     -> Process collected data and run Calibration")
    print(" [Q]     -> Quit/Abort script")
    print("-----------------------------------------------------------\n")

    state_lock = threading.Lock()
    shared_state = {
        "processing": False,
        "found": False,
        "corners": None,
        "match_frame": None,
        "match_gray": None
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        display_frame = frame.copy()
        
        with state_lock:
            is_processing = shared_state["processing"]
            
        if not is_processing:
            with state_lock:
                shared_state["processing"] = True
            
            frame_snapshot = frame.copy()
            gray = cv2.cvtColor(frame_snapshot, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            
            worker_thread = threading.Thread(
                target=detection_worker, 
                args=(gray_small, frame_snapshot, board_corners, find_flags, state_lock, shared_state)
            )
            worker_thread.daemon = True
            worker_thread.start()

        with state_lock:
            local_found = shared_state["found"]
            local_corners = shared_state["corners"]
            local_match_gray = shared_state["match_gray"]
            local_match_frame = shared_state["match_frame"]

        if local_found and local_corners is not None:
            cv2.drawChessboardCorners(display_frame, board_corners, local_corners, local_found)
            cv2.putText(display_frame, "READY (Press SPACE to capture)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Searching for board...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
        cv2.putText(display_frame, f"Collected Views: {len(imgpoints)}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(f"TX2 Live Cam {args.id} Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            if local_found and local_match_gray is not None:
                corners_refined = cv2.cornerSubPix(local_match_gray, local_corners.copy(), (5, 5), (-1, -1), subpix_criteria)
                
                objpoints.append(objp.copy())
                imgpoints.append(corners_refined)
                captured_frames.append(local_match_frame.copy())
                
                # NEW: Save raw 1080p source image instantly to the file system
                img_filename = os.path.join(raw_img_dir, f"frame_{len(imgpoints):03d}.jpg")
                # Using a high JPEG quality setting (95) to preserve crisp edges for processing
                cv2.imwrite(img_filename, local_match_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                cx, cy = corners_refined[:, 0, 0].mean(), corners_refined[:, 0, 1].mean()
                col_idx = min(2, int(cx / args.width * 3))
                row_idx = min(2, int(cy / args.height * 3))
                cov[row_idx, col_idx] += 1
                
                print(f"\n[+] Target locked! Frame saved: {os.path.basename(img_filename)}")
                print(f"    Total collected frames: {len(imgpoints)}")
                print("Current 3x3 Spatial Coverage Matrix Map:")
                for r in cov:
                    print(f"   {' '.join(f'{v:3d}' for v in r)}")
            else:
                print("[!] Warning: No valid board tracked in this processing window. Skipping.")
                
        elif key == ord('c') or key == ord('C'):
            if len(imgpoints) < args.min_views:
                print(f"[!] Target count insufficient. Got {len(imgpoints)}, need >= {args.min_views}")
                continue
            print("\nComputing Kannala-Brandt calibration matrices... Please wait...")
            break
            
        elif key == ord('q') or key == ord('Q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit("Calibration cancelled by user.")

    cap.release()
    cv2.destroyAllWindows()
    
    size = (args.width, args.height)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, size, K, D, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    
    print(f"\n==========================================")
    print(f"RMS Reprojection Error: {rms:.4f} pixels (Target < ~0.5)")
    print(f"==========================================")
    
    prevdir = os.path.join(args.out, f"cam{args.id}_preview")
    os.makedirs(prevdir, exist_ok=True)
    
    for idx, sample_img in enumerate(captured_frames[:3]):
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, size, cv2.CV_16SC2)
        undistorted = cv2.remap(sample_img, m1, m2, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(prevdir, f"undist_sample_{idx}.jpg"), undistorted)

    np.savez(os.path.join(args.out, f"cam{args.id}.npz"), K=K, D=D, size=size, rms=rms)
    
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k = D.ravel()
    
    yaml_path = os.path.join(args.out, f"cam{args.id}.yaml")
    with open(yaml_path, "w") as fp:
        fp.write(f"""%YAML:1.0
---
model_type: KANNALA_BRANDT
camera_name: cam{args.id}
image_width: {size[0]}
image_height: {size[1]}
distortion_parameters:
   k2: {k[0]:.10f}
   k3: {k[1]:.10f}
   k4: {k[2]:.10f}
   k5: {k[3]:.10f}
projection_parameters:
   mu: {fx:.10f}
   mv: {fy:.10f}
   u0: {cx:.10f}
   v0: {cy:.10f}
""")
    print(f"\nOutputs written to:\n -> {yaml_path}\n -> {prevdir}/\n")

if __name__ == "__main__":
    main()
