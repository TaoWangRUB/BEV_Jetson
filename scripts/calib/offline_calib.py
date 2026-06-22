#!/usr/bin/env python3
"""
Offline Outlier-Rejection Fisheye Calibration Engine.
Processes saved raw frames, injects intrinsic guesses, and purges bad views.

Usage:
    python3 offline_calib.py --id 1 --board 11x9 --square 30.0
"""

import argparse
import os
import glob
import cv2
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser(description="Offline Fisheye Calibration Engine")
    ap.add_argument("--id", type=int, default=1, help="Camera sensor ID (matches your camX folder prefix)")
    ap.add_argument("--board", default="11x9", help="Checkerboard SQUARES count (columns x rows, e.g., 11x9)")
    ap.add_argument("--square", type=float, default=30.0, help="Square size in mm")
    ap.add_argument("--out", default="config/calib", help="Output base directory containing the raw folders")
    ap.add_argument("--img-dir", default=None, help="Explicit path to raw images (overrides standard --out/camX_raw mapping)")
    return ap.parse_args()

def main():
    args = parse_args()
    
    # Dynamically build paths based on parsed parameters
    cam_id = args.id
    out_dir = args.out
    
    if args.img_dir:
        image_dir = args.img_dir
    else:
        image_dir = os.path.join(out_dir, f"cam{cam_id}_raw")
        
    sx, sy = (int(v) for v in args.board.lower().split("x"))
    board_corners = (sx - 1, sy - 1)
    square_size = args.square
    
    # 3D object points configuration
    objp = np.zeros((1, board_corners[0] * board_corners[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:board_corners[0], 0:board_corners[1]].T.reshape(-1, 2)
    objp *= square_size

    # Read saved images
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not img_paths:
        print(f"[-] Error: No images found in targeting directory: {image_dir}")
        return

    print(f"\n--- Running Offline Calibration Engine ---")
    print(f" Target Folder:  {image_dir}/")
    print(f" Target Board:   {args.board} grid size ({square_size}mm squares)")
    print(f" Output Destination: {out_dir}/cam{cam_id}.yaml")
    print(f"-------------------------------------------\n")
    print(f"[+] Found {len(img_paths)} saved raw frames. Extracting corners at full resolution...")
    
    objpoints = []
    imgpoints = []
    valid_filenames = []
    size = None

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        if size is None:
            size = (img.shape[1], img.shape[0])
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_corners, find_flags)
        
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), subpix_criteria)
            objpoints.append(objp.copy())
            imgpoints.append(corners_refined)
            valid_filenames.append(path)
        else:
            print(f" [-] Skipping {os.path.basename(path)}: Checkerboard pattern not clear.")

    if len(imgpoints) < 10:
        print("[-] Error: Too few valid images recognized. Need at least 10 clean frames.")
        return

    print(f"[+] Successfully extracted tracking data from {len(imgpoints)} frames.")
    print("[+] Running iterative optimizer with outlier rejection loops...\n")

    # Optimization loop
    while len(imgpoints) >= 10:
        # Seed OpenCV with a safe initial focal-length guess based on sensor geometry
        fx_guess = size[0] / np.pi  
        fy_guess = size[1] / np.pi
        cx_guess = size[0] / 2.0
        cy_guess = size[1] / 2.0
        
        K = np.array([[fx_guess, 0, cx_guess],
                      [0, fy_guess, cy_guess],
                      [0, 0, 1]], dtype=np.float32)
        D = np.zeros((4, 1), dtype=np.float32)
        
        flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS + 
                 cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + 
                 cv2.fisheye.CALIB_FIX_SKEW)

        try:
            rms, _, _, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints, imgpoints, size, K, D, flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        except cv2.error as e:
            print(f"[-] Mathematical optimization broke down: {e}")
            break

        # Calculate exact error contribution for every single frame
        frame_errors = []
        for i in range(len(objpoints)):
            imgpoints_projected, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
            
            pts_measured = imgpoints[i].reshape(-1, 2)
            pts_projected = imgpoints_projected.reshape(-1, 2)
            
            err = cv2.norm(pts_measured, pts_projected, cv2.NORM_L2) / np.sqrt(len(pts_projected))
            frame_errors.append(err)

        max_err_idx = np.argmax(frame_errors)
        max_err = frame_errors[max_err_idx]
        worst_frame_name = os.path.basename(valid_filenames[max_err_idx])

        print(f" Current Global RMS: {rms:.4f} pixels | Worst Frame: {worst_frame_name} ({max_err:.2f} px error)")

        # Target reached! Exit optimization loop.
        if rms <= 0.5 and max_err < 1.5:
            print("\n[+] Target convergence reached successfully!")
            break
            
        # If the worst frame is terribly corrupted, drop it and recalculate instantly
        if max_err > 1.2 or rms > 0.5:
            print(f"  [!] Purging outlier frame: {worst_frame_name}")
            objpoints.pop(max_err_idx)
            imgpoints.pop(max_err_idx)
            valid_filenames.pop(max_err_idx)
        else:
            break

    print(f"\n==========================================")
    print(f"FINAL OPTIMIZED RMS ERROR: {rms:.4f} pixels")
    print(f"Remaining Valid Analytical Views: {len(imgpoints)}")
    print(f"==========================================")

    # Save final optimized matrices
    yaml_path = os.path.join(out_dir, f"cam{cam_id}.yaml")
    k = D.ravel()
    with open(yaml_path, "w") as fp:
        fp.write(f"""%YAML:1.0
---
model_type: KANNALA_BRANDT
camera_name: cam{cam_id}
image_width: {size[0]}
image_height: {size[1]}
distortion_parameters:
   k2: {k[0]:.10f}
   k3: {k[1]:.10f}
   k4: {k[2]:.10f}
   k5: {k[3]:.10f}
projection_parameters:
   mu: {K[0,0]:.10f}
   mv: {K[1,1]:.10f}
   u0: {K[0,2]:.10f}
   v0: {K[1,2]:.10f}
""")
    print(f"\nCleaned parameters successfully written to:\n -> {yaml_path}\n")

if __name__ == "__main__":
    main()
