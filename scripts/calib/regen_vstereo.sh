#!/bin/bash
# Regenerate the four virtual-stereo pairs from the raw pair recordings and report the
# epipolar residual for the measured rig against the ring-closed rig. Runs in the
# tartancalib container: docker run -v <dataset>:/data -v <repo>:/repo ... bash /repo/scripts/calib/regen_vstereo.sh
set -e
source /catkin_ws/devel/setup.bash
cd /data
declare -A SRC=( [front]=CAM_A-CAM_B [right]=CAM_B-CAM_C [rear]=CAM_C-CAM_D [left]=CAM_D-CAM_A )
declare -A FROM=( [front]=cam1 [right]=cam2 [rear]=cam4 [left]=cam3 )
declare -A TO=(   [front]=cam2 [right]=cam4 [rear]=cam3 [left]=cam1 )
export VS_W=768 VS_H=576 VS_FOV=160
for p in left front right rear; do
  a=${FROM[$p]}; b=${TO[$p]}
  python3 /repo/scripts/calib/gen_virtual_stereo.py \
    --bag /data/ros1/${SRC[$p]}.bag --out /data/ros1/vclosed_$p.bag \
    --topic-a /$a/image_raw --topic-b /$b/image_raw \
    --calib-a /repo/config/calib/imx296_1456x1088/$a.yaml \
    --calib-b /repo/config/calib/imx296_1456x1088/$b.yaml \
    --extrinsic /data/closed.yaml --pair $p --fov 160 --width 768 --height 576 >/dev/null 2>&1
  echo "--- $p"
  echo -n "  measured: "; python3 /repo/scripts/calib/vstereo_epipolar.py /data/ros1/v160_$p.bag    /repo/config/rig/rig_extrinsics_imx296.yaml $p 2>/dev/null | grep -E "median"
  echo -n "  closed:   "; python3 /repo/scripts/calib/vstereo_epipolar.py /data/ros1/vclosed_$p.bag /data/closed.yaml $p 2>/dev/null | grep -E "median"
done
