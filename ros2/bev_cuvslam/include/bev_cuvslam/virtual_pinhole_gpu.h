// GPU side of the virtual-pinhole carve. Kept as a C header so the .cu and the node's .cpp
// agree on the descriptor layout without either including the other's toolchain.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// One virtual pinhole: where to read and where to write.
//
// `src`/`spitch` live here rather than as kernel parameters because the Argus NVMM buffers
// are re-registered per frame and their device pointers and pitches are not guaranteed
// stable. Re-uploading 8 of these is under 200 bytes a frame, which is free next to being
// wrong once.
//
// `map` is width*height float2 of SOURCE pixel coordinates; negative means "this virtual
// pixel has no source", which happens past the fisheye's valid circle.
struct VPinDesc {
  const float2* map;
  const uint8_t* src;
  uint8_t* dst;
  int spitch;
  int sw, sh;
  int _pad;
};

#ifdef __cplusplus
extern "C" {
#endif

// One launch for all n_vcam virtual cameras (gridDim.z indexes them). The CPU path paid
// ~1 ms per cv::remap call in thread dispatch, eight times a set (task 4.5b); this pays it
// once, and the GPU has no equivalent cost.
cudaError_t launch_vpin_remap(const VPinDesc* desc, int n_vcam, int W, int H,
                              cudaStream_t stream);

#ifdef __cplusplus
}
#endif
