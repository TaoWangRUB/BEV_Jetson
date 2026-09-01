// GPU side of the virtual-pinhole carve. Kept as a C header so the .cu and the node's .cpp
// agree on the descriptor layout without either including the other's toolchain.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// One virtual pinhole: where to read (map, into source `src_idx`) and where to write.
// `map` is W*H float2 of SOURCE pixel coordinates, negative meaning "outside the fisheye".
struct VPinDesc {
  const float2* map;
  uint8_t* dst;
  int src_idx;
  int _pad;
};

#ifdef __cplusplus
extern "C" {
#endif

// One launch for all n_vcam virtual cameras (gridDim.z indexes them). `srcs` is the 4 device
// Y-plane pointers for this frame; they change per frame, the descriptors do not.
cudaError_t launch_vpin_remap(const VPinDesc* desc, int n_vcam, int W, int H,
                              const uint8_t* const* srcs, int sw, int sh, int spitch,
                              cudaStream_t stream);

#ifdef __cplusplus
}
#endif
