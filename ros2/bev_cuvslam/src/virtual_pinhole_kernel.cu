// Virtual-pinhole carve on the GPU: bilinear gather from the Argus NVMM Y planes straight
// into the 8 virtual pinhole images cuVSLAM consumes. mono8 in, mono8 out, extern "C"
// launcher so the node stays a plain .cpp (same arrangement as stitch_kernel.cu).
//
// WHY THIS EXISTS. The modular node does this same carve with cv::remap on the CPU and it
// costs ~31 ms per set against a 33 ms budget at 30 Hz (task 4.5). Measuring the CPU path
// showed the cost is not the resolution - halving the virtual output buys 39% and quartering
// the source buys 7% - but ~8.5 ms of fixed per-call overhead plus memory bandwidth on a
// cache-hostile gather (4.5b). Doing it here keeps the pixels on the GPU they already live
// on: no NVMM->CPU copy, no DDS hop, no cv_bridge.
//
// ONE LAUNCH, NOT EIGHT. gridDim.z indexes the virtual camera. The CPU version paid ~1 ms
// per cv::remap call spinning up six threads eight times a set; a batched launch pays that
// once, and the GPU has no equivalent dispatch cost to begin with.
//
// The maps are the SAME ones the modular node builds (bev_cuvslam/virtual_pinhole.hpp,
// BuildVirtualPinhole), converted to float2 and uploaded once at startup - so the two nodes
// carve identically by construction rather than by two implementations agreeing.
#include <cstdint>

#include "bev_cuvslam/virtual_pinhole_gpu.h"

namespace {

__global__ void vpin_remap_batch(const VPinDesc* __restrict__ desc, int W, int H,
                                 const uint8_t* __restrict__ s0, const uint8_t* __restrict__ s1,
                                 const uint8_t* __restrict__ s2, const uint8_t* __restrict__ s3,
                                 int sw, int sh, int spitch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  const VPinDesc d = desc[blockIdx.z];

  const uint8_t* src = d.src_idx == 0 ? s0 : d.src_idx == 1 ? s1 : d.src_idx == 2 ? s2 : s3;
  const float2 m = d.map[y * W + x];

  // Outside the fisheye's valid image the map carries a negative sentinel; write black
  // rather than clamping, so an invalid region reads as "no data" and not as a smeared
  // copy of the border - cuVSLAM would happily track features on the smear.
  uint8_t out = 0;
  if (m.x >= 0.0f && m.y >= 0.0f) {
    const float fx = floorf(m.x), fy = floorf(m.y);
    const int x0 = (int)fx, y0 = (int)fy;
    if (x0 >= 0 && y0 >= 0 && x0 + 1 < sw && y0 + 1 < sh) {
      const float ax = m.x - fx, ay = m.y - fy;
      const uint8_t* r0 = src + (size_t)y0 * spitch + x0;
      const uint8_t* r1 = r0 + spitch;
      const float top = r0[0] * (1.0f - ax) + r0[1] * ax;
      const float bot = r1[0] * (1.0f - ax) + r1[1] * ax;
      out = (uint8_t)lrintf(top * (1.0f - ay) + bot * ay);
    }
  }
  d.dst[y * W + x] = out;
}

}  // namespace

extern "C" cudaError_t launch_vpin_remap(const VPinDesc* desc, int n_vcam, int W, int H,
                                         const uint8_t* const* srcs, int sw, int sh, int spitch,
                                         cudaStream_t stream) {
  const dim3 blk(32, 8), grd((W + blk.x - 1) / blk.x, (H + blk.y - 1) / blk.y, n_vcam);
  vpin_remap_batch<<<grd, blk, 0, stream>>>(desc, W, H, srcs[0], srcs[1], srcs[2], srcs[3],
                                            sw, sh, spitch);
  return cudaGetLastError();
}
