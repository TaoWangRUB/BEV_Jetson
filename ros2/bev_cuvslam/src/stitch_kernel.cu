// Equirectangular stitch kernel (surround-panorama). Per output pixel, weight-blend the
// in-FOV cameras' bilinear samples. Maps (per-camera uv + weight) are precomputed on the host
// from the KB calib + rig extrinsics and uploaded once. mono8 in/out. extern "C" launcher so
// the gcc-9 ROS node can call it across the nvcc/g++-8 boundary.
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ float sample_mono(const uint8_t* img, int pitch, int w, int h,
                                             float u, float v) {
  if (u < 0.0f || v < 0.0f || u > w - 1.0f || v > h - 1.0f) return 0.0f;
  int x0 = (int)floorf(u), y0 = (int)floorf(v);
  int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
  float fx = u - x0, fy = v - y0;
  const uint8_t* r0 = img + (size_t)y0 * pitch;
  const uint8_t* r1 = img + (size_t)y1 * pitch;
  float a = r0[x0] * (1.0f - fx) + r0[x1] * fx;
  float b = r1[x0] * (1.0f - fx) + r1[x1] * fx;
  return a * (1.0f - fy) + b * fy;
}

__global__ void equirect_stitch(uint8_t* out, int W, int H,
    const uint8_t* c0, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3,
    int cpitch, int cw, int ch,
    const float2* uv0, const float2* uv1, const float2* uv2, const float2* uv3,
    const float* w0, const float* w1, const float* w2, const float* w3) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  int idx = y * W + x;
  const uint8_t* cams[4] = {c0, c1, c2, c3};
  const float2* uvs[4] = {uv0, uv1, uv2, uv3};
  const float* ws[4] = {w0, w1, w2, w3};
  float acc = 0.0f, wsum = 0.0f;
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    float w = ws[c][idx];
    if (w > 0.0f) {
      float2 uv = uvs[c][idx];
      acc += w * sample_mono(cams[c], cpitch, cw, ch, uv.x, uv.y);
      wsum += w;
    }
  }
  out[idx] = wsum > 0.0f ? (uint8_t)fminf(255.0f, acc / wsum + 0.5f) : 0;
}

extern "C" void launch_equirect_stitch(uint8_t* out, int W, int H,
    const void* c0, const void* c1, const void* c2, const void* c3,
    int cpitch, int cw, int ch,
    const void* uv0, const void* uv1, const void* uv2, const void* uv3,
    const void* w0, const void* w1, const void* w2, const void* w3) {
  dim3 blk(16, 16), grd((W + 15) / 16, (H + 15) / 16);
  equirect_stitch<<<grd, blk>>>(out, W, H,
    (const uint8_t*)c0, (const uint8_t*)c1, (const uint8_t*)c2, (const uint8_t*)c3, cpitch, cw, ch,
    (const float2*)uv0, (const float2*)uv1, (const float2*)uv2, (const float2*)uv3,
    (const float*)w0, (const float*)w1, (const float*)w2, (const float*)w3);
}
