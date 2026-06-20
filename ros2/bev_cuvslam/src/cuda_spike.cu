// CUDA-in-colcon build spike (surround-panorama task 1): verify a .cu kernel compiled by
// nvcc (host g++-8, sm_62) builds + links + runs inside the colcon (gcc-9) workspace.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void fill(float* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = i * 2.0f;
}

int main() {
  const int n = 1024;
  float* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) { printf("cudaMalloc FAIL\n"); return 2; }
  fill<<<(n + 255) / 256, 256>>>(d, n);
  cudaDeviceSynchronize();
  float h[n];
  cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d);
  printf("CUDA-in-colcon spike: a[10]=%.1f (expect 20.0) -> %s\n", h[10], h[10] == 20.0f ? "OK" : "FAIL");
  return h[10] == 20.0f ? 0 : 1;
}
