// Runtime smoke test for the CUDA-10.2 cuVSLAM port on the TX2.
// Calls GetVersion() (no GPU) then WarmUpGPU() (initializes the CUDA/GPU context
// + warms kernels) — if this runs clean on the r440 driver, the ported GPU path
// executes, not just compiles.
#include "cuvslam2.h"
#include <cstdio>

int main() {
  int major = 0, minor = 0, patch = 0;
  cuvslam::GetVersion(&major, &minor, &patch);
  printf("cuVSLAM version: %d.%d.%d\n", major, minor, patch);
  printf("Calling WarmUpGPU() ...\n");
  fflush(stdout);
  cuvslam::WarmUpGPU();
  printf("OK: WarmUpGPU() completed — GPU/CUDA context + kernels initialized on the r440 driver.\n");
  return 0;
}
