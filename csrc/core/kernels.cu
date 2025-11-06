#include "core/cuda_utils.h"
#include "kernels.h"

#include <cstdint>

#include <hip/hip_runtime.h>

__global__ void sleep_kernel(uint64_t ms) {
#ifdef __HIP_PLATFORM_AMD__
  // HIP doesn't have __nanosleep, use a busy-wait loop instead
  uint64_t start = clock64();
  uint64_t clocks_per_ms = 1000000; // Approximate, may need tuning
  for (int i = 0; i < ms; i++) {
    while ((clock64() - start) < (i + 1) * clocks_per_ms) {
      // Busy wait
    }
  }
#else
  for (int i = 0; i < ms; i++) {
    __nanosleep(1000000);
  }
#endif
}

void pplx::sleepOnStream(double seconds, hipStream_t stream) {
  uint64_t ms = seconds * 1000;
  void *args[] = {&ms};
  dim3 grid(1);
  dim3 block(1);
  HIPCHECK(hipLaunchKernel((void *)sleep_kernel, grid, block, args, 0, stream));
}
