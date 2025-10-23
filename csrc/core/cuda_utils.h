#pragma once

#include <cstdio>
#include <cstdlib>
// #include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#define CUDACHECK(cmd)                                                                             \
  do {                                                                                             \
    hipError_t e = cmd;                                                                           \
    if (e != hipSuccess) {                                                                        \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, hipGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

namespace pplx {
template <typename T> T *mallocZeroBuffer(size_t size) {
  T *ptr;
  CUDACHECK(hipMalloc(&ptr, size * sizeof(T)));
  hipMemset(ptr, 0, size * sizeof(T));
  return ptr;
}

inline int get_sm_count() {
  int device;
  CUDACHECK(hipGetDevice(&device));
  int numSMs;
  CUDACHECK(hipDeviceGetAttribute(&numSMs, hipDeviceAttributeMultiprocessorCount, device));

  return numSMs;
}

} // namespace pplx
