#pragma once

#include "core/common_utils.h"

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

#define PPLX_ENABLE_DEVICE_ASSERT 0

#if PPLX_ENABLE_DEVICE_ASSERT == 1
#define PPLX_DEVICE_ASSERT(cond)                                                                   \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                         \
      asm("trap;");                                                                                \
    }                                                                                              \
  } while (0)
#else
#define PPLX_DEVICE_ASSERT(cond)
#endif

namespace pplx {
namespace device {

// A wrapper for the kernels that is used to guard against compilation on
// architectures that will never use the kernel.
template <typename Kernel> struct enable_sm90_or_later : Kernel {
  template <typename... Args> __device__ void operator()(Args &&...args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

#ifdef __HIP_PLATFORM_AMD__
// HIP version
static __attribute__((always_inline)) __device__ unsigned warp_sum(unsigned value) {
  value += __shfl_xor(value, 16);
  value += __shfl_xor(value, 8);
  value += __shfl_xor(value, 4);
  value += __shfl_xor(value, 2);
  value += __shfl_xor(value, 1);
  return value;
}

static __attribute__((always_inline)) __device__ bool warp_and(bool value) {
  value &= __shfl_xor(value, 16);
  value &= __shfl_xor(value, 8);
  value &= __shfl_xor(value, 4);
  value &= __shfl_xor(value, 2);
  value &= __shfl_xor(value, 1);
  return value;
}

static __attribute__((always_inline)) __device__ float half_warp_reduce_max(float value) {
  value = max(value, __shfl_xor(value, 8));
  value = max(value, __shfl_xor(value, 4));
  value = max(value, __shfl_xor(value, 2));
  value = max(value, __shfl_xor(value, 1));
  return value;
}
#else
// CUDA version
__forceinline__ __device__ unsigned warp_sum(unsigned value) {
  value += __shfl_xor_sync(0xffffffff, value, 16);
  value += __shfl_xor_sync(0xffffffff, value, 8);
  value += __shfl_xor_sync(0xffffffff, value, 4);
  value += __shfl_xor_sync(0xffffffff, value, 2);
  value += __shfl_xor_sync(0xffffffff, value, 1);
  return value;
}

__forceinline__ __device__ bool warp_and(bool value) {
  value &= __shfl_xor_sync(0xffffffff, value, 16);
  value &= __shfl_xor_sync(0xffffffff, value, 8);
  value &= __shfl_xor_sync(0xffffffff, value, 4);
  value &= __shfl_xor_sync(0xffffffff, value, 2);
  value &= __shfl_xor_sync(0xffffffff, value, 1);
  return value;
}

__forceinline__ __device__ float half_warp_reduce_max(float value) {
  auto mask = __activemask();
  value = max(value, __shfl_xor_sync(mask, value, 8));
  value = max(value, __shfl_xor_sync(mask, value, 4));
  value = max(value, __shfl_xor_sync(mask, value, 2));
  value = max(value, __shfl_xor_sync(mask, value, 1));
  return value;
}
#endif

} // namespace device
} // namespace pplx
