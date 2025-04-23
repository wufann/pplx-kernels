#pragma once

#define ROSE_ENABLE_DEVICE_ASSERT 1

#ifdef ROSE_ENABLE_DEVICE_ASSERT
#define ROSE_DEVICE_ASSERT(cond)                                                                   \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                         \
      asm("trap;");                                                                                \
    }                                                                                              \
  } while (0)
#else
#define ROSE_DEVICE_ASSERT(cond)
#endif

namespace pplx {
namespace device {
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

template <typename T> __device__ T ceil_div(T x, T y) { return (x + y - 1) / y; }

template <typename T> __device__ T round_up(T x, T y) { return ceil_div<T>(x, y) * y; }

} // namespace device
} // namespace pplx
