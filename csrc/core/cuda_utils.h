#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDACHECK(cmd)                                                                             \
  do {                                                                                             \
    cudaError_t e = cmd;                                                                           \
    if (e != cudaSuccess) {                                                                        \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)
