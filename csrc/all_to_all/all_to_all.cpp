#include "all_to_all.h"

#include "core/cuda_utils.h"
#include "core/utils.h"

#include <cuda_runtime.h>

using namespace pplx;

namespace {
template <typename T> T *mallocZeroBuffer(size_t size) {
  T *ptr;
  CUDACHECK(cudaMalloc(&ptr, size * sizeof(T)));
  cudaMemset(ptr, 0, size * sizeof(T));
  return ptr;
}
} // namespace

AllToAll::AllToAll(
    size_t maxNumTokens,
    size_t numExperts,
    size_t expertsPerToken,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    size_t hiddenDimBytes,
    size_t hiddenDimScaleBytes
)
    : maxNumTokens(maxNumTokens),
      numExperts(numExperts),
      numLocalExperts(ceil_div<uint32_t>(numExperts, worldSize)),
      numDPGroups(ceil_div<uint32_t>(worldSize, dpSize)),
      expertsPerToken(expertsPerToken),
      hiddenDim(hiddenDim),
      hiddenDimBytes(hiddenDimBytes),
      hiddenDimScaleBytes(hiddenDimScaleBytes),
      rank(rank),
      worldSize(worldSize),
      dpSize(dpSize),
      maxBatchTokens(numLocalExperts * numDPGroups * maxNumTokens) {

  ROSE_ASSERT(hiddenDimBytes % 16 == 0, "invalid hidden dim bytes");
  ROSE_ASSERT(hiddenDimScaleBytes % 16 == 0, "invalid hidden dim scale bytes");
  const size_t perTokenBytes =
      round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + sizeof(uint32_t), 16);
  const size_t maxBatchTokens = numLocalExperts * numDPGroups * maxNumTokens;

  ROSE_ASSERT(numLocalExperts != 0, "numLocalExperts is 0");
  ROSE_ASSERT(numDPGroups > 1, "at least 2 DP groups are required");
  ROSE_ASSERT(hiddenDimScaleBytes <= hiddenDimBytes, "invalid hidden dim bytes");

  // Buffers for token tracking.
  numTokensPerDP = mallocZeroBuffer<uint32_t>(numLocalExperts * numDPGroups);
  sourceIndex = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceExpert = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceOffset = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceGroup = mallocZeroBuffer<uint32_t>(maxBatchTokens);
}

AllToAll::~AllToAll() {
  CUDACHECK(cudaFree(numTokensPerDP));
  CUDACHECK(cudaFree(sourceIndex));
  CUDACHECK(cudaFree(sourceExpert));
  CUDACHECK(cudaFree(sourceOffset));
  CUDACHECK(cudaFree(sourceGroup));
}
