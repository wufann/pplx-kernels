
#include <nvshmem.h>

#include <cassert>
#include <cstdint>

#include "all_to_all/internode.h"
#include "core/utils.h"

using namespace pplx;

AllToAllInterNode::AllToAllInterNode(
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
    : AllToAll(
          maxNumTokens,
          numExperts,
          expertsPerToken,
          rank,
          worldSize,
          dpSize,
          hiddenDim,
          hiddenDimBytes,
          hiddenDimScaleBytes
      ) {
  // Buffers for token counts.
  numTokensBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * numLocalExperts * numDPGroups);
  ROSE_ASSERT(numTokensBuffer != nullptr, "failed to allocate numTokensBuffer");
  cudaMemset(numTokensBuffer, 0, sizeof(uint64_t) * numLocalExperts * numDPGroups);

  numScatterRecvBuffer =
      (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * numLocalExperts * numDPGroups);
  ROSE_ASSERT(numScatterRecvBuffer != nullptr, "failed to allocate numScatterRecvBuffer");
  cudaMemset(numScatterRecvBuffer, 0, sizeof(uint64_t) * numLocalExperts * numDPGroups);

  gatherSignalBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * maxNumTokens);
  ROSE_ASSERT(gatherSignalBuffer != nullptr, "failed to allocate gatherSignalBuffer");
  cudaMemset(gatherSignalBuffer, 0, sizeof(uint64_t) * maxNumTokens);

  gatherSyncBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * worldSize);
  ROSE_ASSERT(gatherSyncBuffer != nullptr, "failed to allocate gatherSyncBuffer");
  cudaMemset(gatherSyncBuffer, 0, sizeof(uint64_t) * worldSize);

  // Buffers for scatter.
  const size_t perTokenBytes =
      round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + sizeof(uint32_t), 16);
  xScatterIn = (std::byte *)nvshmem_malloc(maxNumTokens * perTokenBytes);
  ROSE_ASSERT(xScatterIn != nullptr, "failed to allocate xScatterIn");
  xScatterOut = (std::byte *)nvshmem_malloc(maxBatchTokens * perTokenBytes);
  ROSE_ASSERT(xScatterOut != nullptr, "failed to allocate xScatterOut");

  // Buffers for gather. The allocations are a bit wider to accommodate all
  // possible data types (primarily float for testing and bfloat16 for prod).
  xGatherIn = (std::byte *)nvshmem_malloc(maxBatchTokens * hiddenDim * sizeof(float));
  ROSE_ASSERT(xGatherIn != nullptr, "failed to allocate xGatherIn");
  xGatherOut = (std::byte *)nvshmem_malloc(maxNumTokens * numExperts * hiddenDim * sizeof(float));
  ROSE_ASSERT(xGatherOut != nullptr, "failed to allocate xGatherOut");
}

AllToAllInterNode::~AllToAllInterNode() {
  nvshmem_free(numTokensBuffer);
  nvshmem_free(numScatterRecvBuffer);
  nvshmem_free(gatherSignalBuffer);
  nvshmem_free(gatherSyncBuffer);
  nvshmem_free(xScatterIn);
  nvshmem_free(xScatterOut);
  nvshmem_free(xGatherIn);
  nvshmem_free(xGatherOut);
}
