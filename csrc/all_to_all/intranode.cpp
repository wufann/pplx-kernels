#include "all_to_all/intranode.h"

#include "core/distributed.h"
#include "core/utils.h"

#include <cassert>
#include <cstdint>

using namespace pplx;

AllToAllIntraNode::AllToAllIntraNode(
    size_t maxNumTokens,
    size_t numExperts,
    size_t expertsPerToken,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    size_t hiddenDimBytes,
    size_t hiddenDimScaleBytes,
    std::shared_ptr<Distributed> distributed
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

  // Determine the per-token buffer size. Allocate extra storage for the index.
  // Pad to a 16-byte alignment to allow loads via int4.
  size_t bufferSize;
  {
    size_t dispatchBufferSize = 0;
    {
      const size_t metaSize = sizeof(uint32_t);
      const size_t dispatchPerTokenBytes =
          round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + metaSize, 16);

      dispatchBufferSize += sizeof(int4);
      dispatchBufferSize += round_up<size_t>(numLocalExperts * sizeof(uint32_t), sizeof(int4));
      dispatchBufferSize += maxNumTokens * numLocalExperts * dispatchPerTokenBytes;
    }

    size_t combineBufferSize = 0;
    {
      const size_t combinePerTokenBytes = round_up<size_t>(hiddenDim * sizeof(float), 16);
      combineBufferSize += sizeof(int4);
      combineBufferSize += round_up<size_t>(numLocalExperts * sizeof(uint32_t), sizeof(int4));
      combineBufferSize += maxNumTokens * numLocalExperts * combinePerTokenBytes;
    }

    // The size of the buffer is the max of dispatch and combine.
    bufferSize = std::max(dispatchBufferSize, combineBufferSize);
  }

  // Allocate pointers to the buffer collections.
  CUDACHECK(hipMalloc(&sendBuffersPtr, sizeof(std::byte *) * worldSize));
  CUDACHECK(hipMalloc(&recvBuffersPtr, sizeof(std::byte *) * worldSize));

  // On the current rank, allocate a buffer to communicate with every other rank.
  // Synchronize via the distributed group. Create indirect pointer arrays to buffers.
  {
    std::vector<hipIpcMemHandle_t> srcHandlesHost;
    for (unsigned i = 0; i < worldSize; i++) {
      auto &ptr = sendBuffers.emplace_back();
      auto &handle = srcHandlesHost.emplace_back();
      CUDACHECK(hipMalloc(&ptr, bufferSize));
      CUDACHECK(hipMemset(ptr, 0, bufferSize));
      CUDACHECK(hipIpcGetMemHandle(&handle, ptr));
    }

    auto dstHandlesHost = distributed->allToAll(srcHandlesHost);
    for (unsigned i = 0; i < worldSize; i++) {
      auto &ptr = recvBuffers.emplace_back();
      if (i == rank) {
        ptr = sendBuffers[i];
      } else {
        CUDACHECK(
            hipIpcOpenMemHandle((void **)&ptr, dstHandlesHost[i], hipIpcMemLazyEnablePeerAccess)
        );
      }
    }

    CUDACHECK(hipMemcpy(
        sendBuffersPtr, sendBuffers.data(), sizeof(std::byte *) * worldSize, hipMemcpyHostToDevice
    ));

    CUDACHECK(hipMemcpy(
        recvBuffersPtr, recvBuffers.data(), sizeof(std::byte *) * worldSize, hipMemcpyHostToDevice
    ));
  }

  // Allocate the local buffer for dispatch counts.
  CUDACHECK(hipMalloc(&localRecvCountPtr, sizeof(uint32_t) * maxNumTokens));
  CUDACHECK(hipMemset(localRecvCountPtr, 0, sizeof(uint32_t) * maxNumTokens));
  CUDACHECK(hipMalloc(&countBuffersPtr, sizeof(uint32_t *) * worldSize));
  {
    hipIpcMemHandle_t countHandle;
    CUDACHECK(hipIpcGetMemHandle(&countHandle, localRecvCountPtr));
    auto countHandlesHost = distributed->allGather(countHandle);

    countBuffers.resize(worldSize);
    for (unsigned i = 0; i < worldSize; i++) {
      if (i == rank) {
        countBuffers[i] = localRecvCountPtr;
      } else {
        CUDACHECK(hipIpcOpenMemHandle(
            (void **)&countBuffers[i], countHandlesHost[i], hipIpcMemLazyEnablePeerAccess
        ));
      }
    }

    CUDACHECK(hipMemcpy(
        countBuffersPtr, countBuffers.data(), sizeof(uint32_t *) * worldSize, hipMemcpyHostToDevice
    ));
  }

  // Allocate the local buffers.
  tokenCount = mallocZeroBuffer<uint32_t>(numExperts);
  numTokensPerRank = mallocZeroBuffer<uint32_t>(numLocalExperts * worldSize);

  // Buffers for token tracking.
  const size_t maxBatchTokens = numLocalExperts * maxNumTokens * worldSize;
  sourceIndex = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceExpert = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceOffset = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceRank = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceToken = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceRoute = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  tokenIndex = mallocZeroBuffer<uint32_t>(1);
}

AllToAllIntraNode::~AllToAllIntraNode() {
  for (unsigned i = 0; i < worldSize; i++) {
    CUDACHECK(hipFree(sendBuffers[i]));
    if (i != rank) {
      CUDACHECK(hipIpcCloseMemHandle(recvBuffers[i]));
      CUDACHECK(hipIpcCloseMemHandle(countBuffers[i]));
    }
  }

  CUDACHECK(hipFree(recvBuffersPtr));
  CUDACHECK(hipFree(sendBuffersPtr));
  CUDACHECK(hipFree(countBuffersPtr));
  CUDACHECK(hipFree(localRecvCountPtr));

  CUDACHECK(hipFree(tokenCount));
  CUDACHECK(hipFree(numTokensPerRank));

  CUDACHECK(hipFree(sourceIndex));
  CUDACHECK(hipFree(sourceExpert));
  CUDACHECK(hipFree(sourceOffset));
  CUDACHECK(hipFree(sourceRank));
  CUDACHECK(hipFree(sourceToken));
  CUDACHECK(hipFree(tokenIndex));
}
