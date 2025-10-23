#include "core/distributed.h"
#include "core/cuda_utils.h"
#include "core/utils.h"

using namespace pplx;

Distributed::Distributed(unsigned rank, unsigned worldSize)
    : rank(rank),
      worldSize(worldSize) {}

Distributed::~Distributed() {}

DistributedNVSHMEM::DistributedNVSHMEM(unsigned rank, unsigned worldSize)
    : Distributed(rank, worldSize) {}

void DistributedNVSHMEM::allToAllImpl(const void *input, void *output, size_t size, size_t count) {
  PPLX_ASSERT(count == worldSize, "count must be equal to world size");

  void *srcBuffer = rocshmem::rocshmem_malloc(size * count);
  PPLX_ASSERT(srcBuffer != nullptr, "Failed to allocate src buffer");
  void *dstBuffer = rocshmem::rocshmem_malloc(size * count);
  PPLX_ASSERT(dstBuffer != nullptr, "Failed to allocate dst buffer");

  CUDACHECK(hipMemcpy(srcBuffer, input, size * count, hipMemcpyHostToDevice));

  // rocshmem::rocshmem_alltoallmem(ROCSHMEM_TEAM_WORLD, dstBuffer, srcBuffer, size);
  rocshmem::rocshmem_quiet();

  CUDACHECK(hipMemcpy(output, dstBuffer, size * count, hipMemcpyDeviceToHost));

  rocshmem::rocshmem_free(dstBuffer);
  rocshmem::rocshmem_free(srcBuffer);
}
