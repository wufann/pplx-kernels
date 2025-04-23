#pragma once

#include <nvshmem.h>

#include <vector>

namespace pplx {

class Distributed {
public:
  Distributed();

  virtual ~Distributed();

  template <typename T> std::vector<T> allToAll(const std::vector<T> &input) {
    std::vector<T> output(input.size());
    allToAllImpl(input.data(), output.data(), sizeof(T), input.size());
    return output;
  }

protected:
  virtual void allToAllImpl(const void *input, void *output, size_t size, size_t count) = 0;
};

class DistributedNVSHMEM final : public Distributed {
public:
  DistributedNVSHMEM(unsigned rank, unsigned worldSize);

private:
  void allToAllImpl(const void *input, void *output, size_t size, size_t count) override;

private:
  unsigned rank;
  unsigned worldSize;
};

} // namespace pplx
