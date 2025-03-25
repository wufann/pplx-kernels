#pragma once

#include <torch/library.h>

namespace pplx {
void register_nvshmem_ops(torch::Library &m);
} // namespace pplx
