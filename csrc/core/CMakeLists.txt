# Core library with common kernels.

add_library(core_lib STATIC
    kernels.cu
    distributed.cpp
)
target_link_libraries(core_lib PUBLIC
    CUDA::cudart
    nvshmem::nvshmem
)
set_cuda_compile_options(core_lib)
