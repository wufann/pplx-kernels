# # Core library with common kernels.
# 
# add_library(core_lib STATIC
#     kernels.cu
#     distributed.cpp
# )
# target_link_libraries(core_lib PUBLIC
#     CUDA::cudart
# )
# target_link_libraries(core_lib INTERFACE
#     nvshmem::nvshmem_host
# )
# target_include_directories(core_lib PRIVATE ${NVSHMEM_INCLUDE_DIR})
# set_cuda_compile_options(core_lib)

# Core library with common kernels.

if(WITH_ROCM)
    add_library(core_lib STATIC
        kernels.cu
        distributed.cpp
    )
    target_link_libraries(core_lib PUBLIC
        hip::host
        hip::device
        hip::amdhip64
        /root/rocshmem/lib/librocshmem.a
    )
    message(STATUS "OMPI_INCLUDE_DIR = ${OMPI_INCLUDE_DIR}")
    message(STATUS "ROCSHMEM_INCLUDE_DIR = ${ROCSHMEM_INCLUDE_DIR}")
    target_include_directories(core_lib PRIVATE ${OMPI_INCLUDE_DIR} ${ROCSHMEM_INCLUDE_DIR})
else()
    add_library(core_lib STATIC
        kernels.cu
        distributed.cpp
    )
    target_link_libraries(core_lib PUBLIC
        CUDA::cudart
    )
    target_link_libraries(core_lib INTERFACE
        nvshmem::nvshmem_host
    )
    target_include_directories(core_lib PRIVATE ${NVSHMEM_INCLUDE_DIR})
endif()

set_cuda_compile_options(core_lib)