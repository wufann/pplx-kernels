cmake_minimum_required(VERSION 3.22)

# === Configuration options ===
option(WITH_TESTS "Build tests" OFF)
option(WITH_BENCHMARKS "Build benchmarks" OFF)
option(WITH_ROCM "Build with ROCm" OFF)

# 根据构建类型设置语言和项目配置
if(WITH_ROCM)
    project(PPLXKernels
            VERSION 0.0.1
            DESCRIPTION "PPLX Kernels"
            LANGUAGES CXX C)
    set(CMAKE_HIP_ARCHITECTURES "gfx950" CACHE STRING "HIP architecture to target")
    enable_language(HIP)
    if(DEFINED PYTORCH_ROCM_ARCH)
        set(CMAKE_HIP_ARCHITECTURES ${PYTORCH_ROCM_ARCH})
    endif()
else()
    project(PPLXKernels
            VERSION 0.0.1
            DESCRIPTION "PPLX Kernels"
            LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 90a CACHE STRING "CUDA architecture to target")
endif()

# === CMake configuration ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(NOT WITH_ROCM)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
else()
    set(ROCSHMEM_INCLUDE_DIR /root/rocshmem/include)
    set(ROCSHMEM_LIB_DIR /root/rocshmem/lib)
    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_OPERATORS__=1)
    # list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_CONVERSIONS__=1)
    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_BFLOAT16_CONVERSIONS__=1)
    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF2_OPERATORS__=1)
    list(APPEND HIP_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    list(APPEND HIP_CXX_FLAGS -mavx2)
    list(APPEND HIP_CXX_FLAGS -mf16c)
    list(APPEND HIP_CXX_FLAGS -mfma)
endif()

# === Dependencies ===
include(FetchContent)

# 根据构建类型查找不同的依赖
if(WITH_ROCM)
    find_package(HIP REQUIRED)
    # find_package(ROCM REQUIRED)
    find_package(ROCSHMEM REQUIRED HINTS /root/rocshmem/lib/cmake/rocshmem)
    
    # 为 ROCm 构建添加 MPI 支持
    set(MPI_DIR /opt/dependencies/ompi)
    find_package(MPI REQUIRED HINTS ${MPI_DIR})
else()
    find_package(CUDAToolkit REQUIRED)  # Modern replacement for find_package(CUDA)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)

# NVSHMEM目前只支持CUDA，ROCm构建需要跳过或使用替代方案
if(NOT WITH_ROCM)
    find_package(NVSHMEM REQUIRED)
endif()

if(WITH_TESTS)
    enable_testing()
    find_package(MPI REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(NCCL nccl)
endif()

# Create imported target for PyTorch
add_library(torch_imported INTERFACE)
add_library(torch::py_limited ALIAS torch_imported)
target_include_directories(torch_imported SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
# NOTE(lequn): We don't link against all ${TORCH_LIBRARIES} because we use py_limited_api.
# See: https://github.com/pytorch/pytorch/blob/9017becf1d895999a1c819c9d35b8139c090e7a9/torch/utils/cpp_extension.py#L1256-L1270
if(WITH_ROCM)
    target_link_libraries(torch_imported INTERFACE c10 torch torch_cpu hip::host)
else()
    target_link_libraries(torch_imported INTERFACE c10 torch torch_cpu c10_cuda torch_cuda CUDA::cudart)
endif()

# === Compiler and linker flags ===
add_compile_options(-Wno-deprecated-declarations)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_compile_definitions(Py_LIMITED_API=0x03090000)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# CUDA/HIP-specific compile options function
if(WITH_ROCM)
    function(set_cuda_compile_options target)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:--threads=32 -O3>)
    endfunction()
else()
    function(set_cuda_compile_options target)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--threads=32 -O3>)
    endfunction()
endif()

# === Library targets ===
add_subdirectory(all_to_all)
add_subdirectory(core)

# Main shared library
add_library(pplx_kernels SHARED
    bindings/all_to_all_ops.cpp
    bindings/bindings.cpp
)
target_link_libraries(pplx_kernels PUBLIC
    all_to_all_internode_lib
    all_to_all_intranode_lib
    core_lib
    torch::py_limited
    Python::Module
)

# 根据构建类型链接不同的运行时库
if(WITH_ROCM)
    target_link_libraries(pplx_kernels PUBLIC hip::host)
else()
    target_link_libraries(pplx_kernels PUBLIC
        CUDA::cuda_driver
        CUDA::cudart
        nvshmem::nvshmem_host
        nvshmem::nvshmem_device
    )
endif()

set_target_properties(pplx_kernels PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/pplx_kernels
)

if(NOT WITH_ROCM)
    set_target_properties(pplx_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()
