# All-to-All library

add_library(all_to_all_lib STATIC
    all_to_all.cpp
    internode_dispatch.cu
    internode_combine.cu
    internode.cpp
)
target_link_libraries(all_to_all_lib PUBLIC
    nvshmem::nvshmem_host
    nvshmem::nvshmem_device
    CUDA::cudart
)
set_cuda_compile_options(all_to_all_lib)

if(WITH_TESTS)
    # All-to-All test
    add_executable(test_all_to_all
        test_all_to_all.cpp
    )
    target_link_libraries(test_all_to_all PUBLIC
        all_to_all_lib
        CUDA::cudart
        CUDA::cuda_driver
        MPI::MPI_CXX
        nvshmem::nvshmem_host
        nvshmem::nvshmem_device
    )
    set_cuda_compile_options(test_all_to_all)
    add_test(NAME AllToAllTest
             COMMAND ${MPIEXEC_EXECUTABLE} -np 4 $<TARGET_FILE:test_all_to_all>)
    set_tests_properties(AllToAllTest PROPERTIES ENVIRONMENT "NVSHMEM_REMOTE_TRANSPORT=None")
endif()

if (WITH_BENCHMARKS)
    add_executable(bench_all_to_all
        bench_all_to_all.cpp
    )
    target_link_libraries(bench_all_to_all PUBLIC
        all_to_all_lib
        CUDA::cudart
        CUDA::cuda_driver
        MPI::MPI_CXX
        nvshmem::nvshmem_host
        nvshmem::nvshmem_device
    )
endif()
