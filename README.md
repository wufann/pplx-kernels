Perplexity MoE Kernels
==========

Installation
-----

```
cd pplx-kernels
pip install -e . -vvv
```

Testing
-----

To build the C++ tests and benchmarks:

```
cd pplx-kernels
mkdir build-cmake
cd build-cmake

TORCH_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')

cmake ../csrc \
    -GNinja \
    -DCMAKE_PREFIX_PATH=$TORCH_PREFIX_PATH \
    -DTORCH_CUDA_ARCH_LIST=9.0a+PTX \
    -DWITH_TESTS=ON \
    -DWITH_BENCHMARKS=ON

ninja test_all_to_all bench_all_to_all
```

To run the all-to-all tests on one node:

```
NVSHMEM_REMOTE_TRANSPORT=None mpirun -np 4 ./test_all_to_all
```


To run the all-to-all benchmarks on one node:

```
NVSHMEM_REMOTE_TRANSPORT=None mpirun -np 4 ./bench_all_to_all
```


Inter-Node Benchmarks
-----

To test on a 32-device cluster spread across 4 nodes, run the following command on all nodes, alternating the rank from 0 to 3 and setting the master address to point to one of the nodes:

```
cd pplx-kernels
pip install -e . -vvv
NVSHMEM_IB_ENABLE_IBGDA=1 NODE_RANK=<rank> WORLD_SIZE=32 WORLD_LOCAL_SIZE=8 MASTER_ADDR=<master-address> MASTER_PORT=29500 python3 -m tests.bench_all_to_all
```
