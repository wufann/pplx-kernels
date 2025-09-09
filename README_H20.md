# Install on H20
## Clone repo
```
git clone https://github.com/perplexityai/pplx-kernels.git
```
## Install nvshmem
```
wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
mkdir nvshmem_src_3.2.5-1
tar xf nvshmem_src_3.2.5-1.txz -C nvshmem_src_3.2.5-1
cd nvshmem_src_3.2.5-1/nvshmem_src
mkdir -p build
cd build
cmake \
    -DNVSHMEM_PREFIX=/opt/nvshmem-3.2.5 \
    -DCMAKE_CUDA_ARCHITECTURES=90a \
    -DNVSHMEM_MPI_SUPPORT=0 \
    -DNVSHMEM_PMIX_SUPPORT=0 \
    -DNVSHMEM_LIBFABRIC_SUPPORT=0 \
    -DNVSHMEM_IBRC_SUPPORT=0 \
    -DNVSHMEM_IBGDA_SUPPORT=0 \
    -DNVSHMEM_BUILD_TESTS=1 \
    -DNVSHMEM_BUILD_EXAMPLES=1 \
    -DNVSHMEM_BUILD_HYDRA_LAUNCHER=1 \
    -DNVSHMEM_BUILD_TXZ_PACKAGE=1 \
    -DMPI_HOME=/opt/hpcx/ompi/ \
    -DLIBFABRIC_HOME=/opt/amazon/efa \
    -G Ninja \
    ..
ninja build
sudo ninja install
遇到编译nvshmem报example/moe_shuffle.cu的问题，是因为没有moe_shuffle.cu没有#include <getopt.h>头文件  
解决方法：在moe_shuffle.cu中加上该头文件，或者注释CMakeLists.txt中的moe_shuffle.cu。
```
## Install pplx
```
cd pplx-kernels
source env_h20.sh
TORCH_CUDA_ARCH_LIST=9.0a+PTX python3 setup.py bdist_wheel
pip install dist/*.whl
```

# Single-node Test and benchmark
```
# test
pytest -svx --tb=short tests

# benchmark
python3 -m tests.bench_all_to_all
```

# Result
## Test
```
tests/test_all_to_all.py::test_all_to_all_multi_node[float16-bfloat16] SKIPPED (Requires multi-node environment)
tests/test_all_to_all.py::test_all_to_all_multi_node[float16-float8_e4m3fn] SKIPPED (Requires multi-node environment)
tests/test_all_to_all.py::test_all_to_all_multi_node[float16-float16] SKIPPED (Requires multi-node environment)
tests/test_all_to_all.py::test_all_to_all_multi_node[bfloat16-bfloat16] SKIPPED (Requires multi-node environment)
tests/test_all_to_all.py::test_all_to_all_multi_node[bfloat16-float8_e4m3fn] SKIPPED (Requires multi-node environment)
tests/test_all_to_all.py::test_all_to_all_multi_node[bfloat16-float16] SKIPPED (Requires multi-node environment)
tests/test_nvshmem.py::test_nvshmem_1_gpu NVSHMEM v3.2.5
PASSED
tests/test_nvshmem.py::test_nvshmem_4_gpu [Gloo] Rank 0 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 2 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 1 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 3 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
NVSHMEM v3.2.5
PASSED
tests/test_nvshmem.py::test_all_to_all [Gloo] Rank 3 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 1 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 0 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 2 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
NVSHMEM v3.2.5
PASSED
tests/test_nvshmem.py::test_all_to_all_multi_node SKIPPED (Requires multi-node environment)

=================================================================================== 27 passed, 7 skipped in 493.95s (0:08:13) ===================================================================================
```

## Benchmark
[单机H20性能数据](./20250909_050703_all_to_all.tsv)
