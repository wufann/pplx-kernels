cd /opt/rocshmem_dependencies
git clone https://github.com/openucx/ucx.git -b v1.17.x
cd ucx
./autogen.sh
./configure --prefix=/opt/rocshmem_dependencies/ucx --with-rocm=/opt/rocm --enable-mt
make -j 8
make -j 8 install