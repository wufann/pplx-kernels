cd /opt/rocshmem_dependencies
git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x
cd ompi
./autogen.pl
./configure --prefix=--prefix=/opt/rocshmem_dependencies/ompi --with-rocm=/opt/rocm --with-ucx=/opt/rocshmem_dependencies/ucx
make -j 8
make -j 8 install