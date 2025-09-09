# #!/bin/bash
# # BUILD_DIR=/opt/dependencies ./build.sh

# set -e
# set -o pipefail

# if [[ -z "${_ROCM_DIR}" ]]; then
#   export _ROCM_DIR=/opt/rocm
# fi

# # Location of dependencies source code
# export _INSTALL_DIR=$BUILD_DIR
# export _DEPS_SRC_DIR=$_INSTALL_DIR

# mkdir -p $_DEPS_SRC_DIR

# #Adjust branches and installation location as necessary
# export _UCX_INSTALL_DIR=$_INSTALL_DIR/ucx
# export _UCX_REPO=https://github.com/openucx/ucx.git
# export _UCX_BRANCH=v1.17.x

# export _OMPI_INSTALL_DIR=$_INSTALL_DIR/ompi
# # export _OMPI_REPO=https://github.com/open-mpi/ompi.git
# # export _OMPI_BRANCH=v5.0.x

# # Step 1: Build UCX with ROCm support
# cd $_DEPS_SRC_DIR
# rm -rf ucx
# git clone $_UCX_REPO
# cd ucx
# git checkout -b $_UCX_BRANCH
# ./autogen.sh
# ./configure --prefix=$_UCX_INSTALL_DIR \
#             --with-rocm=$_ROCM_DIR     \
#             --enable-mt
# make -j 8
# make -j 8 install/src


# # Step 2: Install OpenMPI with UCX support
# cd $_DEPS_SRC_DIR
# # git clone --recursive $_OMPI_REPO
# wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.gz
# tar zxvf openmpi-5.0.7.tar.gz
# cd openmpi-5.0.7
# mkdir build
# cd build
# ../configure --prefix=$_OMPI_INSTALL_DIR  \
#             --with-rocm=$_ROCM_DIR       \
#             --with-ucx=$_UCX_INSTALL_DIR
# make -j 8
# make -j 8 install

# # rm -rf $_DEPS_SRC_DIR

# echo "Dependencies for rocSHMEM are now installed"
# echo ""
# echo "UCX ($_UCX_COMMIT_HASH) Installed to $_UCX_INSTALL_DIR"
# echo "OpenMPI ($_OMPI_COMMIT_HASH) Installed to $_OMPI_INSTALL_DIR"
# echo ""
# echo "Please update your PATH and LD_LIBRARY_PATH"


export INSTALL_DIR=/opt/ompi_for_gpu
export BUILD_DIR=/tmp/ompi_for_gpu_build
mkdir -p $BUILD_DIR

export UCX_DIR=$INSTALL_DIR/ucx
cd $BUILD_DIR
git clone https://github.com/openucx/ucx.git -b v1.15.x
cd ucx
./autogen.sh
mkdir build
cd build
../configure -prefix=$UCX_DIR \
    --with-rocm=/opt/rocm
make -j $(nproc)
make -j $(nproc) install

export OMPI_DIR=$INSTALL_DIR/ompi
cd $BUILD_DIR
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.gz
tar zxvf openmpi-5.0.7.tar.gz
cd openmpi-5.0.7
mkdir build
cd build
../configure --prefix=$OMPI_DIR --with-ucx=$UCX_DIR \
    --with-rocm=/opt/rocm
make -j $(nproc)
make install