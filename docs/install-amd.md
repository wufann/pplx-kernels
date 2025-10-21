# Build pplx-kernel for AMD GPU

## Start container
```
#/bin/bash
export MY_CONTAINER="fanwu103-pplx"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
docker run -e  DISPLAY=$DISPLAY --net=host --pid=host --ipc=host \
        --shm-size 64g \
        --privileged \
        -it \
        -v /tools/:/tools/ \
        -v /mnt/:/mnt/ \
        -v /models/:/models/ \
        -v /home/:/home/ \
        --name $MY_CONTAINER  \
        rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta \
        /bin/bash
else
docker start $MY_CONTAINER
docker exec -ti $MY_CONTAINER /bin/bash
fi
```
## Install dependencies
```
apt-get update -y
apt install -y libopenmpi-dev

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

```
## souce env.sh
```
export NVSHMEM_HOME=/root/rocshmem/
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

# For single-node
export NVSHMEM_REMOTE_TRANSPORT=none
```

## build pplx-kernels
```
PYTORCH_ROCM_ARCH=gfx950 python3 setup.py bdist_wheel
pip install dist/*.whl
```