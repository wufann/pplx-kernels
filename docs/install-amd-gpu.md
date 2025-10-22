# Build pplx-kernel for AMD gpu
## start container
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
        rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.7.1 \
        /bin/bash
else
docker start $MY_CONTAINER
docker exec -ti $MY_CONTAINER /bin/bash
fi
```

## Install dependencies
```
apt-get update -y
apt install -y libopenmpi-dev flex bison

# Install ucx and ompi
cd /opt
git clone https://github.com/ROCm/rocSHMEM.git
export BUILD_DIR=$PWD
./rocSHMEM/scripts/install_dependencies.sh

# Add PATH
export PATH=/opt/install/ompi/bin:$PATH
export LD_LIBRARY_PATH=/opt/install/ucx/lib:/opt/install/ompi/lib:$LD_LIBRARY_PATH


# Build rocSHMEM library, library will be installed in $HOME/rocshmem
# Three banckends for rocSHMEM: IPC, Reverse Offload (RO), and GDA.
# IPC backend as example:
apt install -y cmake // if need
cd rocSHMEM
mkdir build.ipc && cd build.ipc
MPI_ROOT=/opt/install/ompi UCX_ROOT=/opt/install/ucx CMAKE_PREFIX_PATH="/opt/rocm:$CMAKE_PREFIX_PATH" ../scripts/build_configs/ipc_single /opt/install/ 
```