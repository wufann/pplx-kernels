export NVSHMEM_HOME=/root/rocshmem/
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

# For single-node
export NVSHMEM_REMOTE_TRANSPORT=none
