export NVSHMEM_HOME=/opt/nvshmem-3.2.5
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

# For single-node
export NVSHMEM_REMOTE_TRANSPORT=none