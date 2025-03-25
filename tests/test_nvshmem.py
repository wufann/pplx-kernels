import pytest
import torch

from pplx_kernels.nvshmem import (
    nvshmem_alloc_empty_unique_id,
    nvshmem_alltoall,
    nvshmem_barrier_all_on_current_stream,
    nvshmem_finalize,
    nvshmem_get_unique_id,
    nvshmem_init,
    nvshmem_malloc,
    nvshmem_my_pe,
    nvshmem_n_pes,
)

from .distributed_utils import (
    ProcessGroupInfo,
    parallel_launch,
    parallel_launch_from_env,
    require_multi_node,
)


def test_nvshmem_1_gpu() -> None:
    uid = nvshmem_get_unique_id()
    nvshmem_init(uid, 0, 1)
    assert nvshmem_my_pe() == 0
    assert nvshmem_n_pes() == 1
    nvshmem_finalize()


def _worker_test_nvshmem_4_gpu(pgi: ProcessGroupInfo) -> None:
    uid = nvshmem_get_unique_id() if pgi.rank == 0 else nvshmem_alloc_empty_unique_id()
    torch.distributed.broadcast(uid, src=0)
    nvshmem_init(uid, pgi.rank, pgi.world_size)
    assert nvshmem_my_pe() == pgi.rank
    assert nvshmem_n_pes() == pgi.world_size
    nvshmem_finalize()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_nvshmem_4_gpu() -> None:
    parallel_launch(4, _worker_test_nvshmem_4_gpu)


def _worker_test_all_to_all(pgi: ProcessGroupInfo) -> None:
    uid = nvshmem_get_unique_id() if pgi.rank == 0 else nvshmem_alloc_empty_unique_id()
    torch.distributed.broadcast(uid, src=0)
    nvshmem_init(uid, pgi.rank, pgi.world_size)
    try:
        t_in = nvshmem_malloc([pgi.world_size], dtype=torch.int32, device=pgi.device)
        t_in.copy_(
            torch.full([pgi.world_size], pgi.rank, dtype=torch.int32, device=pgi.device)
        )

        t_out = nvshmem_malloc([pgi.world_size], dtype=torch.int32, device=pgi.device)

        nvshmem_alltoall(t_out, t_in)
        nvshmem_barrier_all_on_current_stream()
        torch.cuda.synchronize()

        assert t_out.tolist() == list(range(pgi.world_size))
    finally:
        del t_in
        del t_out
        nvshmem_finalize()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_all_to_all() -> None:
    parallel_launch(4, _worker_test_all_to_all)


@require_multi_node
def test_all_to_all_multi_node() -> None:
    parallel_launch_from_env(_worker_test_all_to_all)
