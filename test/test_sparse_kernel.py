import logging
from pathlib import Path

import pytest
import torch

from unum_ops.sparse_kernel_extension import (
    get_block_table_ref_torch,
    get_block_table_ref_triton,
    get_block_table_ref_triton_v2,
    get_block_table_ref_triton_v3,
)

logging.basicConfig(level=logging.DEBUG)
torch.manual_seed(42)
# decode
kHeadGroup = 2
kSparseTopK = 96
kSparseBlockSize = 64
# token_num = 1
batch_size = 4
seqlen_q_max = 8192
topk_max_val = seqlen_q_max // kSparseBlockSize


benchmark_output_dir = Path("./benchmark/output/")
benchmark_output_dir.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.npu.is_available():
        return torch.device("npu:0")
    else:
        return torch.device("cpu")


def gen_topk(batch_size, topk_max_val, device):
    rand_scores = torch.rand((kHeadGroup, batch_size, topk_max_val))
    indices = rand_scores.topk(kSparseTopK, dim=-1, largest=False).indices
    topk_idx = indices.sort(dim=-1).values.to(dtype=torch.int32).to(device=device)
    return topk_idx


@pytest.fixture(scope="module")
def test_data(device):
    topk_idx = gen_topk(batch_size, topk_max_val, device)
    block_table = torch.tensor(
        [_ for _ in range(1, seqlen_q_max * batch_size + 1)],
        dtype=torch.int32,
        device=device,
    ).reshape(batch_size, seqlen_q_max)
    token_to_bs = torch.arange(0, batch_size, dtype=torch.int32, device=device)
    seqlen_q = torch.full((batch_size,), seqlen_q_max, dtype=torch.int32, device=device)

    return {
        "topk_idx": topk_idx,
        "block_table": block_table,
        "token_to_bs": token_to_bs,
        "seqlen_q": seqlen_q,
    }


def generate_data(
    batch_size_val: int,
    seqlen_val: int,
    topk_val: int,
    device: torch.device,
    seed_offset=0,
):
    topk_max_val = seqlen_val // kSparseBlockSize
    actual_topk = min(topk_val, topk_max_val)

    local_seed = (42 + seed_offset + batch_size_val + seqlen_val + topk_val) % 1000000
    gen = torch.Generator(device="cpu")
    gen.manual_seed(local_seed)

    rand_scores = torch.rand((kHeadGroup, batch_size_val, topk_max_val), generator=gen)
    indices = rand_scores.topk(actual_topk, dim=-1, largest=False).indices
    topk_idx_sorted = indices.sort(dim=-1).values

    topk_idx = torch.zeros(
        (kHeadGroup, batch_size_val, kSparseTopK), dtype=torch.int32, device=device
    )
    topk_idx[:, :, :actual_topk] = topk_idx_sorted.to(device=device)

    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(local_seed + 10000)
    block_table = torch.randint(
        1, 10000, (batch_size_val, seqlen_val), dtype=torch.int32, generator=gen2
    ).to(device=device)

    token_to_bs = torch.arange(0, batch_size_val, dtype=torch.int32, device=device)

    max_blocks = seqlen_val // kSparseBlockSize
    gen3 = torch.Generator(device="cpu")
    gen3.manual_seed(local_seed + 20000)
    seqlen_q_blocks = torch.randint(
        max_blocks // 4,
        max_blocks + 1,
        (batch_size_val,),
        dtype=torch.int32,
        generator=gen3,
    )
    seqlen_q = seqlen_q_blocks * kSparseBlockSize

    # 打印shape或len
    logging.debug("\n")
    logging.debug(f"{topk_idx.shape=}")
    logging.debug(f"{block_table.shape=}")
    logging.debug(f"{token_to_bs.shape=}")
    logging.debug(f"{seqlen_q.shape=}")
    logging.debug(f"{topk_val=}")

    return {
        "topk_idx": topk_idx,
        "block_table": block_table,
        "token_to_bs": token_to_bs,
        "seqlen_q": seqlen_q.to(device=device),
        "batch_size": batch_size_val,
        "seqlen_val": seqlen_val,
        "topk_val": kSparseTopK,
    }


def ops_call():
    ops = [
        get_block_table_ref_torch,
        get_block_table_ref_triton,
        get_block_table_ref_triton_v2,
        get_block_table_ref_triton_v3,
    ]
    if torch.cuda.is_available():
        import sparse_kernel_extension

        ops.append(sparse_kernel_extension.get_block_table_v2)
        ops.append(sparse_kernel_extension.get_block_table_v3)

    return ops


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_get_block_table_v1(test_data):
    import sparse_kernel_extension

    out_block_table = sparse_kernel_extension.get_block_table_v1(
        test_data["topk_idx"],
        test_data["block_table"],
        test_data["token_to_bs"],
        test_data["seqlen_q"],
        test_data["seqlen_q"],
        kSparseTopK,
    )
    assert out_block_table.device.type == "cuda"
    assert out_block_table.shape is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
@pytest.mark.parametrize("ops_call", ops_call())
@pytest.mark.parametrize(
    "batch, seq_len, topk",
    [
        (1, 256, 4),
        (2, 512, 8),
        (4, 1024, 16),
        (8, 2048, 32),
        (4, 4096, 64),
        (2, 8192, 96),
    ],
)
def test_get_block_table_matches_v1(batch, seq_len, topk, device, ops_call):
    import sparse_kernel_extension

    data = generate_data(batch, seq_len, topk, device, 0)
    out_v1 = sparse_kernel_extension.get_block_table_v1(
        data["topk_idx"],
        data["block_table"],
        data["token_to_bs"],
        data["seqlen_q"],
        data["seqlen_q"],
        data["topk_val"],
    )
    out_triton = ops_call(
        data["topk_idx"],
        data["block_table"],
        data["token_to_bs"],
        data["seqlen_q"],
        data["seqlen_q"],
        data["topk_val"],
    )
    assert out_triton is not None, "Triton output is None"

    assert torch.allclose(out_v1, out_triton)
