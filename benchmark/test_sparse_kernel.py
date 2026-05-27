import logging
from pathlib import Path

import pytest
import torch
import triton

from unum_ops.sparse_kernel_extension import (
    get_block_table_ref_torch,
    get_block_table_ref_triton,
    get_block_table_ref_triton_v2,
)

if torch.cuda.is_available():
    import sparse_kernel_extension

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
    elif hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    else:
        return torch.device("cpu")


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


def test_bench_get_table_triton(device):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "seq_len", "topk"],
            x_vals=[
                (1, 2048, 32),
                (4, 2048, 32),
                (16, 2048, 64),
                (64, 2048, 64),
                (256, 4096, 96),
                (256, 8192, 96),
            ],
            x_log=True,  # x axis is logarithmic.
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
            line_vals=["torch", "cuda", "cuda2", "cuda3", "triton", "triton_v2"],
            line_names=[
                "Torch(ms)",
                "CUDA v1(ms)",
                "CUDA v2(ms)",
                "CUDA v3(ms)",
                "Triton(ms)",
                "Triton v2(ms)",
            ],
            styles=[
                ("blue", "-"),
                ("green", "-"),
                ("red", "-"),
                ("yellow", "-"),
                ("orange", "-"),
                ("purple", "-"),
            ],
            ylabel="Latency (ms)",  # Label name for the y-axis.
            plot_name="Performance_get_block_table",  # Name for the plot. Used also as a file name for saving the plot.
            args={},  # Values for function arguments not in `x_names` and `y_name`.
        )
    )
    def benchmark(batch_size: int, seq_len: int, topk: int, provider):
        data = generate_data(batch_size, seq_len, topk, device, 0)
        topk_idx = data["topk_idx"]
        block_table = data["block_table"]
        token_to_bs = data["token_to_bs"]
        seqlen_q = data["seqlen_q"]

        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: get_block_table_ref_torch(
                    topk_idx, block_table, token_to_bs, seqlen_q, seqlen_q
                ),
                quantiles=quantiles,
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: get_block_table_ref_triton(
                    topk_idx, block_table, token_to_bs, seqlen_q, seqlen_q
                ),
                quantiles=quantiles,
            )
        elif provider == "triton_v2":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: get_block_table_ref_triton_v2(
                    topk_idx, block_table, token_to_bs, seqlen_q, seqlen_q
                ),
                quantiles=quantiles,
            )
        elif provider == "cuda":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: sparse_kernel_extension.get_block_table_v1(
                    topk_idx, block_table, token_to_bs, seqlen_q, seqlen_q, kSparseTopK
                ),
                quantiles=quantiles,
            )
        elif provider == "cuda2":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: sparse_kernel_extension.get_block_table_v2(
                    topk_idx, block_table, token_to_bs, seqlen_q, seqlen_q, kSparseTopK
                ),
                quantiles=quantiles,
            )
        elif provider == "cuda3":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: sparse_kernel_extension.get_block_table_v3(
                    topk_idx, block_table, token_to_bs, seqlen_q, seqlen_q, kSparseTopK
                ),
                quantiles=quantiles,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        return ms, min_ms, max_ms

    benchmark.run(print_data=True, show_plots=False, save_path=benchmark_output_dir)
