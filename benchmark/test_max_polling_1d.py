from pathlib import Path

import torch
import triton
from infllm_v2.max_pooling_1d import max_pooling_1d_varlen

from unum_ops.infllm_v2 import max_pooling_1d_varlen_ref_triton

benchmark_output_dir = Path("./benchmark/output/")
benchmark_output_dir.mkdir(parents=True, exist_ok=True)


def test_max_pooling_1d_varlen(
    num_heads=8,
    kernel_size=32,
    kernel_stride=16,
    block_size=64,
    init_blocks=1,
    local_blocks=32,
    dtype=torch.bfloat16,
    device="cuda",
) -> None:
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seqlen"],
            x_vals=[64, 128, 256, 512, 1024, 2048, 4096],
            x_log=True,
            line_arg="provider",
            line_vals=["cuda", "triton"],
            line_names=["CUDA(ms)", "Triton(ms)"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="Latency (ms)",
            plot_name="Performance_max_pooling_1d",
            args={},
        )
    )
    def benchmark(seqlen, provider):
        batch_size = 1
        max_seqlen_q = seqlen
        max_seqlen_k = seqlen * 2
        cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, seqlen * 2], dtype=torch.int32, device=device)
        attn_score_full = torch.randn(num_heads, seqlen, max_seqlen_k, device=device, dtype=dtype)
        cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "cuda":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: max_pooling_1d_varlen(
                    attn_score_full,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    cache_lens,
                    max_seqlen_q,
                    max_context_len=max_seqlen_q,
                    local_blocks=local_blocks,
                    init_blocks=init_blocks,
                    block_size=block_size,
                    stride=kernel_stride,
                ),
                quantiles=quantiles,
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: max_pooling_1d_varlen_ref_triton(
                    attn_score_full,
                    kernel_size,
                    kernel_stride,
                    block_size,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    init_blocks=init_blocks,
                    local_blocks=local_blocks,
                ),
                quantiles=quantiles,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        return ms, min_ms, max_ms

    benchmark.run(print_data=True, show_plots=False, save_path=benchmark_output_dir)
