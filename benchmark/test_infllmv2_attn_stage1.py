from pathlib import Path

import torch
import triton
from infllm_v2 import infllmv2_attn_stage1

from unum_ops.infllm_v2 import infllmv2_attn_stage1_ref_torch, infllmv2_attn_stage1_triton

benchmark_output_dir = Path("./benchmark/output/")
benchmark_output_dir.mkdir(parents=True, exist_ok=True)


def test_infllmv2_attn_stage1(
    head_dim=128, n_heads=32, n_kv_heads=2, dtype=torch.bfloat16, device="cuda"
) -> None:
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seqlen_q", "seqlen_k"],
            x_vals=[
                (64, 128),
                (64, 256),
                (64, 512),
                (64, 1024),
                (128, 128),
                (256, 256),
                (512, 512),
                (1024, 1024),
            ],
            x_log=True,  # x axis is logarithmic.
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
            line_vals=["torch", "cuda", "triton"],
            line_names=[
                "Torch(ms)",
                "CUDA(ms)",
                "Triton(ms)",
            ],
            styles=[
                ("green", "-"),
                ("blue", "-"),
                ("red", "-"),
            ],
            ylabel="Latency (ms)",  # Label name for the y-axis.
            plot_name="Performance_infllmv2_attn_stage1",  # Name for the plot. Used also as a file name for saving the plot.
            args={},  # Values for function arguments not in `x_names` and `y_name`.
        )
    )
    def benchmark(seqlen_q, seqlen_k, provider):
        q = torch.randn(seqlen_q, n_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(seqlen_k, n_kv_heads, head_dim, dtype=dtype, device=device)
        cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)
        q = q.clone()
        k = k.clone()
        v = k.clone()
        call_ops = {
            "torch": infllmv2_attn_stage1_ref_torch,
            "cuda": infllmv2_attn_stage1,
            "triton": infllmv2_attn_stage1_triton,
        }
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: call_ops[provider](
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                cu_seqlens_v=cu_seqlens_k,
                max_seqlen_q=seqlen_q,
                max_seqlen_k=seqlen_k,
                causal=False,
            ),
            quantiles=quantiles,
        )

        return ms, min_ms, max_ms

    benchmark.run(print_data=True, show_plots=False, save_path=benchmark_output_dir)
