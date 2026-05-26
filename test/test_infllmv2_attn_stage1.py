import pytest
import torch
from infllm_v2 import infllmv2_attn_stage1

from unum_ops.infllm_v2 import infllmv2_attn_stage1_ref_torch
from unum_ops.infllm_v2.infllmv2_attn_stage1_triton import infllmv2_attn_stage1_triton


def data(
    seqlen_q,
    seqlen_k,
    n_heads=32,
    n_kv_heads=2,
    head_dim=128,
    dtype=torch.bfloat16,
    device="cuda",
    causal=True,
    batch_size=1,
):
    seqlen_qs = [seqlen_q]
    seqlen_ks = [seqlen_k]
    total_seqlen_q = sum(seqlen_qs)
    total_seqlen_k = sum(seqlen_ks)

    q = torch.randn(n_heads, total_seqlen_q, head_dim, dtype=dtype, device=device)
    k = torch.randn(n_kv_heads, total_seqlen_k, head_dim, dtype=dtype, device=device)

    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i in range(batch_size):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seqlen_qs[i]
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seqlen_ks[i]

    return q, k, cu_seqlens_q, cu_seqlens_k, max(seqlen_qs), max(seqlen_ks)


@pytest.mark.parametrize("seqlen_q", [64, 256], ids=lambda v: f"seqlen_q={v}")
@pytest.mark.parametrize("seqlen_k", [15, 16, 129], ids=lambda v: f"seqlen_k={v}")
def test_infllmv2_attn_stage1_cuda(
    seqlen_q,
    seqlen_k,
    n_heads=32,
    n_kv_heads=2,
    head_dim=128,
    dtype=torch.bfloat16,
    causal=True,
    batch_size=1,
):
    q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = data(
        seqlen_q,
        seqlen_k,
        n_heads,
        n_kv_heads,
        head_dim,
        dtype,
        causal=causal,
        batch_size=batch_size,
    )

    naive_score = infllmv2_attn_stage1_ref_torch(
        q, k, k, cu_seqlens_q, cu_seqlens_k, causal=causal
    )

    q = q.transpose(0, 1).contiguous().clone()
    k = k.transpose(0, 1).contiguous().clone()

    flash_score = infllmv2_attn_stage1(
        q,
        k,
        k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_v=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
    )
    assert flash_score.shape == naive_score.shape


@pytest.mark.parametrize("seqlen_q", [64, 256], ids=lambda v: f"seqlen_q={v}")
@pytest.mark.parametrize("seqlen_k", [128, 256], ids=lambda v: f"seqlen_k={v}")
def test_infllmv2_attn_stage1_triton_vs_cuda(
    seqlen_q,
    seqlen_k,
    n_heads=32,
    n_kv_heads=2,
    head_dim=128,
    dtype=torch.bfloat16,
    batch_size=1,
):
    """
    Compare Triton vs CUDA for causal=False.

    NOTE: The CUDA kernel has two known limitations:
    1. (cu_seqlens_q bug) Only the first total_q/n_heads_per_group q positions
       produce valid output. Positions beyond that are silently zero.
    2. (causal mask) The CUDA kernel's causal mask differs from the InfLLMv2
       reference, so causal=True comparisons are not meaningful.
    """
    q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = data(
        seqlen_q,
        seqlen_k,
        n_heads,
        n_kv_heads,
        head_dim,
        dtype,
        causal=False,
        batch_size=batch_size,
    )

    q_contig = q.transpose(0, 1).contiguous().clone()
    k_contig = k.transpose(0, 1).contiguous().clone()

    cuda_out = infllmv2_attn_stage1(
        q_contig,
        k_contig,
        k_contig,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_v=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False,
    )

    triton_out = infllmv2_attn_stage1_triton(
        q_contig,
        k_contig,
        k_contig,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_v=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False,
    )

    assert (
        cuda_out.shape == triton_out.shape
    ), f"Shape mismatch: {cuda_out.shape} vs {triton_out.shape}"
    assert triton_out.isnan().sum().item() == 0, "Triton output contains NaN"
    # CUDA only produces output for first total_q/n_heads_per_group positions
    n_heads_per_group = 32 // 2
    valid_q_count = seqlen_q // n_heads_per_group
    cuda_mask = cuda_out != 0
    diff = (cuda_out.float() - triton_out).abs()

    if valid_q_count > 0:
        valid_mask = torch.zeros_like(cuda_out, dtype=torch.bool)
        valid_mask[:, :valid_q_count, :] = True
        nz_diff = diff[valid_mask & cuda_mask]
    else:
        nz_diff = diff[cuda_mask]

    max_err = nz_diff.max().item() if nz_diff.numel() > 0 else 0.0
    mean_err = nz_diff.mean().item() if nz_diff.numel() > 0 else 0.0

    print(f"  CUDA valid non-zero: {nz_diff.numel()}/{cuda_mask.numel()}")
    print(f"  At valid positions: max_diff={max_err:.6f} mean_diff={mean_err:.6f}")

    # bf16 precision: ~0.8% relative error per softmax computation.
    # With 16 summed heads at each position, max error < 0.01 is expected.
    assert max_err < 0.02, f"Max diff {max_err} exceeds bf16 precision tolerance"


@pytest.mark.parametrize("seqlen_q", [64, 128, 256, 512], ids=lambda v: f"seqlen_q={v}")
@pytest.mark.parametrize("seqlen_k", [128, 256, 512], ids=lambda v: f"seqlen_k={v}")
def test_infllmv2_attn_stage1_precision_comparison(
    seqlen_q,
    seqlen_k,
    n_heads=32,
    n_kv_heads=2,
    head_dim=128,
    dtype=torch.bfloat16,
    batch_size=1,
):
    """
    Compare both torch and triton against CUDA to determine which is more precise.
    """
    q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = data(
        seqlen_q,
        seqlen_k,
        n_heads,
        n_kv_heads,
        head_dim,
        dtype,
        causal=False,
        batch_size=batch_size,
    )

    q_contig = q.transpose(0, 1).contiguous().clone()
    k_contig = k.transpose(0, 1).contiguous().clone()

    cuda_out = infllmv2_attn_stage1(
        q_contig,
        k_contig,
        k_contig,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_v=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False,
    )

    torch_out = infllmv2_attn_stage1_ref_torch(
        q,
        k,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        causal=False,
    )

    triton_out = infllmv2_attn_stage1_triton(
        q_contig,
        k_contig,
        k_contig,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_v=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False,
    )

    n_heads_per_group = 32 // 2
    valid_q_count = seqlen_q // n_heads_per_group
    cuda_mask = cuda_out != 0

    torch_diff = (cuda_out.float() - torch_out.float()).abs()
    triton_diff = (cuda_out.float() - triton_out.float()).abs()

    if valid_q_count > 0:
        valid_mask = torch.zeros_like(cuda_out, dtype=torch.bool)
        valid_mask[:, :valid_q_count, :] = True
        mask = valid_mask & cuda_mask
        torch_nz = torch_diff[mask]
        triton_nz = triton_diff[mask]
    else:
        torch_nz = torch_diff[cuda_mask]
        triton_nz = triton_diff[cuda_mask]

    t_max = torch_nz.max().item() if torch_nz.numel() > 0 else 0.0
    tr_max = triton_nz.max().item() if triton_nz.numel() > 0 else 0.0
    t_mean = torch_nz.mean().item() if torch_nz.numel() > 0 else 0.0
    tr_mean = triton_nz.mean().item() if triton_nz.numel() > 0 else 0.0

    print(
        f"\n    torch  vs CUDA: max_diff={t_max:.6f} mean_diff={t_mean:.6f}"
        f"\n    triton vs CUDA: max_diff={tr_max:.6f} mean_diff={tr_mean:.6f}"
    )

    assert (
        max(t_max, tr_max) < 0.02
    ), f"Max diff exceeds bf16 precision tolerance: torch={t_max:.6f} triton={tr_max:.6f}"
