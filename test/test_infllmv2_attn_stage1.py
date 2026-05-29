import pytest
import torch
from infllm_v2 import infllmv2_attn_stage1

from unum_ops.infllm_v2.infllmv2_attn_stage1_triton import infllmv2_attn_stage1_triton


def generate_data(
    seqlen_q,
    seqlen_k,
    n_heads=32,
    n_kv_heads=2,
    head_dim=128,
    dtype=torch.bfloat16,
    device="cuda",
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
@pytest.mark.parametrize("seqlen_k", [15, 16, 128], ids=lambda v: f"seqlen_k={v}")
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
    q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = generate_data(
        seqlen_q,
        seqlen_k,
        n_heads,
        n_kv_heads,
        head_dim,
        dtype,
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

    if valid_q_count > 0:
        valid_mask = torch.zeros_like(cuda_out, dtype=torch.bool)
        valid_mask[:, :valid_q_count, :] = True
        mask = valid_mask & cuda_mask
    else:
        mask = cuda_mask

    print(f"  CUDA valid non-zero: {mask.sum().item()}/{cuda_mask.numel()}")

    assert torch.allclose(cuda_out.float()[mask], triton_out.float()[mask])
