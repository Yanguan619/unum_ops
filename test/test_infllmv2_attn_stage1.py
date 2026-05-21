import pytest
import torch
from infllm_v2 import infllmv2_attn_stage1

from unum_ops.infllm_v2 import infllmv2_attn_stage1_ref_torch


@pytest.mark.parametrize("seqlen_q", [64, 256])
@pytest.mark.parametrize("seqlen_k", [15, 16, 129])
def test_flash_attn_varlen(
    seqlen_q,
    seqlen_k,
    n_heads=32,
    n_kv_heads=2,
    head_dim=128,
    dtype=torch.bfloat16,
    bench=False,
    causal=True,
    batch_size=1,
):
    # 生成不同长度的序列
    seqlen_qs = [seqlen_q]  # 两个序列，长度不同
    seqlen_ks = [seqlen_k]  # k 也使用不同长度
    total_seqlen_q = sum(seqlen_qs)
    total_seqlen_k = sum(seqlen_ks)

    # 准备输入数据
    q = torch.randn(n_heads, total_seqlen_q, head_dim, dtype=dtype).cuda()
    k = torch.randn(n_kv_heads, total_seqlen_k, head_dim, dtype=dtype).cuda()
    v = torch.randn(n_kv_heads, total_seqlen_k, head_dim, dtype=dtype).cuda()

    # 计算累积序列长度
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seqlen_qs[i]
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seqlen_ks[i]

    # 朴素实现
    naive_score = infllmv2_attn_stage1_ref_torch(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        causal=causal,
    )

    q = q.transpose(0, 1).contiguous().clone()
    k = k.transpose(0, 1).contiguous().clone()
    v = v.transpose(0, 1).contiguous().clone()

    flash_score = infllmv2_attn_stage1(
        q,
        k,
        k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_v=cu_seqlens_k,
        max_seqlen_q=max(seqlen_qs),
        max_seqlen_k=max(seqlen_ks),
        causal=causal,
    )
    flash_score = flash_score.to(torch.float32)
    assert flash_score.shape == naive_score.shape

    print(f"{seqlen_q=} {seqlen_k=} {causal=}")
    diff = (naive_score - flash_score).abs()
    max_val = diff.max()
    max_idx = (diff == max_val).nonzero()
    print("score max diff:", max_val.item())
    if max_idx.numel() > 0:
        # print a few indices (head, q_idx, k_idx)
        print("max diff indices (up to 10):", max_idx[:10].tolist())
    # print some nonzero diff coordinates for inspection
    nz = (diff > 0).nonzero()
    if nz.numel() > 0:
        print("nonzero diff count:", nz.shape[0])
        print("sample nonzero diff indices (up to 20):", nz[:20].tolist())
    # print("online score max diff :", (online_score - flash_score).abs().max())
    # breakpoint()
    if (naive_score - flash_score).abs().max() > 1e-2:
        print(f"error: seqlen_qs={seqlen_qs}, seqlen_ks={seqlen_ks}")
