import torch
import triton
import triton.language as tl


def round_multiple(x, m):
    return (x + m - 1) // m * m


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 256}, num_warps=8, num_stages=2),
    ],
    key=["max_seqlen_k"],
)
@triton.jit
def _infllmv2_attn_stage1_kernel(
    q_ptr,
    k_ptr,
    out_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_out_h,
    stride_out_q,
    stride_out_k,
    total_q,
    num_batches,
    max_seqlen_k,
    HEAD_DIM: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    N_HEADS_PER_GROUP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    causal: tl.constexpr,
    scale_f32,
):
    pid = tl.program_id(0)
    head_group = pid // total_q
    q_pos = pid % total_q

    k_start = 0
    k_len = 0
    local_q_pos = 0
    q_len = 0

    for b in range(num_batches):
        qs = tl.load(cu_seqlens_q_ptr + b)
        qe = tl.load(cu_seqlens_q_ptr + b + 1)
        ks = tl.load(cu_seqlens_k_ptr + b)
        ke = tl.load(cu_seqlens_k_ptr + b + 1)
        hit = (q_pos >= qs) & (q_pos < qe)
        local_q_pos = tl.where(hit, q_pos - qs, local_q_pos)
        k_start = tl.where(hit, ks, k_start)
        k_len = tl.where(hit, ke - ks, k_len)
        q_len = tl.where(hit, qe - qs, q_len)

    if k_len <= 0:
        return

    if causal:
        q_compress_off = ((local_q_pos - 15) // 16) + k_len - (q_len - 16 + 1) // 16
        q_compress_off = tl.where(q_compress_off < 0, 0, q_compress_off)
        q_compress_off = tl.where(q_compress_off > k_len, k_len, q_compress_off)
    else:
        q_compress_off = 0

    n_blocks = tl.cdiv(k_len, BLOCK_K)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_g = tl.arange(0, N_HEADS_PER_GROUP)

    q_group_bf16 = tl.load(
        q_ptr
        + q_pos * stride_q_t
        + (head_group * N_HEADS_PER_GROUP + offs_g[:, None]) * stride_q_h
        + offs_d[None, :] * stride_q_d,
    ).to(tl.bfloat16)

    m_all = tl.full([N_HEADS_PER_GROUP], float("-inf"), dtype=tl.float32)
    se_all = tl.zeros([N_HEADS_PER_GROUP], dtype=tl.float32)

    for bi in range(n_blocks):
        ks_off = bi * BLOCK_K
        offs_k = ks_off + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_len

        k_tile_bf16 = tl.load(
            k_ptr
            + (k_start + offs_k[:, None]) * stride_k_t
            + head_group * stride_k_h
            + offs_d[None, :] * stride_k_d,
            mask=k_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.bfloat16)

        scores_2d = (
            tl.dot(q_group_bf16, tl.trans(k_tile_bf16)).to(tl.float32) * scale_f32
        )
        scores_2d = tl.where(k_mask[None, :], scores_2d, float("-inf"))

        if causal:
            causal_mask = (ks_off + tl.arange(0, BLOCK_K)) < q_compress_off
            scores_2d = tl.where(causal_mask[None, :], scores_2d, float("-inf"))

        block_max = tl.max(scores_2d, axis=1)
        new_m = tl.maximum(m_all, block_max)
        exp_scores = tl.exp(scores_2d - new_m[:, None])
        block_sum = tl.sum(exp_scores, axis=1)
        new_se = se_all * tl.exp(m_all - new_m) + block_sum

        m_all = new_m
        se_all = new_se

    for bi in range(n_blocks):
        ks_off = bi * BLOCK_K
        offs_k = ks_off + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_len

        k_tile_bf16 = tl.load(
            k_ptr
            + (k_start + offs_k[:, None]) * stride_k_t
            + head_group * stride_k_h
            + offs_d[None, :] * stride_k_d,
            mask=k_mask[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)

        scores_2d = (
            tl.dot(q_group_bf16, tl.trans(k_tile_bf16)).to(tl.float32) * scale_f32
        )
        scores_2d = tl.where(k_mask[None, :], scores_2d, float("-inf"))

        if causal:
            causal_mask = (ks_off + tl.arange(0, BLOCK_K)) < q_compress_off
            scores_2d = tl.where(causal_mask[None, :], scores_2d, float("-inf"))

        softmax_all = tl.exp(scores_2d - m_all[:, None]) / se_all[:, None]
        block_out = tl.sum(softmax_all, axis=0)

        tl.store(
            out_ptr
            + head_group * stride_out_h
            + q_pos * stride_out_q
            + offs_k * stride_out_k,
            block_out,
            mask=k_mask,
        )


def infllmv2_attn_stage1_triton_v2(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqlens_v=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=True,
    block_table=None,
):
    q, k, v = [
        x.contiguous() if x is not None and x.stride(-1) != 1 else x for x in (q, k, v)
    ]

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    total_q, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    n_heads_per_group = n_heads // n_kv_heads
    batch_size = cu_seqlens_q.numel() - 1

    if max_seqlen_k is None or max_seqlen_k == 0:
        max_seqlen_k = max(
            int(cu_seqlens_k[i + 1] - cu_seqlens_k[i]) for i in range(batch_size)
        )

    round_block_k = 128
    max_k_rounded = (
        round_multiple(max_seqlen_k, round_block_k) if max_seqlen_k > 0 else 0
    )

    out = torch.zeros(
        n_kv_heads,
        total_q,
        max_k_rounded,
        device=q.device,
        dtype=torch.bfloat16,
    )

    if max_k_rounded == 0 or total_q == 0:
        return out

    grid = (n_kv_heads * total_q,)

    _infllmv2_attn_stage1_kernel[grid](
        q,
        k,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        total_q,
        batch_size,
        max_seqlen_k,
        HEAD_DIM=head_dim,
        N_HEADS=n_heads,
        N_KV_HEADS=n_kv_heads,
        N_HEADS_PER_GROUP=n_heads_per_group,
        causal=causal,
        scale_f32=softmax_scale,
    )

    out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
    return out
