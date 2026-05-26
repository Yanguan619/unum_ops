import math

import torch
import torch.nn.functional as F


def round_multiple(x, m):
    return (x + m - 1) // m * m


def infllmv2_attn_stage1_ref_torch(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqlens_v=None,
    causal=False,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    # Support both layout:
    #   (n_heads, total_seqlen, head_dim)  — legacy torch layout
    #   (total_seqlen, n_heads, head_dim)  — cuda/triton layout
    total_seqlen_q = cu_seqlens_q[-1].item()
    if q.shape[0] == total_seqlen_q:
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
    batch_size = len(cu_seqlens_q) - 1
    max_seqlen_q = max(cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(batch_size))
    max_seqlen_k = max(cu_seqlens_k[i + 1] - cu_seqlens_k[i] for i in range(batch_size))

    # 对齐到 128（与 infllmv2_attn_stage1 保持一致）
    max_seqlen_k = round_multiple(max_seqlen_k, 128)

    # 创建 padded 张量
    q_padded = torch.zeros(
        q.shape[0],
        batch_size,
        max_seqlen_q,
        q.shape[-1],
        device=q.device,
        dtype=q.dtype,
    )
    k_padded = torch.zeros(
        k.shape[0],
        batch_size,
        max_seqlen_k,
        k.shape[-1],
        device=k.device,
        dtype=k.dtype,
    )
    # v_padded = torch.zeros(v.shape[0], batch_size, max_seqlen_k, v.shape[-1], device=v.device, dtype=v.dtype)

    # 填充数据
    for i in range(batch_size):
        q_start = cu_seqlens_q[i]
        q_end = cu_seqlens_q[i + 1]
        k_start = cu_seqlens_k[i]
        k_end = cu_seqlens_k[i + 1]
        q_padded[:, i, : q_end - q_start] = q[:, q_start:q_end]
        k_padded[:, i, : k_end - k_start] = k[:, k_start:k_end]
        # v_padded[:, i, :k_end-k_start] = v[:, k_start:k_end]

    # 计算 attention
    k_padded = k_padded.repeat_interleave(q_padded.shape[0] // k_padded.shape[0], dim=0)
    # v_padded = v_padded.repeat_interleave(q_padded.shape[0] // v_padded.shape[0], dim=0)

    scale = 1.0 / math.sqrt(q_padded.size(-1))
    attn = q_padded @ k_padded.transpose(-2, -1) * scale

    if causal:
        q_len = max_seqlen_q
        k_len = max_seqlen_k
        q_idx = torch.arange(q_len, device=attn.device)
        q_compress_idx = ((q_idx - 15) // 16) + k_len - (q_len - 16 + 1) // 16
        q_compress_idx = q_compress_idx.clamp(0, k_len)
        mask = [
            [0] * q_compress_idx[i] + [1] * (k_len - q_compress_idx[i])
            for i in range(q_len)
        ]
        # breakpoint()
        mask = torch.tensor(mask, dtype=torch.bool, device=attn.device)
        causal_mask = mask.expand(batch_size, q_len, k_len)
        # print(mask)
        # breakpoint()
        # score = S.masked_fill(mask, float('-inf'))
        # causal_mask = torch.zeros(batch_size, max_seqlen_q, max_seqlen_k, device=q.device).bool()
        # for b in range(batch_size):
        #     for i in range(max_seqlen_q):
        #         for j in range(max_seqlen_k):
        #             # i + 1 - 32 / 16 = j? (i + 1) / 16 - 1 = j i + 1 = 16 j + 16 ?
        #             if i >= (j * 16 + 32 - 1):
        #                 causal_mask[b, i, j] = True
        attn = attn.masked_fill(causal_mask, -float("inf"))

    score = F.softmax(attn, dim=-1)
    score = score.reshape(2, 16, batch_size, max_seqlen_q, max_seqlen_k).sum(dim=1)

    # 返回 padded 形式，形状为 (2, max_seqlen_q, max_seqlen_k)
    final_result = torch.zeros(
        2, max_seqlen_q, max_seqlen_k, device=q.device, dtype=q.dtype
    )
    for i in range(batch_size):
        q_start = cu_seqlens_q[i]
        q_end = cu_seqlens_q[i + 1]
        k_start = cu_seqlens_k[i]
        k_end = cu_seqlens_k[i + 1]
        q_len = q_end - q_start
        k_len = k_end - k_start
        final_result[:, :q_len, :k_len] = score[:, i, :q_len, :k_len]

    final_result = torch.where(torch.isnan(final_result), 0, final_result)
    return final_result
