import math

import torch
import triton
import triton.language as tl


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


@triton.jit
def _max_pooling_1d_varlen_kernel(
    score_ptr,
    block_score_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    total_q,
    flat_stride,
    num_heads,
    max_blocks,
    BLOCK_SIZE: tl.constexpr,
    STRIDE_POOL: tl.constexpr,
    KSIZE: tl.constexpr,
    KSIZE_POW2: tl.constexpr,
    PADDING: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    batch_size,
):
    pid = tl.program_id(0)
    head = pid // max_blocks
    bk = pid % max_blocks

    offs_q = tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, KSIZE_POW2)

    for q_start in range(0, total_q, BLOCK_Q):
        q_abs = q_start + offs_q
        q_mask = q_abs < total_q

        k_len = tl.zeros([BLOCK_Q], dtype=tl.int32)
        off_bq = tl.zeros([BLOCK_Q], dtype=tl.int32)
        for b in range(batch_size):
            qs = tl.load(cu_seqlens_q_ptr + b)
            qe = tl.load(cu_seqlens_q_ptr + b + 1)
            ks = tl.load(cu_seqlens_k_ptr + b)
            ke = tl.load(cu_seqlens_k_ptr + b + 1)
            hit = (q_abs >= qs) & (q_abs < qe)
            k_len = tl.where(hit, ke - ks, k_len)
            off_bq = tl.where(hit, (q_abs - qs) // BLOCK_SIZE, off_bq)

        base = head * total_q * flat_stride + q_abs * flat_stride

        should_mask_inf = (bk < INIT_BLOCKS) | (
            (off_bq >= bk) & (off_bq <= bk + LOCAL_BLOCKS)
        )

        win_start = tl.maximum(0, bk * STRIDE_POOL - PADDING)
        win_end = tl.minimum(bk * STRIDE_POOL - PADDING + KSIZE, k_len)
        has_window = win_end > win_start

        offs = base[:, None] + win_start + offs_k[None, :]
        mask = (win_start + offs_k[None, :] < win_end[:, None]) & q_mask[:, None]
        vals = tl.load(score_ptr + offs, mask=mask, other=float("-inf"))
        max_val = tl.max(vals, axis=1)

        result = tl.where(
            should_mask_inf, float("inf"), tl.where(has_window, max_val, float("-inf"))
        )
        out_off = head * total_q * max_blocks + q_abs * max_blocks + bk
        tl.store(
            block_score_ptr + out_off,
            result.to(score_ptr.dtype.element_ty),
            mask=q_mask,
        )


def max_pooling_1d_varlen_ref_triton(
    score: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
) -> torch.Tensor:
    num_heads, total_q, _ = score.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    max_blocks = math.ceil(max_seqlen_q / block_size)

    stride_pool = block_size // kernel_stride
    ksize = stride_pool + 1
    padding = 1
    flat_stride = max_seqlen_q // kernel_stride

    block_score = torch.full(
        (num_heads, total_q, max_blocks),
        -float("inf"),
        dtype=score.dtype,
        device=score.device,
    )

    ksize_pow2 = _next_pow2(ksize)
    block_q = 64

    grid = (num_heads * max_blocks,)
    _max_pooling_1d_varlen_kernel[grid](
        score,
        block_score,
        cu_seqlens_q,
        cu_seqlens_k,
        total_q,
        flat_stride,
        num_heads,
        max_blocks,
        BLOCK_SIZE=block_size,
        STRIDE_POOL=stride_pool,
        KSIZE=ksize,
        KSIZE_POW2=ksize_pow2,
        PADDING=padding,
        INIT_BLOCKS=init_blocks,
        LOCAL_BLOCKS=local_blocks,
        BLOCK_Q=block_q,
        batch_size=batch_size,
    )

    return block_score
