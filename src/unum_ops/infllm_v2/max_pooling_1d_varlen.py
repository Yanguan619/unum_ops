import math

import torch
import triton
import triton.language as tl


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 32, "BLOCK_BK": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_BK": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_BK": 2}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 32, "BLOCK_BK": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 16, "BLOCK_BK": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_BK": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_BK": 4}, num_warps=8, num_stages=2),
    ],
    key=["total_q", "max_blocks"],
)
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
    BLOCK_BK: tl.constexpr,
    batch_size,
):
    pid_head = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_bk = tl.program_id(2)

    q_start = pid_q * BLOCK_Q
    bk_start = pid_bk * BLOCK_BK

    offs_q = tl.arange(0, BLOCK_Q)
    q_abs = q_start + offs_q
    q_mask = q_abs < total_q

    offs_bk = tl.arange(0, BLOCK_BK)
    bk_vals = bk_start + offs_bk
    bk_mask = bk_vals < max_blocks

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

    base_q = pid_head * total_q * flat_stride + q_abs * flat_stride

    win_start_val = bk_vals * STRIDE_POOL - PADDING

    max_vals = tl.full([BLOCK_Q, BLOCK_BK], value=float("-inf"), dtype=tl.float32)

    for ki in range(KSIZE):
        k_pos = win_start_val + ki

        win_start_clamped = tl.maximum(0, win_start_val)

        valid_bk = (k_pos >= win_start_clamped) & (k_pos >= 0)

        valid_qbk = (
            (k_pos[None, :] < k_len[:, None])
            & valid_bk[None, :]
            & q_mask[:, None]
            & bk_mask[None, :]
        )

        load_offs = base_q[:, None] + k_pos[None, :]
        vals = tl.load(score_ptr + load_offs, mask=valid_qbk, other=float("-inf"))
        max_vals = tl.maximum(max_vals, vals.to(tl.float32))

    should_mask_inf = (bk_vals[None, :] < INIT_BLOCKS) | (
        (off_bq[:, None] >= bk_vals[None, :])
        & (off_bq[:, None] <= bk_vals[None, :] + LOCAL_BLOCKS)
    )

    win_start_clamped = tl.maximum(0, win_start_val)
    win_end = tl.minimum(win_start_val + KSIZE, k_len[:, None])
    has_window = win_end > win_start_clamped[None, :]

    result = tl.where(
        should_mask_inf, float("inf"), tl.where(has_window, max_vals, float("-inf"))
    )

    out_offs = (
        pid_head * total_q * max_blocks + q_abs[:, None] * max_blocks + bk_vals[None, :]
    )
    store_mask = q_mask[:, None] & bk_mask[None, :]
    tl.store(
        block_score_ptr + out_offs,
        result.to(score_ptr.dtype.element_ty),
        mask=store_mask,
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
    flat_stride = max_seqlen_q // kernel_stride

    block_score = torch.full(
        (num_heads, total_q, max_blocks),
        -float("inf"),
        dtype=score.dtype,
        device=score.device,
    )

    ksize_pow2 = _next_pow2(ksize)

    grid = lambda META: (
        num_heads,
        triton.cdiv(total_q, META["BLOCK_Q"]),
        triton.cdiv(max_blocks, META["BLOCK_BK"]),
    )

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
        PADDING=1,
        INIT_BLOCKS=init_blocks,
        LOCAL_BLOCKS=local_blocks,
        batch_size=batch_size,
    )

    return block_score
