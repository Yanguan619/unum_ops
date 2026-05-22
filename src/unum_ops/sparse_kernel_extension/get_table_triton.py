import torch
import triton
import triton.language as tl


@triton.jit
def get_block_table_kernel(
    topk_idx_ptr,
    block_table_ptr,
    token_to_bs_ptr,
    token_pos_in_bs_ptr,
    seqlen_q_ptr,
    out_ptr,
    stride_topk_idx_0,
    stride_topk_idx_1,
    stride_topk_idx_2,
    stride_block_table_0,
    stride_out_0,
    stride_out_1,
    seqlen_q_max,
    token_num,
    HEAD_GROUP: tl.constexpr,
    SPARSE_BLOCK_SIZE: tl.constexpr,
    SPARSE_TOPK: tl.constexpr,
):
    pid = tl.program_id(0)

    total_per_token = HEAD_GROUP * SPARSE_TOPK
    token_idx = pid // total_per_token
    remaining = pid % total_per_token
    head_idx = remaining // SPARSE_TOPK
    topk_idx_in_head = remaining % SPARSE_TOPK

    if token_idx >= token_num:
        return

    bs = tl.load(token_to_bs_ptr + token_idx)
    pos_in_bs = tl.load(token_pos_in_bs_ptr + token_idx)
    seqlen_q_bs = tl.load(seqlen_q_ptr + bs)

    topk_offset = (
        head_idx * stride_topk_idx_0
        + token_idx * stride_topk_idx_1
        + topk_idx_in_head * stride_topk_idx_2
    )
    sparse_block_idx = tl.load(topk_idx_ptr + topk_offset)

    if sparse_block_idx < 0:
        return

    offsets = tl.arange(0, SPARSE_BLOCK_SIZE)
    token_idx_in_batch = sparse_block_idx * SPARSE_BLOCK_SIZE + offsets

    out_offset = (
        token_idx * stride_out_0
        + head_idx * stride_out_1
        + topk_idx_in_head * SPARSE_BLOCK_SIZE
        + offsets
    )

    valid = (token_idx_in_batch < seqlen_q_bs) & (token_idx_in_batch < pos_in_bs)

    tbl_offset = bs * stride_block_table_0 + token_idx_in_batch
    tbl_val = tl.load(block_table_ptr + tbl_offset, mask=valid, other=0)

    out_val = tl.where(valid, HEAD_GROUP * tbl_val + head_idx, 0)
    tl.store(out_ptr + out_offset, out_val)


@triton.jit
def get_block_table_kernel_opt(
    topk_idx_ptr,
    block_table_ptr,
    token_to_bs_ptr,
    token_pos_in_bs_ptr,
    seqlen_q_ptr,
    out_ptr,
    stride_topk_idx_0,
    stride_topk_idx_1,
    stride_topk_idx_2,
    stride_block_table_0,
    stride_out_0,
    stride_out_1,
    seqlen_q_max,
    token_num,
    HEAD_GROUP: tl.constexpr,
    SPARSE_BLOCK_SIZE: tl.constexpr,
    SPARSE_TOPK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    topk_idx_in_head = tl.program_id(2)

    bs = tl.load(token_to_bs_ptr + token_idx)
    pos_in_bs = tl.load(token_pos_in_bs_ptr + token_idx)
    seqlen_q_bs = tl.load(seqlen_q_ptr + bs)

    sparse_block_idx = tl.load(
        topk_idx_ptr
        + head_idx * stride_topk_idx_0
        + token_idx * stride_topk_idx_1
        + topk_idx_in_head * stride_topk_idx_2
    )

    if sparse_block_idx < 0:
        return
    offsets = tl.arange(0, SPARSE_BLOCK_SIZE)
    token_idx_in_batch = sparse_block_idx * SPARSE_BLOCK_SIZE + offsets

    limit = tl.minimum(seqlen_q_bs, pos_in_bs)
    valid = token_idx_in_batch < limit

    tbl_val = tl.load(
        block_table_ptr + bs * stride_block_table_0 + token_idx_in_batch,
        mask=valid,
        other=0,
    )

    out_val = tl.where(valid, HEAD_GROUP * tbl_val + head_idx, 0)
    tl.store(
        out_ptr
        + token_idx * stride_out_0
        + head_idx * stride_out_1
        + topk_idx_in_head * SPARSE_BLOCK_SIZE
        + offsets,
        out_val,
    )


def get_block_table_ref_triton(
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    seqlen_q: torch.Tensor,
    topk=None,
):
    kSparseBlockSize = 64

    H, T, K = topk_idx.shape
    batch_size = block_table.size(0)

    if topk is not None:
        assert topk == K, f"topk={topk} != K={K}"

    if token_pos_in_bs.numel() == 1:
        token_pos_in_bs = token_pos_in_bs.expand(T).contiguous()
    if seqlen_q.numel() == 1:
        seqlen_q = seqlen_q.expand(batch_size).contiguous()

    out = torch.zeros(
        T, H, K * kSparseBlockSize, dtype=torch.int32, device=topk_idx.device
    )

    grid = (T * H * K,)
    get_block_table_kernel[grid](
        topk_idx,
        block_table,
        token_to_bs,
        token_pos_in_bs,
        seqlen_q,
        out,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        block_table.size(1),
        T,
        HEAD_GROUP=H,
        SPARSE_BLOCK_SIZE=kSparseBlockSize,
        SPARSE_TOPK=K,
    )

    return out


def get_block_table_ref_triton_v2(
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    seqlen_q: torch.Tensor,
    topk=None,
):
    kSparseBlockSize = 64

    H, T, K = topk_idx.shape
    batch_size = block_table.size(0)

    if topk is not None:
        assert topk == K, f"topk={topk} != K={K}"

    if token_pos_in_bs.numel() == 1:
        token_pos_in_bs = token_pos_in_bs.expand(T).contiguous()
    if seqlen_q.numel() == 1:
        seqlen_q = seqlen_q.expand(batch_size).contiguous()

    out = torch.zeros(
        T, H, K * kSparseBlockSize, dtype=torch.int32, device=topk_idx.device
    )

    grid = (T, H, K)
    get_block_table_kernel_opt[grid](
        topk_idx,
        block_table,
        token_to_bs,
        token_pos_in_bs,
        seqlen_q,
        out,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        block_table.size(1),
        T,
        HEAD_GROUP=H,
        SPARSE_BLOCK_SIZE=kSparseBlockSize,
        SPARSE_TOPK=K,
    )

    return out
