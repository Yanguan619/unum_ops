import torch


def get_block_table_ref_torch(
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    seqlen_q: torch.Tensor,
    topk=None,
):
    kHeadGroup = 2
    kSparseBlockSize = 64

    H, T, K = topk_idx.shape
    L = block_table.size(1)

    offsets = torch.arange(
        kSparseBlockSize, device=topk_idx.device, dtype=topk_idx.dtype
    )
    pos = topk_idx.unsqueeze(-1) * kSparseBlockSize + offsets
    neg_flat = (pos < 0).reshape(H, T, K * kSparseBlockSize)
    pos_flat = pos.reshape(H, T, K * kSparseBlockSize)

    bs = token_to_bs
    seqlen = seqlen_q.expand_as(bs) if seqlen_q.numel() == 1 else seqlen_q[bs]
    pos_limit = token_pos_in_bs

    valid = (
        ~neg_flat
        & (pos_flat < seqlen.unsqueeze(0).unsqueeze(-1))
        & (pos_flat < pos_limit.unsqueeze(0).unsqueeze(-1))
    )

    safe_pos = pos_flat.clamp(0, L - 1)
    tbl = block_table[bs]
    vals = torch.gather(tbl.unsqueeze(0).expand(H, -1, -1), 2, safe_pos)

    out = kHeadGroup * vals + torch.arange(
        H, device=topk_idx.device, dtype=topk_idx.dtype
    ).reshape(-1, 1, 1)
    out = torch.where(valid, out, 0)

    return out.permute(1, 0, 2).contiguous()
