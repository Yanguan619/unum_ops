"""BevPool — 纯 PyTorch 实现。

与 AscendC ``bev_pool`` 内核输出布局一致，可在 CPU / GPU / NPU / ONNX
导出场景使用（AscendC 内核不可用时）。
"""

from dataclasses import dataclass

import torch


@dataclass
class BevPoolOutput:
    out: torch.Tensor

    def __iter__(self):
        return iter((self.out,))


def bev_pool_torch(
    feats: torch.Tensor, coords: torch.Tensor, B: int, D: int, H: int, W: int
) -> BevPoolOutput:
    """Pure-PyTorch implementation of the BEV pillar pooling operator.

    Uses only standard PyTorch ops (``scatter_add``) so it can run on
    CPU / NPU and be exported to ONNX.

    Coordinate / axis convention (matches BEVFusion / mmdet3d):
      - coords ``(x, y, z, batch_id)``
      - output ``out[b, z, h, w, :]`` where **h 轴对应 x、w 轴对应 y**
        （BEV 鸟瞰图 H=前向 x、W=横向 y）
      - flat = b*(W*H*D) + z*(W*H) + x*W + y

    Args:
        feats: (N, C) float32, features to scatter
        coords: (N, 4) int64, ``(x, y, z, batch_id)`` BEV grid coordinates
        B: batch size
        D: depth (z) dimension size
        H: BEV height (x) dimension size
        W: BEV width  (y) dimension size

    Returns:
        BevPoolOutput.out: (B, C, D, H, W) float32
    """
    assert feats.shape[0] == coords.shape[0]

    N, C = feats.shape
    coords = coords.long()
    #   out[b, z, x, y, :] — flat = b*(W*H*D) + z*(W*H) + x*W + y
    flat = (
        coords[:, 3] * W * H * D
        + coords[:, 2] * W * H
        + coords[:, 0] * W
        + coords[:, 1]
    )
    index = flat.unsqueeze(1).expand(-1, C)
    out = torch.zeros(B * D * H * W, C, dtype=feats.dtype, device=feats.device)
    out = torch.scatter_add(out, 0, index, feats)
    out = out.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    return BevPoolOutput(out=out)
