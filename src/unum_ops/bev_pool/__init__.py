"""Ascend 310P BevPool — PyTorch 接口封装（TORCH_LIBRARY）。

LSS Lift-Splat 的 Splat 阶段：把稀疏的 frustum 点特征 segment-sum scatter
到稠密 BEV 体素网格。

用法::

    import torch, torch_npu
    from unum_ops.bev_pool import bev_pool

    feats = torch.randn(100, 80, dtype=torch.float32).npu()
    coords = torch.randint(0, 10, (100, 4), dtype=torch.int64).npu()
    out = bev_pool(feats, coords, B=2, D=8, H=16, W=16)  # [B, C, D, H, W]
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch

_SO_REL = os.path.join(
    "csrc", "ascend", "bev_pool", "op_extension", "build", "libbev_pool_ops.so"
)


@dataclass
class BevPoolOutput:
    """bev_pool 输出。

    Attributes:
        out: (B, C, D, H, W) float32，按深度拼接前的稠密 BEV 特征体
    """

    out: torch.Tensor

    def __iter__(self):
        return iter((self.out,))


def _find_ops_lib() -> str:
    """定位 libbev_pool_ops.so：优先环境变量，其次包内 _libs/，最后源码树。"""
    env = os.environ.get("BEV_POOL_OPS_LIB")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"BEV_POOL_OPS_LIB does not exist: {env}")
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    # pip install 后 .so 在包目录 _libs/ 下
    pkg_lib = os.path.join(here, "_libs", "libbev_pool_ops.so")
    if os.path.exists(pkg_lib):
        return pkg_lib
    # 开发模式：源码树
    for _ in range(6):
        cand = os.path.join(here, _SO_REL)
        if os.path.exists(cand):
            return cand
        here = os.path.dirname(here)
    raise FileNotFoundError(
        f"libbev_pool_ops.so not found. Build it via "
        f"csrc/ascend/bev_pool/op_extension/CMakeLists.txt or set "
        f"BEV_POOL_OPS_LIB=<path>"
    )


_loaded = False


def _ensure_loaded():
    global _loaded
    if not _loaded:
        lib = _find_ops_lib()
        torch.ops.load_library(lib)
        _loaded = True


def bev_pool(feats, coords, B, D, H, W):
    """稀疏点特征 segment-sum 散射到稠密 BEV 体素网格。

    Python 层按 rank 排序 feats/coords 并构造 interval，kernel 做 segment-sum。
    用 torch.sort 一次性拿到排序值和索引，避免单独 gather 排序值。

    Args:
        feats: (N, C) float32, NPU 特征（无需排序）
        coords: (N, 4) int64, (x, y, z, batch_id)，BEV 网格坐标
        B: batch size
        D: depth 维度大小
        H: BEV height
        W: BEV width

    Returns:
        BevPoolOutput.out: (B, C, D, H, W) float32
    """
    _ensure_loaded()
    assert feats.shape[0] == coords.shape[0]
    device = feats.device
    N = feats.shape[0]
    C = feats.shape[1]

    if N == 0:
        out = torch.zeros(B, D, H, W, C, dtype=feats.dtype, device=device)
        out = out.permute(0, 4, 1, 2, 3).contiguous()
        return BevPoolOutput(out=out)

    ranks = (
        coords[:, 0] + coords[:, 1] * W + coords[:, 2] * (W * H) +
        coords[:, 3] * (W * H * D))
    # float32 argsort 在 AiCore 上运行（快），int64 回退到 AiCpu（慢 100x）。
    # 当 B*D*H*W > 2^24 时 float32 精度不足，不同 rank 可能碰撞。
    # 该场景极罕见（需要 >1600 万体素），碰撞时用 int64 兜底。
    if B * D * H * W > 16777216:
        indices = ranks.argsort()
    else:
        indices = ranks.float().argsort()
    # index_select 比 feats[indices] 快 ~10x（NPU 随机读优化）
    feats = torch.index_select(feats, 0, indices)
    coords = torch.index_select(coords, 0, indices).int().contiguous()
    ranks_sorted = ranks[indices]

    kept = torch.ones(N, device=device, dtype=torch.bool)
    kept[1:] = ranks_sorted[1:] != ranks_sorted[:-1]
    interval_starts = torch.where(kept)[0].int().contiguous()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = N - interval_starts[-1]

    out = torch.ops.unum.bev_pool(
        feats,
        coords,
        interval_starts,
        interval_lengths,
        int(B),
        int(D),
        int(H),
        int(W),
    )
    out = out.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    return BevPoolOutput(out=out)


def bev_pool_torch(feats, coords, B, D, H, W):
    """Pure-PyTorch implementation of the BEV pillar pooling operator.

    Uses only standard PyTorch ops (``scatter_add``) so it can run on
    CPU / NPU and be exported to ONNX.  The output layout is identical to
    the AscendC ``bev_pool`` kernel.

    Args:
        feats: (N, C) float32, features to scatter
        coords: (N, 4) int64, ``(x, y, z, batch_id)`` BEV grid coordinates
        B: batch size
        D: depth (z) dimension size
        H: BEV height (y) dimension size
        W: BEV width  (x) dimension size

    Returns:
        BevPoolOutput.out: (B, C, D, H, W) float32
    """
    assert feats.shape[0] == coords.shape[0]

    N, C = feats.shape
    coords = coords.long()
    # flat index matching the AscendC kernel output layout:
    #   out[b, z, y, x, :] — flat = x + y*W + z*W*H + b*W*H*D
    flat = (coords[:, 0] + coords[:, 1] * W +
            coords[:, 2] * W * H +
            coords[:, 3] * W * H * D)
    index = flat.unsqueeze(1).expand(-1, C)
    out = torch.zeros(B * D * H * W, C, dtype=feats.dtype, device=feats.device)
    out = torch.scatter_add(out, 0, index, feats)
    out = out.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    return BevPoolOutput(out=out)


__all__ = ["bev_pool", "bev_pool_torch", "BevPoolOutput"]