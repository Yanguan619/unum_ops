"""Ascend 310P BevPoolV3 — DrivingSDK 移植版本 PyTorch 接口封装。

参考 Ascend DrivingSDK `mx_driving/ops/bev_pool_v3.py`，仅实现
``with_depth=False`` 分支（BEVFusion 实际使用场景）。

用法::

    import torch
    from unum_ops.bev_pool_v3 import bev_pool_v3

    feats = torch.randn(100, 80, dtype=torch.float32).npu()
    coords = torch.randint(0, 10, (100, 4), dtype=torch.int32).npu()
    # coords[:, 0]=x, [1]=y, [2]=z, [3]=batch
    out = bev_pool_v3(feats, coords, B=2, D=8, H=16, W=16)  # [B, C, D, H, W]

与 unum_ops.bev_pool 的差异：
    - 不在 Python 侧 argsort / interval 构建，直接展平 coords 为 1D voxel 索引
    - 内核逐点 atomic-add 写入输出（无显式累加缓冲）
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

_SO_REL = os.path.join(
    "csrc", "ascend", "bevpoolV3", "op_extension", "build", "libbevpoolV3_ops.so"
)


@dataclass
class BevPoolV3Output:
    out: torch.Tensor

    def __iter__(self):
        return iter((self.out,))


def _find_ops_lib() -> str:
    env = os.environ.get("BEVPOOLV3_OPS_LIB")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"BEVPOOLV3_OPS_LIB does not exist: {env}")
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_lib = os.path.join(here, "_libs", "libbevpoolV3_ops.so")
    if os.path.exists(pkg_lib):
        return pkg_lib
    for _ in range(6):
        cand = os.path.join(here, _SO_REL)
        if os.path.exists(cand):
            return cand
        here = os.path.dirname(here)
    raise FileNotFoundError(
        f"libbevpoolV3_ops.so not found. Build it via "
        f"csrc/ascend/bevpoolV3/op_extension/CMakeLists.txt or set "
        f"BEVPOOLV3_OPS_LIB=<path>"
    )


_loaded = False


def _ensure_loaded():
    global _loaded
    if not _loaded:
        lib = _find_ops_lib()
        torch.ops.load_library(lib)
        _loaded = True


def bev_pool_v3(feats, coords, B, D, H, W):
    """稀疏点特征散射到稠密 BEV 体素网格（DrivingSDK bev_pool_v3 移植）。

    Args:
        feats: (N, C) float32, NPU 特征
        coords: (N, 4) int32, (x, y, z, batch_id) — 与 unum_ops.bev_pool 约定一致
        B: batch size
        D: depth 维度大小
        H: BEV height（y 维）
        W: BEV width（x 维）

    Returns:
        BevPoolV3Output.out: (B, C, D, H, W) float32
    """
    _ensure_loaded()
    assert feats.shape[0] == coords.shape[0]
    device = feats.device
    N = feats.shape[0]
    C = feats.shape[1]

    if N == 0:
        out = torch.zeros(B, D, H, W, C, dtype=feats.dtype, device=device)
        out = out.permute(0, 4, 1, 2, 3).contiguous()
        return BevPoolV3Output(out=out)

    # 与 unum_ops.bev_pool 一致：x 是 innermost（与原版 BEVFusion 公式相反，
    # 但 unum_ops AscendC kernel 把 coords[:,0] 当 W 维、coords[:,1] 当 H 维，
    # 这里必须用 unum_ops 约定才能与 unum_ops AscendC 输出位置对齐做精度对比）。
    #   ranks_bev = b*D*H*W + z*H*W + y*W + x
    coords_long = coords.long()
    ranks_bev_1d = (
        coords_long[:, 3] * (D * H * W)
        + coords_long[:, 2] * (H * W)
        + coords_long[:, 1] * W
        + coords_long[:, 0]
    ).int().contiguous()

    out = torch.ops.unum.bev_pool_v3(
        None,                       # depth (with_depth=False → None)
        feats.contiguous(),
        None,                       # ranks_depth
        None,                       # ranks_feat
        ranks_bev_1d,
        int(B), int(D), int(H), int(W), int(C),
    )
    # op_extension 输出 [B, D, H, W, C]；permute 到 [B, C, D, H, W]
    return BevPoolV3Output(out=out)


__all__ = ["bev_pool_v3", "BevPoolV3Output"]
