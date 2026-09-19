"""Ascend 310P Voxelization — AscendC 内核封装（TORCH_LIBRARY）。

用法::

    import torch, torch_npu
    from unum_ops.voxelization import voxelization

    points = torch.randn(17221, 4, dtype=torch.float32).npu()
    out = voxelization(points)
    voxels, coords, num_points, num_voxels = out
"""
from __future__ import annotations

import ctypes
import os

import torch

from unum_ops.utils import find_ops_lib

from ._types import (
    DEFAULT_MAX_NUM_POINTS,
    DEFAULT_MAX_VOXELS,
    DEFAULT_PCR,
    DEFAULT_VOXEL_SIZE,
    VoxelizationOutput,
)

_SO_REL = os.path.join(
    "csrc", "ascend", "voxelization", "op_extension", "build", "libvoxelization_ops.so"
)


def _find_ops_lib() -> str:
    """定位 libvoxelization_ops.so：优先环境变量，其次包内 _libs/，最后源码树。"""
    return find_ops_lib("VOXELIZATION_OPS_LIB", "libvoxelization_ops.so", _SO_REL, __file__)


_loaded = False


def _ensure_loaded():
    global _loaded
    if not _loaded:
        lib = _find_ops_lib()
        # 用 RTLD_GLOBAL 加载，避免与 torch_npu 已加载的 ACL 库符号冲突
        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        torch.ops.load_library(lib)
        _loaded = True


def voxelization(
    points: torch.Tensor,
    voxel_size=DEFAULT_VOXEL_SIZE,
    pcr=DEFAULT_PCR,
    max_num_points: int = DEFAULT_MAX_NUM_POINTS,
    max_voxels: int = DEFAULT_MAX_VOXELS,
) -> VoxelizationOutput:
    """在 NPU 上执行 voxelization。

    Args:
        points: (N, 4) float32, NPU 上的点云 (x, y, z, intensity)
        voxel_size: (vx, vy, vz) 每个 voxel 的尺寸
        pcr: (x_min, y_min, z_min, x_max, y_max, z_max) 点云范围
        max_num_points: 每个 voxel 最多点数
        max_voxels: 最大 voxel 数

    Returns:
        VoxelizationOutput（可解包为 voxels, coords, num_points, num_voxels）
    """
    _ensure_loaded()
    raw = torch.ops.npu.voxelization(
        points.contiguous(),
        list(voxel_size),
        list(pcr),
        int(max_num_points),
        int(max_voxels),
    )
    num_voxels = int(raw[3].item())
    return VoxelizationOutput(
        voxels=raw[0][:num_voxels],
        coords=raw[1][:num_voxels, :3],
        num_points=raw[2][:num_voxels, 0],
        num_voxels=num_voxels,
    )
