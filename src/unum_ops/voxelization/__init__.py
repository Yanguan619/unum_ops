"""Ascend 310P Voxelization — PyTorch 接口封装。

依赖:
  - torch + torch_npu（NPU 设备）
  - libvoxelization_ops.so（op_extension 编译产物，见 csrc/ascend/voxelization/op_extension）

用法::

    import torch, torch_npu
    from unum_ops.voxelization import voxelization

    points = torch.randn(17221, 4, dtype=torch.float32).npu()
    out = voxelization(points, voxel_size=[0.16, 0.16, 4.0],
                       pcr=[0.0, -39.68, -3.0, 69.12, 39.68, 1.0])
    voxels, coords, num_points, num_voxels = out
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

_SO_REL = os.path.join(
    "csrc", "ascend", "voxelization", "op_extension", "build", "libvoxelization_ops.so"
)

DEFAULT_MAX_NUM_POINTS = 32
DEFAULT_MAX_VOXELS = 40000
DEFAULT_VOXEL_SIZE = (0.16, 0.16, 4.0)
DEFAULT_PCR = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)


@dataclass
class VoxelizationOutput:
    """voxelization 输出。

    Attributes:
        voxels: (M, max_num_points, 4) float32, 已截断到实际 voxel 数 M
        coords: (M, 3) int32, voxel 网格坐标
        num_points: (M,) int32, 每个 voxel 的点数
        num_voxels: int, 实际 voxel 数 M
    """

    voxels: torch.Tensor
    coords: torch.Tensor
    num_points: torch.Tensor
    num_voxels: int

    def __iter__(self):
        return iter((self.voxels, self.coords, self.num_points, self.num_voxels))


def _find_ops_lib() -> str:
    """定位 libvoxelization_ops.so：优先环境变量，其次源码树。"""
    env = os.environ.get("VOXELIZATION_OPS_LIB")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"VOXELIZATION_OPS_LIB does not exist: {env}")
        return env
    # 从 unum_ops 包向上找到仓库根，再定位 .so
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        cand = os.path.join(here, _SO_REL)
        if os.path.exists(cand):
            return cand
        here = os.path.dirname(here)
    raise FileNotFoundError(
        f"libvoxelization_ops.so not found. Build it via "
        f"csrc/ascend/voxelization/op_extension/CMakeLists.txt or set "
        f"VOXELIZATION_OPS_LIB=<path>"
    )


_loaded = False


def _ensure_loaded():
    global _loaded
    if not _loaded:
        torch.ops.load_library(_find_ops_lib())
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
        points,
        list(voxel_size),
        list(pcr),
        int(max_num_points),
        int(max_voxels),
    )
    num_voxels = int(raw[3].item())
    return VoxelizationOutput(
        voxels=raw[0][:num_voxels],
        coords=raw[1][:num_voxels],
        num_points=raw[2][:num_voxels],
        num_voxels=num_voxels,
    )


__all__ = ["voxelization", "VoxelizationOutput"]
