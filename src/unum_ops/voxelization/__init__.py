"""Ascend 310P Voxelization — PyTorch 接口封装（TORCH_LIBRARY）。

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
        coords: (M, 8→3) int32, 内部填充到 8 列（310P 32B 对齐要求），对外暴露前 3 列
        num_points: (M, 8→) int32, 内部填充到 8 列，对外暴露第 0 列
        num_voxels: int, 实际 voxel 数 M
    """

    voxels: torch.Tensor
    coords: torch.Tensor
    num_points: torch.Tensor
    num_voxels: int

    def __iter__(self):
        return iter((self.voxels, self.coords, self.num_points, self.num_voxels))


def _find_ops_lib() -> str:
    """定位 libvoxelization_ops.so：优先环境变量，其次包内 _libs/，最后源码树。"""
    env = os.environ.get("VOXELIZATION_OPS_LIB")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"VOXELIZATION_OPS_LIB does not exist: {env}")
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    # pip install 后 .so 在包目录 _libs/ 下
    pkg_lib = os.path.join(here, "_libs", "libvoxelization_ops.so")
    if os.path.exists(pkg_lib):
        return pkg_lib
    # 开发模式：源码树
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


def voxelization_torch(
    points: torch.Tensor,
    voxel_size=DEFAULT_VOXEL_SIZE,
    pcr=DEFAULT_PCR,
    max_num_points: int = DEFAULT_MAX_NUM_POINTS,
    max_voxels: int = DEFAULT_MAX_VOXELS,
) -> VoxelizationOutput:
    """纯 PyTorch 硬体素化（hard voxelization），CPU/GPU/NPU 通用，不依赖 CUDA 算子库。

    与 CUDA 版 hard_voxelize 语义一致：
      - 每个 voxel 内取前 max_num_points 个点（按原始点序）
      - 输出 coords 为 (x, y, z)
    与逐点 Python 循环版的差异：voxel 行按 (x, y, z) key 升序排列（循环版为首次
    遇到顺序）。voxel 集合与每个 voxel 内的点内容完全一致，下游按 coords 携带
    处理，行序不影响结果。

    实现说明（向量化，无 Python 逐点循环——NPU 上逐点 .item() 同步实测 >10s
    开销）：argsort(int64, stable) + unique_consecutive + repeat_interleave。
    注意 argsort 必须用 int64 key（float32 在 key > 2^24 时丢精度），int64 sort
    在 NPU 上走 AiCpu，3.4 万点实测整体 < 0.4s。

    Args:
        points: (N, 4) float32, 点云 (x, y, z, intensity)
        voxel_size: (vx, vy, vz) 每个 voxel 的尺寸
        pcr: (x_min, y_min, z_min, x_max, y_max, z_max) 点云范围
        max_num_points: 每个 voxel 最多点数
        max_voxels: 最大 voxel 数

    Returns:
        VoxelizationOutput（可解包为 voxels, coords, num_points, num_voxels）
    """
    pts = points
    vs = torch.as_tensor(voxel_size, dtype=torch.float32, device=pts.device)
    cr = torch.as_tensor(pcr, dtype=torch.float32, device=pts.device)
    N, feat_dim = pts.shape
    if N == 0:
        return VoxelizationOutput(
            voxels=pts.new_zeros(0, max_num_points, feat_dim),
            coords=pts.new_zeros(0, 3, dtype=torch.int32),
            num_points=pts.new_zeros(0, dtype=torch.int32),
            num_voxels=0,
        )
    # 体素坐标
    coords = ((pts[:, :3] - cr[:3]) / vs).floor().long()
    grid = torch.round((cr[3:] - cr[:3]) / vs).long()
    valid = (coords >= 0).all(1) & (coords < grid).all(1)
    coords_v = coords[valid]
    pts_v = pts[valid]
    Nv = pts_v.shape[0]
    if Nv == 0:
        return VoxelizationOutput(
            voxels=pts.new_zeros(0, max_num_points, feat_dim),
            coords=pts.new_zeros(0, 3, dtype=torch.int32),
            num_points=pts.new_zeros(0, dtype=torch.int32),
            num_voxels=0,
        )
    # 编码 key = x*Gy*Gz + y*Gz + z（int64；.item() 仅两次标量同步）
    Gy, Gz = int(grid[1].item()), int(grid[2].item())
    key = coords_v[:, 0] * (Gy * Gz) + coords_v[:, 1] * Gz + coords_v[:, 2]

    sort_order = torch.argsort(key, stable=True)
    sorted_pts = pts_v[sort_order]
    sorted_key = key[sort_order]

    unique_key, inverse, counts = torch.unique_consecutive(
        sorted_key, return_inverse=True, return_counts=True)
    num_voxels = unique_key.shape[0]
    if max_voxels != -1 and num_voxels > max_voxels:
        # 按 key 升序保留前 max_voxels 个 voxel
        keep_mask = inverse < max_voxels
        sorted_pts = sorted_pts[keep_mask]
        inverse = inverse[keep_mask]
        unique_key, _, counts = torch.unique_consecutive(
            sorted_key[keep_mask], return_inverse=True, return_counts=True)
        num_voxels = unique_key.shape[0]

    # 每个 voxel 内取前 max_num_points 个点（stable 排序保证 = 原始点序）
    cum_counts = torch.cumsum(counts, 0)
    voxel_starts = cum_counts - counts           # exclusive 前缀和
    total_pts = sorted_pts.shape[0]
    point_pos = torch.arange(total_pts, device=pts.device) \
        - torch.repeat_interleave(voxel_starts, counts)

    if max_num_points != -1:
        keep = point_pos < max_num_points
        sorted_pts = sorted_pts[keep]
        inverse = inverse[keep]
        point_pos = point_pos[keep]
        counts = counts.clamp(max=max_num_points)

    voxels = pts.new_zeros(num_voxels, max_num_points, feat_dim)
    num_pts_out = torch.zeros(num_voxels, device=pts.device, dtype=torch.int32)
    if sorted_pts.shape[0] > 0:
        voxels[inverse, point_pos] = sorted_pts
        num_pts_out = counts.to(torch.int32)

    # 解码 coords (x, y, z)
    coords_out = torch.zeros(num_voxels, 3, device=pts.device, dtype=torch.int32)
    coords_out[:, 0] = unique_key // (Gy * Gz)
    coords_out[:, 1] = (unique_key % (Gy * Gz)) // Gz
    coords_out[:, 2] = unique_key % Gz

    return VoxelizationOutput(
        voxels=voxels,
        coords=coords_out,
        num_points=num_pts_out,
        num_voxels=int(num_voxels),
    )


__all__ = ["voxelization", "voxelization_torch", "VoxelizationOutput"]