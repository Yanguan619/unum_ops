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


def voxelization_torch(
    points: torch.Tensor,
    voxel_size=DEFAULT_VOXEL_SIZE,
    pcr=DEFAULT_PCR,
    max_num_points: int = DEFAULT_MAX_NUM_POINTS,
    max_voxels: int = DEFAULT_MAX_VOXELS,
) -> VoxelizationOutput:
    """纯 PyTorch 硬体素化（hard voxelization），CPU/GPU 通用，不依赖 CUDA 算子库。

    与 CUDA 版 hard_voxelize 语义一致：
      - 按输入顺序扫描点，voxel 按首次遇到顺序编号
      - 每个 voxel 内取前 max_num_points 个点
      - 输出 coords 为 (x, y, z)

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
    # 编码 key = (x, y, z) 用于 hash
    key = (coords_v[:, 0] * grid[1] * grid[2] +
           coords_v[:, 1] * grid[2] + coords_v[:, 2])
    voxels = pts.new_zeros(max_voxels, max_num_points, feat_dim)
    coords_out = pts.new_zeros(max_voxels, 3, dtype=torch.int)
    num_pts = pts.new_zeros(max_voxels, dtype=torch.int64)
    vmap = {}
    vcnt = 0
    for i in range(Nv):
        k = int(key[i].item())
        if k not in vmap:
            if vcnt >= max_voxels:
                continue
            vmap[k] = vcnt
            # 输出 coords 为 (x, y, z)
            c = coords_v[i]
            coords_out[vcnt, 0] = int(c[0].item())
            coords_out[vcnt, 1] = int(c[1].item())
            coords_out[vcnt, 2] = int(c[2].item())
            vcnt += 1
        vidx = vmap[k]
        if int(num_pts[vidx].item()) < max_num_points:
            n = int(num_pts[vidx].item())
            voxels[vidx, n] = pts_v[i]
            num_pts[vidx] = n + 1
    return VoxelizationOutput(
        voxels=voxels[:vcnt],
        coords=coords_out[:vcnt],
        num_points=num_pts[:vcnt].to(torch.int32),
        num_voxels=vcnt,
    )


__all__ = ["voxelization", "voxelization_torch", "VoxelizationOutput"]