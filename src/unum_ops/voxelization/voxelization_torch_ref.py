"""Voxelization — 朴素参考实现（数学算式版）。

逐点 Python 循环、无向量化，逐行对应体素化数学定义，方便对照理解
`voxelization_torch`（向量化优化版）与 `voxelization_ascendc`（AscendC 内核）。

语义与向量化版一致（voxel 集合与每个 voxel 内的点内容），差异仅在于：
  - voxel 行序为首次遇到顺序，而非 (x, y, z) key 升序
  - 每个 voxel 内点序为原始点序
  - 超过 max_voxels 时保留首次遇到的 voxel，而非 key 升序最小的 voxel
    （两者在截断发生时 voxel 集合可能不同；不截断时完全一致）

注意：此版本仅用于理解与验证，不要在生产路径使用（NPU 上逐点
`.item()` 同步实测 >10s 开销，CPU 上也明显慢于向量化版）。
"""

import torch

from ._types import (
    DEFAULT_MAX_NUM_POINTS,
    DEFAULT_MAX_VOXELS,
    DEFAULT_PCR,
    DEFAULT_VOXEL_SIZE,
    VoxelizationOutput,
)


def voxelization_torch_ref(
    points: torch.Tensor,
    voxel_size=DEFAULT_VOXEL_SIZE,
    pcr=DEFAULT_PCR,
    max_num_points: int = DEFAULT_MAX_NUM_POINTS,
    max_voxels: int = DEFAULT_MAX_VOXELS,
) -> VoxelizationOutput:
    """纯 PyTorch 硬体素化（hard voxelization）朴素参考版。

    数学定义（逐点）：
      voxel_x = floor((x - x_min) / vx)
      voxel_y = floor((y - y_min) / vy)
      voxel_z = floor((z - z_min) / vz)

      第 n 个 voxel 由 (voxel_x, voxel_y, voxel_z) 三元组唯一标识；
      落在同一 voxel 内的前 max_num_points 个点归入该 voxel；
      最多保留前 max_voxels 个 voxel（按首次遇到顺序）。

    Args:
        points: (N, 4) float32, 点云 (x, y, z, intensity)
        voxel_size: (vx, vy, vz) 每个 voxel 的尺寸
        pcr: (x_min, y_min, z_min, x_max, y_max, z_max) 点云范围
        max_num_points: 每个 voxel 最多点数
        max_voxels: 最大 voxel 数（-1 表示不限制）

    Returns:
        VoxelizationOutput（可解包为 voxels, coords, num_points, num_voxels）
    """
    N, feat_dim = points.shape
    max_voxels = N if max_voxels == -1 else max_voxels
    max_num_points = N if max_num_points == -1 else max_num_points

    x_min, y_min, z_min = pcr[0], pcr[1], pcr[2]
    x_max, y_max, z_max = pcr[3], pcr[4], pcr[5]
    vx, vy, vz = voxel_size

    # 体素格数（与向量化版 grid 语义一致：round）
    Gx = round((x_max - x_min) / vx)
    Gy = round((y_max - y_min) / vy)
    Gz = round((z_max - z_min) / vz)

    # voxel -> 点的字典；key 用 (x, y, z) 三元组直接表示数学定义
    voxel_map = {}  # {(vx_, vy_, vz_): [point_idx, ...]}

    for i in range(N):
        x, y, z = points[i, 0].item(), points[i, 1].item(), points[i, 2].item()
        # 体素坐标（数学定义：向下取整到 voxel 格）
        vx_ = int((x - x_min) / vx)
        vy_ = int((y - y_min) / vy)
        vz_ = int((z - z_min) / vz)
        # 超出体素格范围的点丢弃（与向量化版 valid 语义一致）
        if not (0 <= vx_ < Gx and 0 <= vy_ < Gy and 0 <= vz_ < Gz):
            continue
        key = (vx_, vy_, vz_)
        if key in voxel_map:
            if len(voxel_map[key]) < max_num_points:
                voxel_map[key].append(i)
        else:
            if len(voxel_map) < max_voxels:
                voxel_map[key] = [i]

    num_voxels = len(voxel_map)
    voxels = torch.zeros(num_voxels, max_num_points, feat_dim, dtype=points.dtype)
    coords = torch.zeros(num_voxels, 3, dtype=torch.int32)
    num_points = torch.zeros(num_voxels, dtype=torch.int32)

    for m, ((vx_, vy_, vz_), idxs) in enumerate(voxel_map.items()):
        coords[m, 0], coords[m, 1], coords[m, 2] = vx_, vy_, vz_
        num_points[m] = len(idxs)
        for j, idx in enumerate(idxs):
            voxels[m, j] = points[idx]

    return VoxelizationOutput(
        voxels=voxels,
        coords=coords,
        num_points=num_points,
        num_voxels=num_voxels,
    )