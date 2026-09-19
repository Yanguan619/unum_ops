# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""半径类邻域查询：BallQuery / CylinderQuery 的纯 PyTorch 等价实现。

BallQuery / CylinderQuery 含 python 双重循环，检测到 ``device.type == 'npu'``
时主动 fallback 到 CPU 计算再拷回；``cylinder_query_onnx`` 为全向量化变体
（matmul + topk），无 python 循环，ONNX 可导出。近邻搜索类算子
（knn / three_nn）见 ``nn_search`` 模块。
"""

import torch
from torch.autograd import Function


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor):
        r"""
        以 new_xyz 中每个点为球心，收集半径 radius 内的点索引。
        CUDA ball_query 的纯 PyTorch 等价实现。

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        B, _, _ = xyz.shape
        _, npoint, _ = new_xyz.shape
        device = xyz.device
        dist = torch.cdist(new_xyz, xyz, p=2)
        idx = torch.zeros(B, npoint, nsample, dtype=torch.long, device=device)
        # Run on CPU to avoid NPU Python-loop overhead
        if device.type == "npu":
            return BallQuery.apply(radius, nsample, xyz.cpu(), new_xyz.cpu()).to(device)
        for b in range(B):
            for j in range(npoint):
                valid = torch.where(dist[b, j] < radius)[0]
                n_valid = valid.shape[0]
                if n_valid > 0:
                    k = min(n_valid, nsample)
                    idx[b, j, :k] = valid[:k]
                    if k < nsample:
                        idx[b, j, k:] = valid[0]
        return idx

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple:
        return None, None, None, None


ball_query = BallQuery.apply


class CylinderQuery(Function):
    @staticmethod
    def forward(
        ctx,
        radius: float,
        hmin: float,
        hmax: float,
        nsample: int,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
        rot: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        以 new_xyz 为圆柱中心、rot 为姿态的圆柱邻域查询。
        CUDA cylinder_query 的纯 PyTorch 等价实现。

        Parameters
        ----------
        radius : float
            radius of the cylinders
        hmin, hmax : float
            endpoints of cylinder height in x-rotation axis
        nsample : int
            maximum number of features in the cylinders
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the cylinder query
        rot: torch.Tensor
            (B, npoint, 9) flatten rotation matrices from
                           cylinder frame to world frame

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        B, _, _ = xyz.shape
        npoint = new_xyz.shape[1]
        device = xyz.device
        radius2 = radius * radius
        idx = torch.zeros(B, npoint, nsample, dtype=torch.long, device=device)
        # Run on CPU to avoid NPU Python-loop overhead
        if device.type == "npu":
            return CylinderQuery.apply(
                radius, hmin, hmax, nsample, xyz.cpu(), new_xyz.cpu(), rot.cpu()
            ).to(device)
        for b in range(B):
            # Center all points by each query center: (npoint, N, 3)
            centered = new_xyz[b].unsqueeze(1) - xyz[b].unsqueeze(0)  # (npoint, 1, 3) - (1, N, 3)
            centered = -centered  # xyz - center
            # Reshape rotations: (npoint, 9) -> (npoint, 3, 3)
            R_mat = rot[b].reshape(npoint, 3, 3)
            # Rotate all points for all query centers at once: (npoint, N, 3)
            rotated = torch.matmul(centered, R_mat)
            # Cylinder check: y^2 + z^2 < radius^2, hmin < x < hmax
            mask = (
                (rotated[..., 1] ** 2 + rotated[..., 2] ** 2 < radius2)
                & (rotated[..., 0] > hmin)
                & (rotated[..., 0] < hmax)
            )
            for j in range(npoint):
                valid = torch.where(mask[j])[0]
                n_valid = valid.shape[0]
                if n_valid > 0:
                    k = min(n_valid, nsample)
                    idx[b, j, :k] = valid[:k]
                    if k < nsample:
                        idx[b, j, k:] = valid[0]
        return idx

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple:
        return None, None, None, None, None, None, None


cylinder_query = CylinderQuery.apply


def cylinder_query_torch(
    radius: float,
    hmin: float,
    hmax: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    rot: torch.Tensor,
) -> torch.Tensor:
    """ONNX-exportable cylinder query using matmul + topk.
    No Python loops — fully vectorized and ONNX-traceable.

    Args:
        radius: cylinder radius
        hmin, hmax: cylinder height bounds in x-rotation axis
        nsample: max points to gather
        xyz: (B, N, 3) point cloud
        new_xyz: (B, npoint, 3) query centers
        rot: (B, npoint, 3, 3) rotation matrices (world -> cylinder frame)

    Returns:
        idx: (B, npoint, nsample) int32 indices
    """
    radius2 = radius * radius

    # Center all points relative to each query center: (B, npoint, N, 3)
    centered = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)
    centered = -centered  # xyz - new_xyz

    # Batch rotate: (B, npoint, N, 3) @ (B, npoint, 3, 3).transpose(-1, -2) -> (B, npoint, N, 3)
    rotated = torch.matmul(centered, rot.transpose(-1, -2))

    # Cylinder bounds: y^2+z^2 < radius^2, hmin < x < hmax
    d2 = rotated[..., 1] ** 2 + rotated[..., 2] ** 2  # (B, npoint, N)
    valid = (d2 < radius2) & (rotated[..., 0] > hmin) & (rotated[..., 0] < hmax)

    # Set invalid to large value, get nearest nsample
    large = torch.full_like(d2, 1e10)
    dist = torch.where(valid, d2, large)
    _, idx = torch.topk(dist, k=nsample, dim=-1, largest=False)

    return idx.to(torch.int32)
