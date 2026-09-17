# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""近邻搜索：knn / three_nn 的纯 PyTorch 等价实现。

两者同为"最近邻搜索"，但来源与布局约定不同：
  - knn 来自原 graspnet 的独立 knn 包（自带 CUDA 扩展），
    输入为 (B, C, M) 通道优先布局，返回 (B, k, N) 索引；
  - three_nn 属 pointnet2 算子族，输入为 (B, N, 3) xyz 布局，
    专供 ``three_interpolate`` 特征传播使用（k 固定为 3）。

ATC 要求 GatherElements 索引为 int32：``three_nn`` 索引输出统一
``.to(torch.int32)``。
"""

import torch
from torch.autograd import Function


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> tuple:
        r"""
        查找 unknown 在 known 中的三个最近邻。
        CUDA three_nn 的纯 PyTorch 等价实现。

        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        # 平方在非负区间单调，topk(dist) 与 topk(dist**2) 选点等价，
        # 直接取距离免去 **2 全矩阵平方与 sqrt 往返
        dist = torch.cdist(unknown, known, p=2)
        dist_small, idx = torch.topk(dist, k=3, dim=2, largest=False, sorted=True)
        # Cast to int32: ATC requires int32 for GatherElements indices
        return dist_small, idx.to(torch.int32)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple:
        return None, None


three_nn = ThreeNN.apply


def knn(
    ref: torch.Tensor,
    query: torch.Tensor,
    k: int = 1,
) -> torch.Tensor:
    """Compute k nearest neighbors for each query point.
    CUDA knn 的纯 PyTorch 等价实现。

    Args:
        ref: (B, C, M) reference points
        query: (B, C, N) query points
        k: number of nearest neighbors

    Returns:
        inds: (B, k, N) indices of k nearest neighbors in ref
    """
    dist = torch.cdist(query.transpose(1, 2), ref.transpose(1, 2), p=2)  # (B, N, M)
    _, inds = torch.topk(dist, k=k, dim=2, largest=False, sorted=True)  # (B, N, k)
    inds = inds.transpose(1, 2).contiguous()  # (B, k, N)
    return inds
