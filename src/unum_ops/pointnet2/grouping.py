# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""按索引取/分组特征：GatherOperation 与 GroupingOperation 的纯 PyTorch 等价实现。

``grouping_operation_onnx`` 为不包裹 autograd.Function 的 ONNX 可导出变体。
"""

import torch
from torch.autograd import Function


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""
        CUDA gather_operation 的纯 PyTorch 等价实现。

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor
        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """
        _, C, N = features.size()
        ctx.for_backwards = (idx, C, N)
        idx_exp = idx[:, None, :].expand(-1, C, -1)
        return torch.gather(features, 2, idx_exp)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple:
        idx, C, N = ctx.for_backwards
        grad_features = torch.zeros(grad_out.size(0), C, N, device=grad_out.device)
        idx_exp = idx[:, None, :].expand(-1, C, -1)
        grad_features.scatter_add_(2, idx_exp, grad_out.contiguous())
        return grad_features, None


gather_operation = GatherOperation.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""
        CUDA grouping_operation 的纯 PyTorch 等价实现。

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        ctx.for_backwards = (idx, N)
        out = torch.zeros(B, C, nfeatures, nsample, device=features.device, dtype=features.dtype)
        for k in range(nsample):
            idx_k = idx[:, :, k]
            idx_exp = idx_k[:, None, :].expand(-1, C, -1)
            out[:, :, :, k] = torch.gather(features, 2, idx_exp)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple:
        idx, N = ctx.for_backwards
        B, C, _, nsample = grad_out.shape
        grad_features = torch.zeros(B, C, N, device=grad_out.device, dtype=grad_out.dtype)
        for k in range(nsample):
            idx_k = idx[:, :, k]
            idx_exp = idx_k[:, None, :].expand(-1, C, -1)
            grad_features.scatter_add_(2, idx_exp, grad_out[:, :, :, k])
        return grad_features, None


grouping_operation = GroupingOperation.apply


def grouping_operation_torch(
    features: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """torch-exportable grouping: same logic as GroupingOperation but not wrapped in autograd.Function.
    The loop over nsample is unrolled during ONNX tracing since nsample is a compile-time constant.
    """
    B, npoint, nsample = idx.shape
    _, C, _ = features.shape
    out = torch.zeros(B, C, npoint, nsample, device=features.device, dtype=features.dtype)
    for k in range(nsample):
        idx_k = idx[:, :, k]
        idx_exp = idx_k[:, None, :].expand(-1, C, -1)
        out[:, :, :, k] = torch.gather(features, 2, idx_exp)
    return out
