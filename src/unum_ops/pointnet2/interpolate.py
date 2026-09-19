# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""三近邻加权插值：ThreeInterpolate 的纯 PyTorch 等价实现。

与 ``nn_search.three_nn`` 配套（其输出的 idx/weight 直接作为本算子输入）；
idx 约定 int32（ATC 要求 GatherElements 索引为 int32）。
``three_interpolate_onnx`` 为 ONNX 可导出变体。
"""

import torch
from torch.autograd import Function

class ThreeInterpolate(Function):
    @staticmethod
    def forward(
        ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        r"""
        按三近邻索引和权重对特征做加权线性插值。
        CUDA three_interpolate 的纯 PyTorch 等价实现。

        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        out = torch.zeros(B, c, n, device=features.device, dtype=features.dtype)
        for k in range(3):
            idx_k = idx[:, :, k]
            idx_exp = idx_k[:, None, :].expand(-1, c, -1)
            gathered = torch.gather(features, 2, idx_exp)
            w = weight[:, :, k].unsqueeze(1)
            out += gathered * w
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple:
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, _ = grad_out.shape
        grad_features = torch.zeros(B, c, m, device=grad_out.device, dtype=grad_out.dtype)
        for k in range(3):
            idx_k = idx[:, :, k]
            idx_exp = idx_k[:, None, :].expand(-1, c, -1)
            w = weight[:, :, k].unsqueeze(1)
            grad_features.scatter_add_(2, idx_exp, grad_out * w)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


def three_interpolate_torch(
    features: torch.Tensor,
    idx: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """ONNX-exportable three_interpolate: unrolled k=3 loop, NOT wrapped in autograd.Function.
    During ONNX tracing, the loop is unrolled into 3 gather + multiply-accumulate operations.
    """
    B, c, _ = features.shape
    n = idx.shape[1]
    out = torch.zeros(B, c, n, device=features.device, dtype=features.dtype)
    for k in range(3):
        idx_k = idx[:, :, k]
        idx_exp = idx_k[:, None, :].expand(-1, c, -1)
        gathered = torch.gather(features, 2, idx_exp)
        w = weight[:, :, k].unsqueeze(1)
        out += gathered * w
    return out
