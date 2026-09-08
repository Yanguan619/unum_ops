"""spconv_gemm 的 AscendC 后端封装（可选加速路径，Cube 加速版）。

当输入位于 NPU 且 AscendC 扩展可加载时，``SparseConvolution._gather``
会把特征聚合（gather 后的 GEMM + bias）卸载到自定义 AscendC 内核
``torch.ops.unum.spconv_gemm``；否则回退到 torch 2D GEMM 路径。

内核契约（与 conv.py::_gather 中 2D GEMM 完全等价）::

    out[n, o] = bias[o] + sum_k feats[n, k] * weight[k, o]

  - feats   (N, K_flat) float16，已按邻居表 gather 展平，无效邻居已置 0
  - weight  (K_flat, C_out) float16（即 ``_wT2d`` 布局）
  - bias    (C_out,) float32
  - out     (N, C_out) float32

反向传播用 torch einsum 计算（同一线性算子），保证训练时梯度正确。
"""
from __future__ import annotations

import os

import torch

_SO_REL = os.path.join(
    "csrc", "ascend", "spconv", "op_extension", "build", "libspconv_gemm_ops.so"
)

_loaded = False
_BLOCK_NUM = 8


def _find_ops_lib() -> str:
    """定位 libspconv_gemm_ops.so：优先环境变量，其次包内 _libs/，最后源码树。"""
    env = os.environ.get("SPCONV_OPS_LIB")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"SPCONV_OPS_LIB does not exist: {env}")
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    # pip install 后 .so 在包目录 _libs/ 下
    pkg_lib = os.path.join(here, "_libs", "libspconv_gemm_ops.so")
    if os.path.exists(pkg_lib):
        return pkg_lib
    # 开发模式：源码树
    for _ in range(6):
        cand = os.path.join(here, _SO_REL)
        if os.path.exists(cand):
            return cand
        here = os.path.dirname(here)
    raise FileNotFoundError(
        f"libspconv_gemm_ops.so not found. Build it via "
        f"csrc/ascend/spconv/op_extension/CMakeLists.txt or set "
        f"SPCONV_OPS_LIB=<path>"
    )


def available() -> bool:
    """AscendC 扩展是否可加载（首次调用会实际加载 .so）。"""
    global _loaded
    if _loaded:
        return True
    try:
        torch.ops.load_library(_find_ops_lib())
        _loaded = True
        return True
    except Exception:
        return False


class _SpconvGemmAscendC(torch.autograd.Function):
    """autograd 封装：forward 走 AscendC Cube 内核，backward 用 torch einsum。"""

    @staticmethod
    def forward(ctx, feats, weight, bias, params):
        ctx.save_for_backward(feats, weight)
        return torch.ops.unum.spconv_gemm(feats.contiguous(), weight.contiguous(),
                                          bias.contiguous(), params.contiguous())

    @staticmethod
    def backward(ctx, grad_out):
        feats, weight = ctx.saved_tensors
        # 用 CPU einsum 计算梯度（NPU einsum 对某些方程不兼容）
        # feats (N, K_flat) fp16, weight (K_flat, C_out) fp16
        gF = torch.einsum('no,ko->nk', grad_out.cpu(), weight.cpu().float()).to(grad_out.device)
        gW = torch.einsum('no,nk->ko', grad_out.cpu(), feats.cpu().float()).to(grad_out.device)
        gB = grad_out.sum(0)
        return gF, gW, gB, None


def _make_params(M, N, K, has_bias, block_num, device):
    return torch.tensor([M, N, K, int(has_bias), block_num],
                        dtype=torch.int32, device=device)


def spconv_gemm(feats, weight, bias):
    """AscendC 稀疏卷积 GEMM（gather 阶段已完成，Cube 加速）。

    Args:
        feats: (N, K_flat) float16, NPU
        weight: (K_flat, C_out) float16, NPU
        bias: (C_out,) float32, NPU

    Returns:
        out: (N, C_out) float32
    """
    N, K = feats.shape
    C_out = weight.shape[1]
    has_bias = bias.numel() == C_out
    params = _make_params(N, C_out, K, has_bias, _BLOCK_NUM, feats.device)
    available()
    return _SpconvGemmAscendC.apply(feats, weight, bias, params)


__all__ = ["spconv_gemm", "available", "_find_ops_lib"]