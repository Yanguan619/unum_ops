#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# 3-way 精度对比：v3 AscendC vs unum_ops AscendC vs torch scatter_add (bev_pool_torch)
#
# 用 BEVFusion 风格输入（C=80, 网格 ~200x200x41, 大量点）三方对比，
# 量化 "v3 atomic add" vs "unum_ops 排序后累加" vs "torch scatter_add" 的真实 max_diff。

import os
import sys
import numpy as np
import torch
import pytest

# Ensure torch.npu is initialized before importing the unum_ops libs.
try:
    import torch_npu  # noqa: F401
    NPU_AVAILABLE = torch.npu.is_available()
except Exception:
    NPU_AVAILABLE = False

# repo paths
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

if NPU_AVAILABLE:
    # 确保走 jit_compile=False 路径
    torch.npu.set_compile_mode(jit_compile=False)
    import unum_ops.bev_pool as um_bev
    import unum_ops.bev_pool_v3 as um_bev3


def make_bevfusion_input(N, B, D, H, W, C, seed=0, device="npu"):
    """BEVFusion 风格输入：随机点云，坐标均匀分布到网格。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    feats_cpu = torch.randn(N, C, dtype=torch.float32, generator=g)
    coords_cpu = torch.zeros(N, 4, dtype=torch.int32)  # kernel 要求 int32
    coords_cpu[:, 0] = torch.randint(0, W, (N,), generator=g, dtype=torch.int32)  # x
    coords_cpu[:, 1] = torch.randint(0, H, (N,), generator=g, dtype=torch.int32)  # y
    coords_cpu[:, 2] = torch.randint(0, D, (N,), generator=g, dtype=torch.int32)  # z
    coords_cpu[:, 3] = torch.randint(0, B, (N,), generator=g, dtype=torch.int32)  # batch
    if device == "npu":
        return feats_cpu.to("npu"), coords_cpu.to("npu")
    return feats_cpu, coords_cpu


def bev_pool_torch(feats, coords, B, D, H, W):
    """Atomic scatter-add reference (matches v3 AscendC summation order).

    使用 torch.index_add_ 而非 cumsum+diff — 顺序累加 vs atomic 在 fp32 下
    有 ~N*eps 量级差异，必须用 atomic reference 才能精确衡量 unum_ops
    AscendC 的真实精度损失。

    coords 约定（与 unum_ops.bev_pool 一致）：[:, 0]=x∈[0,W), [:, 1]=y∈[0,H),
    [:, 2]=z∈[0,D), [:, 3]=batch∈[0,B)。ranks = b*D*H*W + z*H*W + y*W + x。
    """
    coords_long = coords.long()
    linear_idx = (
        coords_long[:, 3] * (D * H * W)
        + coords_long[:, 2] * (H * W)
        + coords_long[:, 1] * W
        + coords_long[:, 0]
    )
    C = feats.shape[1]
    out_flat = torch.zeros(
        B * D * H * W, C, dtype=feats.dtype, device=feats.device
    )
    out_flat.index_add_(0, linear_idx, feats)
    return out_flat.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU required")
class TestBEVPoolV3Accuracy:
    """3-way 精度对比：unum_ops AscendC vs v3 AscendC vs torch scatter_add"""

    def test_bevfusion_shape_match(self):
        """三方输出形状必须一致（v3 内核直接做 5D 输出）"""
        feats, coords = make_bevfusion_input(200, B=2, D=41, H=200, W=200, C=80, seed=0)
        ref = bev_pool_torch(feats.cpu(), coords.cpu(), 2, 41, 200, 200)
        u = um_bev.bev_pool(feats, coords, 2, 41, 200, 200)
        v3 = um_bev3.bev_pool_v3(feats, coords, 2, 41, 200, 200)
        assert u.shape == ref.shape == v3.shape, (u.shape, ref.shape, v3.shape)

    def test_max_diff_unum_vs_torch(self):
        feats, coords = make_bevfusion_input(5000, B=2, D=41, H=200, W=200, C=80, seed=1)
        ref = bev_pool_torch(feats.cpu(), coords.cpu(), 2, 41, 200, 200).to("npu")
        u = um_bev.bev_pool(feats, coords, 2, 41, 200, 200).out
        d = (u.float() - ref.float()).abs()
        print(f"\n[unum_ops AscendC vs torch] max_diff={d.max().item():.3e}  mean_diff={d.mean().item():.3e}")

    def test_max_diff_v3_vs_torch(self):
        feats, coords = make_bevfusion_input(5000, B=2, D=41, H=200, W=200, C=80, seed=1)
        ref = bev_pool_torch(feats.cpu(), coords.cpu(), 2, 41, 200, 200).to("npu")
        v3 = um_bev3.bev_pool_v3(feats, coords, 2, 41, 200, 200).out
        d = (v3.float() - ref.float()).abs()
        print(f"\n[v3 AscendC vs torch]      max_diff={d.max().item():.3e}  mean_diff={d.mean().item():.3e}")

    def test_max_diff_v3_vs_unum(self):
        feats, coords = make_bevfusion_input(5000, B=2, D=41, H=200, W=200, C=80, seed=1)
        u = um_bev.bev_pool(feats, coords, 2, 41, 200, 200).out
        v3 = um_bev3.bev_pool_v3(feats, coords, 2, 41, 200, 200).out
        d = (v3.float() - u.float()).abs()
        print(f"\n[v3 AscendC vs unum_ops]   max_diff={d.max().item():.3e}  mean_diff={d.mean().item():.3e}")

    @pytest.mark.parametrize("N", [1000, 5000, 20000, 50000])
    def test_scale_max_diff(self, N):
        """不同 N 下的最大差异 — 验证 v3 的 atomic add 是否随 N 累积放大差异。"""
        feats, coords = make_bevfusion_input(N, B=2, D=41, H=200, W=200, C=80, seed=N)
        ref = bev_pool_torch(feats.cpu(), coords.cpu(), 2, 41, 200, 200).to("npu")
        u = um_bev.bev_pool(feats, coords, 2, 41, 200, 200).out
        v3 = um_bev3.bev_pool_v3(feats, coords, 2, 41, 200, 200).out
        d_u = (u.float() - ref.float()).abs()
        d_v3 = (v3.float() - ref.float()).abs()
        d_uv = (v3.float() - u.float()).abs()
        print(
            f"\n[N={N:>6}] torch_ref vs unum: max={d_u.max().item():.3e}  mean={d_u.mean().item():.3e}"
            f"  | torch_ref vs v3: max={d_v3.max().item():.3e}  mean={d_v3.mean().item():.3e}"
            f"  | v3 vs unum: max={d_uv.max().item():.3e}  mean={d_uv.mean().item():.3e}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))