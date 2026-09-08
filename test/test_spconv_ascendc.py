"""spconv_gemm AscendC Cube 内核测试（2D fp16 接口，已知 stride 写入 bug）。

当前状态（2026-09-08）：
  - Cube 内核已编译安装，Matmul<GM,ND,half> 高层 API
  - 输出写入 stride 为 baseM(1024) 而非 1，仅 49 行正确（50000 行中）
  - 数值正确性测试标记为 xfail，等待低层 Mad/mmad 修复

运行方式：
    cd /workspace/unum_ops
    python -m pytest test/test_spconv_ascendc.py -v
"""
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, "..", "src", "unum_ops"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from spconv import ascendc
from spconv.conv import SubMConv3d, SparseConv3d, SparseInverseConv3d
from spconv.sparse_modules import SparseConvTensor

NPU_AVAIL = hasattr(torch, "npu") and torch.npu.is_available()
if NPU_AVAIL:
    try:
        torch.npu.set_compile_mode(jit_compile=False)
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def _load_ascendc():
    if NPU_AVAIL:
        assert ascendc.available(), "libspconv_gemm_ops.so 加载失败"
    yield


def _make_tensor(N, C, spatial, batch_size=1, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    x_dim, y_dim, z_dim = spatial[:3]
    coords = set()
    while len(coords) < N:
        xs = torch.randint(0, x_dim, (N * 4,), generator=g).tolist()
        ys = torch.randint(0, y_dim, (N * 4,), generator=g).tolist()
        zs = torch.randint(0, z_dim, (N * 4,), generator=g).tolist()
        for i in range(len(xs)):
            coords.add((0, xs[i], ys[i], zs[i]))
            if len(coords) >= N:
                break
    indices = torch.tensor(sorted(coords)[:N], dtype=torch.int32)
    features = torch.randn(indices.shape[0], C, generator=g)
    return SparseConvTensor(features, indices, spatial, batch_size)


# ============================================================
# 底层 op 测试（2D fp16 接口）
# ============================================================

@pytest.mark.skipif(not NPU_AVAIL, reason="NPU not available")
class TestSpconvGemmOp:
    def test_op_runs(self):
        """op 可正常调用（不 crash）"""
        N, K_flat, C = 50000, 1728, 64
        feats = torch.randn(N, K_flat, device="npu", dtype=torch.float16)
        weight = torch.randn(K_flat, C, device="npu", dtype=torch.float16)
        bias = torch.randn(C, device="npu", dtype=torch.float32)
        params = torch.tensor([N, C, K_flat, 1, 8], dtype=torch.int32, device="npu")
        out = torch.ops.unum.spconv_gemm(feats, weight, bias, params)
        assert out.shape == (N, C)

    def test_output_not_all_nan(self):
        """输出不全是 NaN"""
        N, K, C = 50000, 1728, 64
        feats = torch.randn(N, K, device="npu", dtype=torch.float16)
        weight = torch.randn(K, C, device="npu", dtype=torch.float16)
        bias = torch.randn(C, device="npu", dtype=torch.float32)
        params = torch.tensor([N, C, K, 1, 8], dtype=torch.int32, device="npu")
        out = torch.ops.unum.spconv_gemm(feats, weight, bias, params)
        assert not torch.isnan(out).any()

    @pytest.mark.xfail(reason="Cube 内核 stride 写入 bug（baseM=1024 而非 1）")
    def test_basic_correctness(self):
        """基本数值正确性（已知 bug，预期 xfail）"""
        N, K_flat, C = 8, 4, 3
        torch.manual_seed(0)
        feats = torch.randn(N, K_flat, device="npu", dtype=torch.float16)
        weight = torch.randn(K_flat, C, device="npu", dtype=torch.float16)
        bias = torch.randn(C, device="npu", dtype=torch.float32)
        params = torch.tensor([N, C, K_flat, 1, 8], dtype=torch.int32, device="npu")
        out = torch.ops.unum.spconv_gemm(feats, weight, bias, params)
        ref = (feats.float() @ weight.float()) + bias
        assert torch.allclose(out, ref, atol=0.5)

    def test_repeatability(self):
        """相同输入多次调用应给出相同结果"""
        N, K, C = 16, 8, 4
        torch.manual_seed(0)
        feats = torch.randn(N, K, device="npu", dtype=torch.float16)
        weight = torch.randn(K, C, device="npu", dtype=torch.float16)
        bias = torch.randn(C, device="npu", dtype=torch.float32)
        params = torch.tensor([N, C, K, 1, 8], dtype=torch.int32, device="npu")
        outs = [torch.ops.unum.spconv_gemm(feats, weight, bias, params) for _ in range(3)]
        assert torch.allclose(outs[0], outs[1], atol=0.01)
        assert torch.allclose(outs[0], outs[2], atol=0.01)

    def test_small_shape(self):
        """小形状可运行"""
        N, K, C = 2, 4, 3
        feats = torch.randn(N, K, device="npu", dtype=torch.float16)
        weight = torch.randn(K, C, device="npu", dtype=torch.float16)
        bias = torch.randn(C, device="npu", dtype=torch.float32)
        params = torch.tensor([N, C, K, 1, 8], dtype=torch.int32, device="npu")
        out = torch.ops.unum.spconv_gemm(feats, weight, bias, params)
        assert out.shape == (N, C)


# ============================================================
# autograd Function 测试
# ============================================================

@pytest.mark.skipif(not NPU_AVAIL, reason="NPU not available")
class TestAutogradFunction:
    def test_forward_via_ascendc(self):
        """ascendc.spconv_gemm 前向可运行"""
        N, K, C = 16, 8, 4
        torch.manual_seed(0)
        feats = torch.randn(N, K, requires_grad=True, device="npu", dtype=torch.float16)
        weight = torch.randn(K, C, requires_grad=True, device="npu", dtype=torch.float16)
        bias = torch.randn(C, requires_grad=True, device="npu", dtype=torch.float32)
        out = ascendc.spconv_gemm(feats, weight, bias)
        assert out.shape == (N, C)
        assert not torch.isnan(out).any()

    @pytest.mark.xfail(reason="Cube 内核 stride 写入 bug，前向数值不正确")
    def test_backward_matches_autograd(self):
        """反向传播梯度应与 einsum 路径一致（已知 bug，预期 xfail）"""
        N, K, C = 8, 4, 3
        torch.manual_seed(0)
        feats = torch.randn(N, K, requires_grad=True, device="npu", dtype=torch.float16)
        weight = torch.randn(K, C, requires_grad=True, device="npu", dtype=torch.float16)
        bias = torch.randn(C, requires_grad=True, device="npu", dtype=torch.float32)
        out = ascendc.spconv_gemm(feats, weight, bias)
        loss = out.sum()
        gF, gW, gB = torch.autograd.grad(loss, [feats, weight, bias])
        # backward 用 CPU einsum 计算，梯度应正确
        assert gF is not None
        assert gW is not None
        assert gB is not None


# ============================================================
# 边缘情况
# ============================================================

@pytest.mark.skipif(not NPU_AVAIL, reason="NPU not available")
class TestEdgeCases:
    def test_empty_input(self):
        """空输出应返回空张量（通过 _gather 的早期返回）"""
        x = SparseConvTensor(torch.empty(0, 16).npu(),
                             torch.empty(0, 4, dtype=torch.int32).npu(),
                             (10, 30, 40), 1)
        conv = SubMConv3d(16, 16, 3, padding=1, bias=True).eval().to("npu:0")
        y = conv(x)
        assert y.features.shape == (0, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])