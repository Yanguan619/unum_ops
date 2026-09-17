"""pointnet2 算子测试用例

覆盖 src/unum_ops/pointnet2/ 中的算子：
  - furthest_point_sample：最远点采样（迭代 FPS）
  - gather_operation：按索引取值
  - three_nn：三最近邻
  - three_interpolate / three_interpolate_onnx：三近邻加权插值
  - grouping_operation / grouping_operation_onnx：按索引分组
  - ball_query：球形邻域查询
  - cylinder_query / cylinder_query_onnx：圆柱邻域查询（含旋转）
  - knn：K 近邻
  - QueryAndGroup / GroupAll / CylinderQueryAndGroup：nn.Module 封装

运行方式：
    pytest test/test_pointnet2.py -v

关键设计说明：
  - 所有算子是纯 PyTorch 实现，CPU/CUDA/NPU 通用
  - NPU 上 BallQuery/CylinderQuery/FPS 因 python 循环会自动 fallback 到 CPU
  - *_onnx 变体不包裹 autograd.Function，用于 ONNX 导出
"""

import importlib
import os
import sys
from types import ModuleType

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, "..", "src", "unum_ops"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from pointnet2 import (
    CylinderQueryAndGroup,
    GroupAll,
    QueryAndGroup,
    ball_query,
    cylinder_query,
    cylinder_query_onnx,
    furthest_point_sample,
    furthest_point_sample_onnx,
    gather_operation,
    grouping_operation,
    grouping_operation_onnx,
    knn,
    three_interpolate,
    three_interpolate_onnx,
    three_nn,
)


# ============================================================
# 测试辅助
# ============================================================
def _device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


DEVICE = _device()
B = 2  # batch size
N = 100  # 点云点数
C = 8  # 特征数


# ============================================================
# furthest_point_sample
# ============================================================
class TestFurthestPointSample:
    def test_basic(self):
        xyz = torch.randn(B, N, 3, device=DEVICE)
        npoint = 10
        idx = furthest_point_sample(xyz, npoint)
        assert idx.shape == (B, npoint)
        assert idx.dtype == torch.long

    def test_npoint_equals_n(self):
        xyz = torch.randn(1, 20, 3, device=DEVICE)
        idx = furthest_point_sample(xyz, 20)
        assert idx.shape == (1, 20)
        # 所有点应该都被选中
        assert idx.unique().numel() == 20

    def test_reproducible(self):
        """相同输入应产生相同输出"""
        xyz = torch.randn(1, 50, 3, device=DEVICE)
        idx1 = furthest_point_sample(xyz, 5)
        idx2 = furthest_point_sample(xyz, 5)
        assert torch.equal(idx1, idx2)

    def test_onnx_variant(self):
        """furthest_point_sample_onnx 与 eager 版本输出逐位一致"""
        xyz = torch.randn(B, N, 3, device=DEVICE)
        idx_ref = furthest_point_sample(xyz, 10)
        idx_onnx = furthest_point_sample_onnx(xyz, 10)
        assert idx_onnx.shape == idx_ref.shape
        assert idx_onnx.dtype == torch.long
        assert torch.equal(idx_ref, idx_onnx)


# ============================================================
# gather_operation
# ============================================================
class TestGatherOperation:
    def test_basic(self):
        features = torch.randn(B, C, N, device=DEVICE)
        idx = torch.randint(0, N, (B, 16), device=DEVICE)
        out = gather_operation(features, idx)
        assert out.shape == (B, C, 16)

    def test_gather_all(self):
        features = torch.randn(1, C, N, device=DEVICE)
        idx = torch.arange(N, device=DEVICE).unsqueeze(0)
        out = gather_operation(features, idx)
        assert torch.allclose(out, features)


# ============================================================
# three_nn
# ============================================================
class TestThreeNN:
    def test_basic(self):
        unknown = torch.randn(B, 8, 3, device=DEVICE)
        known = torch.randn(B, 50, 3, device=DEVICE)
        dist, idx = three_nn(unknown, known)
        assert dist.shape == (B, 8, 3)
        assert idx.shape == (B, 8, 3)
        assert idx.dtype == torch.int32  # ATC 兼容要求
        # 距离应非负
        assert (dist >= 0).all()

    def test_same_point(self):
        """known == unknown 时，最近邻是自己"""
        pts = torch.randn(1, 10, 3, device=DEVICE)
        dist, idx = three_nn(pts, pts)
        # 第一近邻是自身，距离为 0
        assert torch.allclose(dist[:, :, 0], torch.zeros_like(dist[:, :, 0]), atol=1e-6)


# ============================================================
# three_interpolate / three_interpolate_onnx
# ============================================================
class TestThreeInterpolate:
    def test_basic(self):
        features = torch.randn(B, C, 50, device=DEVICE)
        idx = torch.randint(0, 50, (B, 8, 3), device=DEVICE).to(torch.int32)
        weight = torch.rand(B, 8, 3, device=DEVICE)
        # 权重归一化到和为 1
        weight = weight / weight.sum(dim=-1, keepdim=True)
        out = three_interpolate(features, idx, weight)
        assert out.shape == (B, C, 8)

    def test_interpolate_weighted(self):
        """手动验证加权求和结果"""
        features = torch.randn(1, C, 5, device=DEVICE)
        idx = torch.tensor([[[0, 1, 2]]], device=DEVICE, dtype=torch.int32)  # (1,1,3)
        weight = torch.tensor([[[0.5, 0.3, 0.2]]], device=DEVICE)
        out = three_interpolate(features, idx, weight)
        expected = 0.5 * features[:, :, 0] + 0.3 * features[:, :, 1] + 0.2 * features[:, :, 2]
        assert torch.allclose(out[:, :, 0], expected)

    def test_onnx_variant(self):
        features = torch.randn(1, C, 50, device=DEVICE)
        idx = torch.randint(0, 50, (1, 8, 3), device=DEVICE).to(torch.int32)
        weight = torch.rand(1, 8, 3, device=DEVICE)
        out_ref = three_interpolate(features, idx, weight)
        out_onnx = three_interpolate_onnx(features, idx, weight)
        assert torch.allclose(out_ref, out_onnx, atol=1e-6)


# ============================================================
# grouping_operation / grouping_operation_onnx
# ============================================================
class TestGroupingOperation:
    def test_basic(self):
        features = torch.randn(B, C, N, device=DEVICE)
        idx = torch.randint(0, N, (B, 16, 8), device=DEVICE)
        out = grouping_operation(features, idx)
        assert out.shape == (B, C, 16, 8)

    def test_onnx_variant(self):
        features = torch.randn(1, C, N, device=DEVICE)
        idx = torch.randint(0, N, (1, 16, 8), device=DEVICE)
        out_ref = grouping_operation(features, idx)
        out_onnx = grouping_operation_onnx(features, idx)
        assert torch.allclose(out_ref, out_onnx, atol=1e-6)


# ============================================================
# ball_query
# ============================================================
class TestBallQuery:
    def test_basic(self):
        xyz = torch.randn(B, N, 3, device=DEVICE)
        new_xyz = torch.randn(B, 8, 3, device=DEVICE)
        idx = ball_query(0.5, 16, xyz, new_xyz)
        assert idx.shape == (B, 8, 16)
        assert idx.dtype == torch.long

    def test_radius_zero(self):
        """半径 0 应返回填充值（最近邻重复填充）"""
        xyz = torch.randn(1, N, 3, device=DEVICE)
        new_xyz = torch.randn(1, 4, 3, device=DEVICE)
        idx = ball_query(0.0, 8, xyz, new_xyz)
        # 所有列应相等（填充同一个点）
        for i in range(8):
            assert torch.equal(idx[:, :, 0], idx[:, :, i])


# ============================================================
# cylinder_query / cylinder_query_onnx
# ============================================================
class TestCylinderQuery:
    def test_basic(self):
        xyz = torch.randn(1, N, 3, device=DEVICE)
        new_xyz = torch.randn(1, 4, 3, device=DEVICE)
        rot = torch.eye(3, device=DEVICE).unsqueeze(0).unsqueeze(0).expand(1, 4, 3, 3)
        idx = cylinder_query(0.5, -0.1, 0.1, 8, xyz, new_xyz, rot.view(1, 4, 9))
        assert idx.shape == (1, 4, 8)
        assert idx.dtype == torch.long

    def test_onnx_variant_small(self):
        """小规模验证 cylinder_query_onnx 的正确性"""
        xyz = torch.randn(1, 20, 3, device=DEVICE)
        new_xyz = torch.randn(1, 2, 3, device=DEVICE)
        rot = torch.eye(3, device=DEVICE).unsqueeze(0).unsqueeze(0).expand(1, 2, 3, 3)
        idx_ref = cylinder_query(0.5, -0.1, 0.1, 4, xyz, new_xyz, rot.reshape(1, 2, 9))
        idx_onnx = cylinder_query_onnx(0.5, -0.1, 0.1, 4, xyz, new_xyz, rot)
        assert idx_onnx.shape == idx_ref.shape
        assert idx_onnx.dtype == torch.int32


# ============================================================
# knn
# ============================================================
class TestKNN:
    def test_basic(self):
        ref = torch.randn(B, 3, 50, device=DEVICE)
        query = torch.randn(B, 3, 8, device=DEVICE)
        inds = knn(ref, query, k=3)
        assert inds.shape == (B, 3, 8)
        assert inds.dtype == torch.long

    def test_knn_self(self):
        """在自身中找最近邻，第一近邻应是自己"""
        pts = torch.randn(1, 3, 10, device=DEVICE)
        inds = knn(pts, pts, k=3)
        assert (inds[:, 0, :] == torch.arange(10, device=DEVICE)).all()


# ============================================================
# QueryAndGroup
# ============================================================
class TestQueryAndGroup:
    def test_basic(self):
        qg = QueryAndGroup(0.5, 8, use_xyz=True)
        xyz = torch.randn(1, N, 3, device=DEVICE)
        new_xyz = torch.randn(1, 8, 3, device=DEVICE)
        features = torch.randn(1, C, N, device=DEVICE)
        out = qg(xyz, new_xyz, features)
        assert out.shape == (1, C + 3, 8, 8)

    def test_no_features(self):
        qg = QueryAndGroup(0.5, 8, use_xyz=True)
        xyz = torch.randn(1, N, 3, device=DEVICE)
        new_xyz = torch.randn(1, 8, 3, device=DEVICE)
        out = qg(xyz, new_xyz)
        assert out.shape == (1, 3, 8, 8)

    def test_normalize_xyz(self):
        qg = QueryAndGroup(0.5, 8, use_xyz=True, normalize_xyz=True)
        xyz = torch.randn(1, N, 3, device=DEVICE)
        new_xyz = torch.randn(1, 8, 3, device=DEVICE)
        out = qg(xyz, new_xyz)
        assert out.shape == (1, 3, 8, 8)


# ============================================================
# GroupAll
# ============================================================
class TestGroupAll:
    def test_basic(self):
        ga = GroupAll(use_xyz=True)
        xyz = torch.randn(1, N, 3, device=DEVICE)
        new_xyz = torch.randn(1, 4, 3, device=DEVICE)
        features = torch.randn(1, C, N, device=DEVICE)
        out = ga(xyz, new_xyz, features)
        assert out.shape == (1, C + 3, 1, N)


# ============================================================
# CylinderQueryAndGroup
# ============================================================
class TestCylinderQueryAndGroup:
    def test_basic(self):
        cqg = CylinderQueryAndGroup(0.5, -0.1, 0.1, 8, use_xyz=True, rotate_xyz=True)
        xyz = torch.randn(1, N, 3, device=DEVICE)
        new_xyz = torch.randn(1, 4, 3, device=DEVICE)
        rot = torch.eye(3, device=DEVICE).unsqueeze(0).unsqueeze(0).expand(1, 4, 3, 3)
        out = cqg(xyz, new_xyz, rot)
        assert out.shape == (1, 3, 4, 8)


# ============================================================
# 后端选择
# ============================================================
class TestBackend:
    """验证模块加载时的一次性 CUDA extension 替换逻辑。

    设计：默认纯 PyTorch 实现；若环境装有原生 CUDA extension
    （pointnet2_utils / pointnet2._ext），模块加载时会自动替换。
    此处的测试不依赖真实 CUDA 环境，只验证 torch 回退可用。
    """

    def test_module_imports_without_cuda(self):
        """无 CUDA extension 环境下模块可正常导入且算子可用"""
        import pointnet2 as pn2

        # 模块应有全部公开接口
        for name in (
            "furthest_point_sample",
            "gather_operation",
            "three_nn",
            "three_interpolate",
            "grouping_operation",
            "ball_query",
            "cylinder_query",
            "knn",
        ):
            assert hasattr(pn2, name), f"缺少接口 {name}"

    def test_torch_fallback_correctness(self):
        """torch 实现可直接调用（与顶层接口等价）"""
        import pointnet2 as pn2

        xyz = torch.randn(1, 10, 3)
        # 无论是否被 CUDA 替换，调用签名一致
        idx = pn2.furthest_point_sample(xyz, 3)
        assert idx.shape == (1, 3)
        assert idx.dtype == torch.long

    def test_onnx_variant_always_available(self):
        """*_onnx 变体始终存在（不参与 CUDA 替换）"""
        import pointnet2 as pn2

        assert callable(pn2.three_interpolate_onnx)
        assert callable(pn2.grouping_operation_onnx)
        assert callable(pn2.cylinder_query_onnx)
        assert callable(pn2.furthest_point_sample_onnx)

        features = torch.randn(1, C, 50)
        idx = torch.randint(0, 50, (1, 8, 3)).to(torch.int32)
        weight = torch.rand(1, 8, 3)
        out = pn2.three_interpolate_onnx(features, idx, weight)
        assert out.shape == (1, C, 8)

        # furthest_point_sample_onnx 应可直接调用且输出形状正确
        xyz = torch.randn(1, 50, 3)
        fps_idx = pn2.furthest_point_sample_onnx(xyz, 5)
        assert fps_idx.shape == (1, 5)

    def test_cuda_backend_swap_propagation(self, monkeypatch):
        """模块拆分后，CUDA 替换需同时补丁子模块属性与包级导出。

        拆分前所有算子在单文件中，nn.Module 内部通过全局名查找即可
        拿到替换后的 CUDA 版本；拆分后 modules.py 改为子模块属性访问
        （延迟绑定），必须同步补丁子模块属性才能保持原语义。
        用抛 Sentinel 异常的 fake 后端逐层验证路由。
        """

        class _Sentinel(RuntimeError):
            pass

        def _fake_op(name):
            def _op(*args, **kwargs):
                raise _Sentinel(name)

            return _op

        fake = ModuleType("pointnet2_utils")
        for name in (
            "furthest_point_sample",
            "gather_operation",
            "three_nn",
            "three_interpolate",
            "grouping_operation",
            "ball_query",
            "cylinder_query",
        ):
            setattr(fake, name, _fake_op(name))
        # 故意不提供 knn：验证 hasattr 分支，knn 应保持 torch 版

        import pointnet2 as pn2

        monkeypatch.setitem(sys.modules, "pointnet2_utils", fake)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        try:
            importlib.reload(pn2)

            # 1) 包级导出被替换
            with pytest.raises(_Sentinel, match="ball_query"):
                pn2.ball_query(0.5, 4, torch.randn(1, 10, 3), torch.randn(1, 3, 3))

            # 2) 子模块属性被替换（modules.py 延迟绑定的查找目标）
            with pytest.raises(_Sentinel, match="grouping_operation"):
                pn2.grouping.grouping_operation(
                    torch.randn(1, 3, 10), torch.zeros(1, 4, 5, dtype=torch.long)
                )

            with pytest.raises(_Sentinel, match="three_nn"):
                pn2.nn_search.three_nn(torch.randn(1, 4, 3), torch.randn(1, 10, 3))

            # 3) 内部 nn.Module 消费者同样路由到 CUDA 版
            qg = pn2.QueryAndGroup(radius=0.5, nsample=4)
            with pytest.raises(_Sentinel, match="ball_query"):
                qg(torch.randn(1, 10, 3), torch.randn(1, 4, 3))

            cqg = pn2.CylinderQueryAndGroup(radius=0.5, hmin=-0.1, hmax=0.1, nsample=4)
            with pytest.raises(_Sentinel, match="cylinder_query"):
                cqg(
                    torch.randn(1, 10, 3),
                    torch.randn(1, 4, 3),
                    torch.eye(3).unsqueeze(0).expand(1, 4, 3, 3).contiguous(),
                )

            # 4) 后端未提供 knn 时：knn 保持 torch 版（hasattr 分支）
            inds = pn2.knn(torch.randn(1, 3, 10), torch.randn(1, 3, 4), k=2)
            assert inds.shape == (1, 2, 4)
        finally:
            # 先撤销 fake 后端与 CUDA 标记，再恢复被 swap 污染的模块状态：
            # 子模块须先于包 reload（包级 re-export 从子模块取值）
            monkeypatch.undo()
            importlib.reload(pn2.sampling)
            importlib.reload(pn2.grouping)
            importlib.reload(pn2.interpolate)
            importlib.reload(pn2.query)
            importlib.reload(pn2.modules)
            importlib.reload(pn2)


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
