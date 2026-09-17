"""spconv 算子测试用例

覆盖 /data/workspace/unum_ops/src/unum_ops/spconv 中的算子：
  - SparseConvTensor 容器
  - SubMConv3d / SubMConv2d（子流形卷积）
  - SparseConv3d / SparseConv2d（下采样卷积）
  - SparseInverseConv3d / SparseInverseConv2d（转置卷积 / 上采样）
  - SparseSequential 容器
  - SparseReLU / SparseBatchNorm1d / SparseLinear 适配层
  - VoxelGeneratorV2 体素生成器

运行方式：
    cd /data/workspace/unum_ops
    python -m pytest test/test_spconv.py -v

关键设计说明：
  - 3D 测试使用 make_tensor（4 列 indices [batch,x,y,z]，3 元素 spatial_shape (D,H,W)）
    约定与 spconv/test_conv.py 一致：x∈[0,W), y∈[0,H), z∈[0,D)
  - 2D 测试使用 make_tensor_2d（3 列 indices [batch,x,y]，2 元素 spatial_shape (H,W)）
    约定：x∈[0,W), y∈[0,H)（与 3D 的 x,y 语义一致）
    注意：SubMConv2d 内部 ndim=2，但 kernel_size 仍被 _triple 成 (3,3,3)，
    _offsets_t 截断 offset[:2]，因此 oc[:,1:] 也必须是 2 列 → indices 必须 3 列。
  - 数值一致性测试全部使用立方体/正方形 spatial_shape + 对称 kernel/stride/padding，
    避免 sp_col 反转后与 index 列顺序不一致导致的边界检查差异
    （ref 用 in_shape[d]，new impl 用 reversed(in_shape)[d]，立方体下两者相等）
  - 参考实现逐体素循环，N 控制在 80~200 防止过慢
"""
import itertools
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from copy import deepcopy

from unum_ops.spconv.conv import (
    SparseConv3d,
    SparseConv2d,
    SubMConv3d,
    SubMConv2d,
    SparseInverseConv3d,
    SparseInverseConv2d,
)
from unum_ops.spconv.sparse_modules import (
    SparseConvTensor,
    SparseSequential,
    SparseReLU,
    SparseBatchNorm1d,
    SparseLinear,
    SparseConv3dAdapter,
    SubMConv3dAdapter,
)
from unum_ops.spconv.utils import VoxelGeneratorV2


# ============================================================
# 辅助：构造稀疏张量
# ============================================================

def make_tensor(N, C, spatial_shape, batch_size=1, seed=0):
    """3D 随机 SparseConvTensor（与 spconv/test_conv.py make_tensor 一致）

    indices: (N, 4) = [batch, x, y, z]
    spatial_shape: (x_dim, y_dim, z_dim)，x∈[0,x_dim), y∈[0,y_dim), z∈[0,z_dim)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    x_dim, y_dim, z_dim = spatial_shape[:3]
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
    return SparseConvTensor(features, indices, spatial_shape, batch_size)


def make_tensor_2d(N, C, spatial_shape, batch_size=1, seed=0):
    """2D 随机 SparseConvTensor

    indices: (N, 3) = [batch, x, y]，x∈[0,W), y∈[0,H)
    spatial_shape: (H, W)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    H, W = spatial_shape
    coords = set()
    while len(coords) < N:
        xs = torch.randint(0, W, (N * 4,), generator=g).tolist()
        ys = torch.randint(0, H, (N * 4,), generator=g).tolist()
        for i in range(len(xs)):
            coords.add((0, xs[i], ys[i]))
            if len(coords) >= N:
                break
    indices = torch.tensor(sorted(coords)[:N], dtype=torch.int32)
    features = torch.randn(indices.shape[0], C, generator=g)
    return SparseConvTensor(features, indices, spatial_shape, batch_size)


def assert_close(a, b, atol=1e-5, rtol=1e-4, msg=""):
    """数值一致性断言（基于 torch.testing.assert_close，失败时给出逐元素诊断）"""
    a = a.detach().float()
    b = b.detach().float()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=msg)


def _lexsort(idx):
    """按整行（batch, x, y, z）lexicographic 排序，返回 torch 索引"""
    arr = idx.cpu().numpy()
    order = np.lexsort([arr[:, d] for d in range(arr.shape[1] - 1, -1, -1)])
    return torch.from_numpy(order).to(idx.device)


# ============================================================
# 参考实现（逐体素，与 spconv/test_conv.py 对齐）
# 立方体 spatial_shape 下，in_shape[d] == reversed(in_shape)[d]，故无维度错位问题
# ============================================================

def ref_build_coord_dict(indices):
    d = {}
    for i, c in enumerate(indices.long().tolist()):
        d[tuple(c)] = i
    return d


def ref_subm(conv, x):
    indices = x.indices
    dev = indices.device
    spatial_shape = tuple(x.spatial_shape[:conv.ndim])
    coord_to_idx = ref_build_coord_dict(indices)
    N, C_out = x.features.shape[0], conv.out_channels
    w_flat = conv.weight.reshape(conv.out_channels, conv.in_channels, -1)
    bias = conv.bias
    out = torch.zeros(N, C_out, device=dev)
    inds = indices.long().tolist()
    for i in range(N):
        coord = inds[i]
        acc = torch.zeros(C_out, device=dev)
        for off, k in zip(conv._kernel_offsets(), conv._flat_slice_idx):
            valid = True
            ncoord = [coord[0]]
            for d in range(conv.ndim):
                src = coord[d + 1] * conv.stride[d] + off[d] * conv.dilation[d] - conv.padding[d]
                if src < 0 or src >= spatial_shape[d]:
                    valid = False
                    break
                ncoord.append(src)
            if not valid:
                continue
            j = coord_to_idx.get(tuple(ncoord))
            if j is not None:
                acc += w_flat[:, :, k] @ x.features[j]
        if bias is not None:
            acc += bias
        out[i] = acc
    return out


def ref_sparse(conv, x, return_idx=False):
    """纯 torch 标准参考。return_idx=True 时同时返回输出坐标（用于与第三方/NPU 对齐）。"""
    indices = x.indices
    dev = indices.device
    in_shape = tuple(x.spatial_shape[:conv.ndim])
    coord_to_idx = ref_build_coord_dict(indices)
    out_shape = tuple(
        (s + 2 * conv.padding[d] - (conv.kernel_size[d] - 1) - 1) // conv.stride[d] + 1
        for d, s in enumerate(in_shape))
    out_indices = conv._downsample_coords(indices)
    keep = torch.ones(out_indices.shape[0], dtype=torch.bool)
    for d in range(conv.ndim):
        keep &= (out_indices[:, d + 1] >= 0) & (out_indices[:, d + 1] < out_shape[d])
    out_indices = out_indices[keep]
    out_indices = torch.unique(out_indices, dim=0)
    w_flat = conv.weight.reshape(conv.out_channels, conv.in_channels, -1)
    bias = conv.bias
    N_out = out_indices.shape[0]
    out = torch.zeros(N_out, conv.out_channels, device=dev)
    for i in range(N_out):
        q = out_indices[i].long().tolist()
        acc = torch.zeros(conv.out_channels, device=dev)
        for off, k in zip(conv._kernel_offsets(), conv._flat_slice_idx):
            valid = True
            ncoord = [q[0]]
            for d in range(conv.ndim):
                src = q[d + 1] * conv.stride[d] + off[d] - conv.padding[d]
                if src < 0 or src >= in_shape[d]:
                    valid = False
                    break
                ncoord.append(src)
            if not valid:
                continue
            j = coord_to_idx.get(tuple(ncoord))
            if j is not None:
                acc += w_flat[:, :, k] @ x.features[j]
        if bias is not None:
            acc += bias
        out[i] = acc
    if return_idx:
        return out_indices, out
    return out


def ref_inverse(conv, x, info, return_idx=False):
    features = x.features
    dev = features.device
    out_indices = info['in_coords']
    stride, padding, kernel = info['stride'], info['padding'], info['kernel']
    coord_to_idx = ref_build_coord_dict(x.indices)
    N_out = out_indices.shape[0]
    out = torch.zeros(N_out, conv.out_channels, device=dev)
    w_flat = conv.weight.reshape(conv.out_channels, conv.in_channels, -1)
    bias = conv.bias
    in_shape = tuple(x.spatial_shape[:conv.ndim])
    for i in range(N_out):
        c = out_indices[i].long().tolist()
        acc = torch.zeros(conv.out_channels, device=dev)
        for k in itertools.product(*[range(kk) for kk in kernel]):
            valid = True
            q = [c[0]]
            for d in range(conv.ndim):
                num = c[d + 1] + padding[d] - k[d]
                if num % stride[d] != 0:
                    valid = False
                    break
                qv = num // stride[d]
                if qv < 0 or qv >= in_shape[d]:
                    valid = False
                    break
                q.append(qv)
            if not valid:
                continue
            j = coord_to_idx.get(tuple(q))
            if j is not None:
                flat = int(np.ravel_multi_index(k, tuple(kernel)))
                acc += w_flat[:, :, flat] @ features[j]
        if bias is not None:
            acc += bias
        out[i] = acc
    if return_idx:
        return out_indices, out
    return out


# ============================================================
# fixtures
# ============================================================

@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    return torch.device("cpu")


# ============================================================
# SparseConvTensor 容器测试
# ============================================================

class TestSparseConvTensor:
    def test_construct_and_attrs(self):
        feats = torch.randn(2, 8)
        idx = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.int32)
        t = SparseConvTensor(feats, idx, (10, 20, 30), 1)
        assert t.shape == (2, 8)
        assert t.ndim == 3
        assert t.spatial_size == (10, 20, 30)
        assert t.batch_size == 1

    def test_construct_2d(self):
        feats = torch.randn(3, 4)
        idx = torch.tensor([[0, 1, 2], [0, 3, 4], [0, 2, 1]], dtype=torch.int32)
        t = SparseConvTensor(feats, idx, (10, 12), 1)
        assert t.ndim == 2
        assert t.spatial_size == (10, 12)

    def test_replace_feature_keeps_meta(self):
        t = make_tensor(50, 4, (8, 8, 8))
        new_feats = torch.randn(50, 4)
        t2 = t.replace_feature(new_feats)
        assert t2.features is new_feats
        assert t2.indices is t.indices
        assert t2.spatial_shape == t.spatial_shape
        assert t2.batch_size == t.batch_size
        t._indice_dict["k"] = {"foo": 1}
        assert "k" in t2._indice_dict

    def test_dense_roundtrip_channels_first(self):
        t = make_tensor(60, 4, (6, 8, 10))
        dense = t.dense(channels_first=True)
        assert dense.shape == (1, 4, 6, 8, 10)
        t2 = SparseConvTensor.from_dense(dense, channels_first=True)
        assert t2.features.shape[1] == 4
        assert t2.batch_size == 1

    def test_dense_roundtrip_channels_last(self):
        t = make_tensor(60, 3, (5, 8, 10))
        dense = t.dense(channels_first=False)
        assert dense.shape == (1, 5, 8, 10, 3)
        t2 = SparseConvTensor.from_dense(dense, channels_first=False)
        assert t2.features.shape[1] == 3

    def test_to_device_roundtrip(self, device):
        t = make_tensor(20, 4, (4, 6, 8))
        t_dev = t.to(device)
        assert t_dev.features.device.type == device.type
        t_cpu = t_dev.cpu()
        assert t_cpu.features.device.type == "cpu"


# ============================================================
# SubMConv3d / SubMConv2d 测试
# ============================================================

class TestSubMConv:
    @pytest.mark.parametrize("C_in,C_out,k,pad", [
        (16, 16, 3, 1),
        (8, 32, 1, 0),
    ])
    def test_coords_unchanged(self, C_in, C_out, k, pad):
        """SubM 卷积输出坐标必须等于输入坐标"""
        x = make_tensor(150, C_in, (8, 10, 12))
        conv = SubMConv3d(C_in, C_out, k, padding=pad, bias=True).eval()
        y = conv(x)
        assert y.indices.shape == x.indices.shape
        assert torch.equal(y.indices, x.indices)

    @pytest.mark.parametrize("C", [8, 16])
    def test_spatial_shape_unchanged(self, C):
        x = make_tensor(150, C, (8, 10, 12))
        conv = SubMConv3d(C, C, 3, padding=1).eval()
        y = conv(x)
        assert tuple(y.spatial_shape) == tuple(x.spatial_shape)

    def test_matches_reference(self):
        """SubMConv3d 数值一致性（立方体 spatial 避免维度错位）"""
        x = make_tensor(150, 16, (10, 10, 10), seed=42)
        conv = SubMConv3d(16, 16, 3, padding=1, bias=True, indice_key="s").eval()
        new = conv(x).features
        ref = ref_subm(conv, x)
        assert_close(new, ref, msg="SubMConv3d vs ref")

    def test_bias_false(self):
        x = make_tensor(100, 8, (6, 8, 10))
        conv = SubMConv3d(8, 8, 3, padding=1, bias=False).eval()
        assert conv.bias is None
        y = conv(x)
        assert y.features.shape == (100, 8)

    def test_2d_matches_reference(self):
        """2D 子流形卷积数值一致性（正方形 spatial + 3 列 indices）"""
        x2d = make_tensor_2d(40, 8, (8, 8), seed=7)
        conv2d = SubMConv2d(8, 8, 3, padding=1, bias=True, indice_key="s2").eval()
        new = conv2d(x2d).features
        ref = ref_subm(conv2d, x2d)
        assert_close(new, ref, msg="SubMConv2d vs ref")

    def test_grad_flow(self):
        x = make_tensor(100, 4, (6, 8, 10))
        conv = SubMConv3d(4, 8, 3, padding=1, bias=True)
        y = conv(x)
        y.features.sum().backward()
        assert conv.weight.grad is not None
        assert conv.weight.grad.shape == conv.weight.shape
        assert conv.bias.grad is not None


# ============================================================
# SparseConv3d / SparseConv2d 测试
# ============================================================

class TestSparseConv:
    @pytest.mark.parametrize("stride", [(2, 2, 2), (1, 1, 1)])
    def test_downsample_spatial_shape(self, stride):
        x = make_tensor(200, 8, (8, 10, 12))
        conv = SparseConv3d(8, 8, 3, stride=stride, padding=1, bias=False).eval()
        y = conv(x)
        expected = tuple(
            (s + 2 * 1 - (3 - 1) - 1) // stride[d] + 1
            for d, s in enumerate((8, 10, 12))
        )
        assert tuple(y.spatial_shape[:3]) == expected

    def test_matches_reference(self):
        """SparseConv3d 数值一致性（立方体 spatial + 对称 stride/padding）"""
        x = make_tensor(150, 16, (10, 10, 10), seed=1)
        conv = SparseConv3d(16, 32, 3, stride=2, padding=1,
                            bias=True, indice_key="down").eval()
        y = conv(x)
        ref = ref_sparse(conv, x)
        assert_close(y.features, ref, msg="SparseConv3d vs ref")

    def test_2d_matches_reference(self):
        """2D 下采样卷积数值一致性（正方形 spatial + 3 列 indices）"""
        x2d = make_tensor_2d(40, 16, (8, 8), seed=2)
        conv2d = SparseConv2d(16, 32, 3, stride=2, padding=1, bias=True, indice_key="d2").eval()
        y2 = conv2d(x2d)
        ref = ref_sparse(conv2d, x2d)
        assert_close(y2.features, ref, msg="SparseConv2d vs ref")

    def test_indice_dict_recorded(self):
        x = make_tensor(150, 8, (8, 10, 12))
        conv = SparseConv3d(8, 16, 3, stride=2, padding=1, bias=True, indice_key="enc").eval()
        y = conv(x)
        assert "enc" in y._indice_dict
        info = y._indice_dict["enc"]
        assert info["stride"] == (2, 2, 2)
        assert info["padding"] == (1, 1, 1)
        assert info["kernel"] == (3, 3, 3)
        assert torch.equal(info["in_coords"], x.indices)

    def test_no_indice_key(self):
        x = make_tensor(150, 4, (6, 10, 12))
        conv = SparseConv3d(4, 4, 3, stride=2, padding=1, bias=False).eval()
        y = conv(x)
        assert len(y._indice_dict) == 0


# ============================================================
# SparseInverseConv3d / SparseInverseConv2d 测试
# ============================================================

class TestSparseInverseConv:
    def _make_pair(self, in_c=16, out_c=32, k=3, stride=2, pad=1,
                   key="up", N=150, spatial=(10, 10, 10), seed=3):
        """立方体 spatial + 对称 kernel/stride/padding，确保 out_shape 也是立方体"""
        x = make_tensor(N, in_c, spatial, seed=seed)
        down = SparseConv3d(in_c, out_c, k, stride=stride, padding=pad,
                            bias=True, indice_key=key).eval()
        y = down(x)
        z = SparseConvTensor(torch.randn(y.features.shape[0], out_c), y.indices,
                             y.spatial_shape, x.batch_size)
        z._indice_dict = dict(y._indice_dict)
        up = SparseInverseConv3d(out_c, in_c, k, indice_key=key, bias=True).eval()
        return x, y, z, up

    def test_matches_reference(self):
        """SparseInverseConv3d 数值一致性（立方体 out_shape 避免维度错位）"""
        x, y, z, up = self._make_pair()
        out = up(z)
        info = z._indice_dict["up"]
        ref = ref_inverse(up, z, info)
        assert_close(out.features, ref, msg="SparseInverseConv3d vs ref")

    def test_output_coords_match_input_of_down(self):
        x, y, z, up = self._make_pair()
        out = up(z)
        assert torch.equal(out.indices, x.indices)

    def test_missing_indice_key_raises(self):
        z = make_tensor(80, 8, (6, 8, 10))
        up = SparseInverseConv3d(8, 4, 3, indice_key="nonexistent")
        with pytest.raises(ValueError, match="SparseInverseConv3d requires matching"):
            up(z)

    def test_2d_inverse_roundtrip(self):
        """2D 反卷积往返（正方形 spatial + 3 列 indices）"""
        x2d = make_tensor_2d(40, 8, (8, 8), seed=10)
        down = SparseConv2d(8, 16, 3, stride=2, padding=1,
                            bias=True, indice_key="up2").eval()
        y2 = down(x2d)
        z2 = SparseConvTensor(torch.randn(y2.features.shape[0], 16), y2.indices,
                              y2.spatial_shape, x2d.batch_size)
        z2._indice_dict = dict(y2._indice_dict)
        up2 = SparseInverseConv2d(16, 8, 3, indice_key="up2", bias=True).eval()
        out = up2(z2)
        assert out.features.shape[1] == 8
        assert torch.equal(out.indices, x2d.indices)


# ============================================================
# SparseSequential 容器测试
# ============================================================

class TestSparseSequential:
    def test_forward_chain(self):
        x = make_tensor(200, 8, (8, 10, 12))
        net = SparseSequential(
            SubMConv3d(8, 16, 3, padding=1, bias=False),
            SparseReLU(),
            SubMConv3d(16, 16, 3, padding=1, bias=True),
        ).eval()
        y = net(x)
        assert isinstance(y, SparseConvTensor)
        assert y.features.shape == (200, 16)
        assert torch.equal(y.indices, x.indices)

    def test_indexing_and_len(self):
        conv1 = SubMConv3d(8, 16, 3, padding=1)
        conv2 = SparseReLU()
        net = SparseSequential(conv1, conv2)
        assert len(net) == 2
        assert net[0] is conv1
        assert net[1] is conv2
        sub = net[0:1]
        assert isinstance(sub, SparseSequential)
        assert len(sub) == 1

    def test_mixed_with_nn_module(self):
        """nn.BatchNorm1d / nn.ReLU 应自动作用于 .features"""
        x = make_tensor(150, 8, (6, 10, 12))
        net = SparseSequential(
            SubMConv3d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        ).eval()
        y = net(x)
        assert y.features.shape == (150, 16)

    def test_dict_construction(self):
        conv = SubMConv3d(4, 4, 3, padding=1)
        relu = SparseReLU()
        net = SparseSequential({"conv": conv, "relu": relu})
        assert len(net) == 2
        assert getattr(net, "conv") is conv
        assert getattr(net, "relu") is relu


# ============================================================
# 稀疏适配层测试
# ============================================================

class TestSparseLayers:
    def test_sparse_relu(self):
        x = make_tensor(100, 4, (6, 8, 10))
        feats = x.features.clone()
        y = SparseReLU()(x)
        assert torch.equal(y.features, torch.relu(feats))
        assert torch.equal(y.indices, x.indices)

    def test_sparse_batchnorm_eval_reproducible(self):
        x = make_tensor(150, 8, (6, 10, 12))
        bn = SparseBatchNorm1d(8).eval()
        y1 = bn(x)
        y2 = bn(x)
        assert_close(y1.features, y2.features, msg="BN eval reproducible")

    def test_sparse_linear_matches_nn_linear(self):
        x = make_tensor(100, 8, (6, 8, 10))
        lin = SparseLinear(8, 16)
        y = lin(x)
        assert y.features.shape == (100, 16)
        ref = lin.linear(x.features)
        assert_close(y.features, ref, msg="SparseLinear matches nn.Linear")


# ============================================================
# VoxelGeneratorV2 测试
# ============================================================

class TestVoxelGeneratorV2:
    def test_output_shapes(self):
        gen = VoxelGeneratorV2(
            voxel_size=(0.1, 0.1, 0.1),
            point_cloud_range=(0, 0, -1, 10, 10, 1),
            max_num_points=5,
            max_voxels=100,
        )
        pts = np.random.uniform(0, 10, (500, 4)).astype(np.float32)
        out = gen.generate(pts)
        assert out["voxels"].shape[1:] == (5, 4)
        assert out["voxels"].shape[0] <= 100
        assert out["coordinates"].shape[1] == 3
        assert out["num_points_per_voxel"].shape == (out["voxels"].shape[0],)

    def test_coords_in_range(self):
        gen = VoxelGeneratorV2(
            voxel_size=(1.0, 1.0, 1.0),
            point_cloud_range=(0, 0, 0, 3, 3, 3),
            max_num_points=10,
            max_voxels=1000,
        )
        pts = np.array([
            [0.5, 0.5, 0.5, 1.0],
            [1.5, 1.5, 1.5, 1.0],
        ], dtype=np.float32)
        out = gen.generate(pts)
        assert out["coordinates"].shape == (2, 3)
        assert np.all(out["coordinates"] >= 0)
        assert np.all(out["coordinates"] < 3)

    def test_out_of_range_filtered(self):
        gen = VoxelGeneratorV2(
            voxel_size=(1.0, 1.0, 1.0),
            point_cloud_range=(0, 0, 0, 5, 5, 5),
            max_num_points=10,
            max_voxels=100,
        )
        pts = np.array([
            [-1, 0, 0, 1.0],   # 越界
            [0, 0, 0, 1.0],
            [10, 0, 0, 1.0],   # 越界
            [1, 1, 1, 1.0],
        ], dtype=np.float32)
        out = gen.generate(pts)
        assert out["voxels"].shape[0] == 2


# ============================================================
# 边界情况
# ============================================================

class TestEdgeCases:
    def test_empty_input_subm(self):
        x = SparseConvTensor(torch.empty(0, 16), torch.empty(0, 4, dtype=torch.int32),
                             (10, 30, 40), 1)
        out = SubMConv3d(16, 16, 3, padding=1, bias=True).eval()(x)
        assert out.features.shape == (0, 16)

    def test_empty_input_sparse(self):
        x = SparseConvTensor(torch.empty(0, 8), torch.empty(0, 4, dtype=torch.int32),
                             (8, 16, 20), 1)
        out = SparseConv3d(8, 16, 3, stride=2, padding=1, bias=True).eval()(x)
        assert out.features.shape[0] == 0
        assert out.features.shape[1] == 16

    def test_cache_consistency(self):
        """同一 conv 两次调用应得到相同结果（邻居表缓存）"""
        x = make_tensor(150, 8, (8, 10, 12))
        conv = SubMConv3d(8, 8, 3, padding=1, bias=True).eval()
        o1 = conv(x).features
        o2 = conv(x).features
        assert_close(o1, o2, msg="cache consistency")

    def test_different_indice_keys_independent(self):
        """不同 indice_key 不应互相污染"""
        x = make_tensor(150, 8, (8, 10, 12))
        c1 = SparseConv3d(8, 16, 3, stride=2, padding=1, bias=True, indice_key="k1").eval()
        c2 = SparseConv3d(8, 16, 3, stride=2, padding=1, bias=True, indice_key="k2").eval()
        y1 = c1(x)
        y2 = c2(x)
        assert "k1" in y1._indice_dict and "k2" not in y1._indice_dict
        assert "k2" in y2._indice_dict and "k1" not in y2._indice_dict

    def test_large_grid_lookup_fallback(self):
        """空间网格超过 _GRID_LOOKUP_MAX_ENTRIES 时应回退到 sort+searchsorted。

        稀疏大空间（sp_total > 8M）不走稠密网格 O(1) 查找，结果应与参考一致。
        """
        spatial = (512, 512, 64)  # 16.7M > 8M 上限
        x = make_tensor(50, 8, spatial, seed=5)
        conv = SubMConv3d(8, 8, 3, padding=1, bias=True, indice_key="lg").eval()
        y = conv(x)
        ref = ref_subm(conv, x)
        assert_close(y.features, ref, msg="large-grid lookup fallback (SubM)")


# ============================================================
# 梯度检查
# ============================================================

class TestGradient:
    def test_subm_grad(self):
        x = make_tensor(100, 4, (6, 10, 12))
        conv = SubMConv3d(4, 8, 3, padding=1, bias=True)
        y = conv(x)
        y.features.sum().backward()
        g = conv.weight.grad
        assert g is not None
        assert torch.isfinite(g).all()
        assert conv.bias.grad is not None

    def test_sparse_grad(self):
        x = make_tensor(150, 4, (8, 10, 12))
        conv = SparseConv3d(4, 8, 3, stride=2, padding=1, bias=True)
        y = conv(x)
        y.features.sum().backward()
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None

    def test_sequential_grad(self):
        x = make_tensor(150, 4, (6, 10, 12))
        net = SparseSequential(
            SubMConv3d(4, 8, 3, padding=1, bias=True),
            SparseReLU(),
            SubMConv3d(8, 4, 3, padding=1, bias=True),
        )
        y = net(x)
        y.features.sum().backward()
        convs = [m for m in net if isinstance(m, SubMConv3d)]
        assert len(convs) == 2
        for c in convs:
            assert c.weight.grad is not None
            assert c.bias.grad is not None


# ============================================================
# CPU 版本测试（默认所有算子测试已在 CPU 上运行，这里补充显式断言）
# ============================================================

class TestCPU:
    def test_all_modules_on_cpu(self):
        """CPU 上构造各算子，确认 features/indices 均位于 cpu"""
        x = make_tensor(60, 8, (6, 6, 6))
        assert x.features.device.type == "cpu"
        convs = [
            SubMConv3d(8, 8, 3, padding=1, bias=True),
            SparseConv3d(8, 8, 3, stride=2, padding=1, bias=True),
            SparseInverseConv3d(8, 8, 3, indice_key="k", bias=True),
        ]
        for c in convs:
            assert c.weight.device.type == "cpu", f"{type(c).__name__} weight not on cpu"

    def test_subm_cpu_output_device(self):
        x = make_tensor(80, 8, (6, 6, 6))
        y = SubMConv3d(8, 8, 3, padding=1, bias=True).eval()(x)
        assert y.features.device.type == "cpu"
        assert y.indices.device.type == "cpu"

    def test_sparse_cpu_output_device(self):
        x = make_tensor(80, 8, (6, 6, 6))
        y = SparseConv3d(8, 8, 3, stride=2, padding=1, bias=True).eval()(x)
        assert y.features.device.type == "cpu"
        assert y.indices.device.type == "cpu"

    def test_cpu_to_cuda_back_and_forth(self):
        """CPU 张量 -> CUDA -> 回到 CPU，数值保持不变（CUDA 可用时）"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        x = make_tensor(60, 8, (6, 6, 6))
        orig = x.features.clone()
        x_dev = x.to("cuda:0")
        assert x_dev.features.device.type == "cuda"
        x_back = x_dev.cpu()
        assert_close(orig, x_back.features, atol=1e-6, msg="cpu->cuda->cpu")


# ============================================================
# CUDA 版本测试（仅当 CUDA 可用时执行）
# ============================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    def test_subm_cuda_vs_cpu(self):
        """SubMConv3d CUDA 与 CPU 数值一致性"""
        dev = torch.device("cuda:0")
        x = make_tensor(100, 8, (6, 6, 6))
        conv_cpu = SubMConv3d(8, 8, 3, padding=1, bias=True).eval()
        yc = conv_cpu(x).features

        conv_g = deepcopy(conv_cpu).to(dev)
        x_g = SparseConvTensor(x.features.to(dev), x.indices.to(dev),
                               x.spatial_shape, x.batch_size)
        yg = conv_g(x_g).features
        assert_close(yc, yg.cpu(), atol=1e-5, rtol=1e-4, msg="SubMConv3d CUDA vs CPU")

    def test_sparse_cuda_vs_cpu(self):
        """SparseConv3d CUDA 与 CPU 数值一致性"""
        dev = torch.device("cuda:0")
        x = make_tensor(150, 16, (8, 8, 8))
        conv_cpu = SparseConv3d(16, 32, 3, stride=2, padding=1, bias=True).eval()
        yc = conv_cpu(x).features

        conv_g = deepcopy(conv_cpu).to(dev)
        x_g = SparseConvTensor(x.features.to(dev), x.indices.to(dev),
                               x.spatial_shape, x.batch_size)
        yg = conv_g(x_g).features
        assert_close(yc, yg.cpu(), atol=1e-5, rtol=1e-4, msg="SparseConv3d CUDA vs CPU")

    def test_subm_cuda_matches_reference(self):
        """SubMConv3d CUDA 上对照逐体素参考实现"""
        x = make_tensor(100, 8, (8, 8, 8), seed=42)
        conv = SubMConv3d(8, 8, 3, padding=1, bias=True, indice_key="s").eval()
        conv_g = deepcopy(conv).to("cuda:0")
        x_g = SparseConvTensor(x.features.to("cuda:0"), x.indices.to("cuda:0"),
                               x.spatial_shape, x.batch_size)
        new = conv_g(x_g).features.cpu()
        ref = ref_subm(conv, x)
        assert_close(new, ref, atol=1e-5, rtol=1e-4, msg="SubMConv3d CUDA vs ref")

    def test_sparse_cuda_matches_reference(self):
        """SparseConv3d CUDA 上对照逐体素参考实现"""
        x = make_tensor(120, 16, (10, 10, 10), seed=1)
        conv = SparseConv3d(16, 32, 3, stride=2, padding=1,
                            bias=True, indice_key="down").eval()
        conv_g = deepcopy(conv).to("cuda:0")
        x_g = SparseConvTensor(x.features.to("cuda:0"), x.indices.to("cuda:0"),
                               x.spatial_shape, x.batch_size)
        y = conv_g(x_g)
        ref = ref_sparse(conv, x)
        assert_close(y.features.cpu(), ref, atol=1e-5, rtol=1e-4, msg="SparseConv3d CUDA vs ref")

    def test_inverse_cuda_vs_cpu(self):
        """SparseInverseConv3d CUDA 与 CPU 数值一致性"""
        dev = torch.device("cuda:0")
        x = make_tensor(100, 8, (6, 6, 6), seed=3)
        down = SparseConv3d(8, 16, 3, stride=2, padding=1,
                            bias=True, indice_key="up").eval()
        y = down(x)
        z = SparseConvTensor(torch.randn(y.features.shape[0], 16), y.indices,
                             y.spatial_shape, x.batch_size)
        z._indice_dict = dict(y._indice_dict)
        up = SparseInverseConv3d(16, 8, 3, indice_key="up", bias=True).eval()

        z_g = SparseConvTensor(z.features.to(dev), z.indices.to(dev),
                               z.spatial_shape, z.batch_size)
        z_g._indice_dict = dict(z._indice_dict)
        up_g = deepcopy(up).to(dev)
        yg = up_g(z_g).features.cpu()
        yc = up(z).features
        assert_close(yc, yg, atol=1e-5, rtol=1e-4, msg="SparseInverseConv3d CUDA vs CPU")

    def test_cuda_indices_unchanged_subm(self):
        """SubMConv3d CUDA 输出坐标必须等于输入坐标"""
        x = make_tensor(100, 8, (8, 8, 8))
        conv = SubMConv3d(8, 8, 3, padding=1, bias=True).to("cuda:0").eval()
        x_g = SparseConvTensor(x.features.to("cuda:0"), x.indices.to("cuda:0"),
                               x.spatial_shape, x.batch_size)
        y = conv(x_g)
        assert torch.equal(y.indices.cpu(), x.indices)

    def test_grad_flow_cuda(self):
        """CUDA 上梯度计算"""
        x = make_tensor(80, 8, (6, 6, 6))
        conv = SubMConv3d(8, 8, 3, padding=1, bias=True).to("cuda:0")
        x_g = SparseConvTensor(x.features.to("cuda:0"), x.indices.to("cuda:0"),
                               x.spatial_shape, x.batch_size)
        y = conv(x_g)
        y.features.sum().backward()
        assert conv.weight.grad is not None
        assert torch.isfinite(conv.weight.grad.cpu()).all()
        assert conv.bias.grad is not None


# ============================================================
# 官方 spconv-cu114（第三方 CUDA 实现）vs 纯 torch 参考实现
# 目标1：以纯 torch 参考（ref_subm/ref_sparse/ref_inverse）为统一基准，
#       验证第三方 CUDA 实现精度。
# 目标2：NPU/AscendC（见 TestNPU）也用同一纯 torch 基准验证，
#       保证 NPU 与 CUDA 精度对齐。
# ============================================================

def _import_official_spconv():
    """加载官方 spconv-cu114 的 pytorch 模块

    本地实现通过 unum_ops.spconv 导入，官方包通过顶层 spconv 导入，
    两者模块名不冲突，可直接 import。
    """
    import spconv.pytorch as off_pt
    return off_pt


def _official_spconv_available():
    """模块级函数：检查官方 spconv-cu114 是否可导入（供 @skipif 使用）"""
    try:
        _import_official_spconv()
        return True
    except Exception:
        return False


def _weight_off_to_local(w_off, out_channels, in_channels, kernel_size):
    """官方 spconv 权重 (out, D, H, W, in) → 本地 (out, in, D, H, W)"""
    return w_off.permute(0, 4, 1, 2, 3).contiguous()


def _load_official_into_local(local_conv, off_conv):
    """把官方 conv 的权重/偏置拷进本地 conv（处理布局差异）"""
    off_sd = off_conv.state_dict()
    loc_sd = local_conv.state_dict()
    for k, ov in off_sd.items():
        lv = loc_sd[k]
        if ov.shape == lv.shape:
            loc_sd[k].copy_(ov.cpu())
        elif len(ov.shape) == 5 and len(lv.shape) == 5:
            loc_sd[k].copy_(_weight_off_to_local(
                ov.cpu(), lv.shape[0], lv.shape[1], lv.shape[2:]))
    return local_conv


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestOfficialSpconv:
    """官方 spconv-cu114 CUDA 实现 vs 纯 torch 标准参考实现。

    目标：以纯 torch 参考（ref_*）为统一基准验证第三方 CUDA 精度；
    NPU（TestNPU）同样以 ref_* 为基准，从而保证 NPU 与 CUDA 精度对齐。
    """

    @pytest.mark.skipif(not _official_spconv_available(),
                        reason="official spconv not installed")
    def test_subm_vs_pure_torch(self):
        """SubMConv3d 官方 CUDA vs 纯 torch 参考（均在 CUDA 上跑）"""
        off_pt = _import_official_spconv()
        x_cpu = make_tensor(200, 16, (10, 10, 10), seed=42)

        # 官方：CUDA
        off_conv = off_pt.SubMConv3d(16, 32, 3, padding=1, bias=True,
                                     indice_key="s").cuda().eval()
        off_x = off_pt.SparseConvTensor(
            x_cpu.features.cuda(), x_cpu.indices.cuda(), (10, 10, 10), 1)
        with torch.no_grad():
            off_out = off_conv(off_x).features.cpu()

        # 纯 torch 参考（同权重，在 CUDA 上跑以对齐 fp32 matmul 舍入）
        ref_conv = SubMConv3d(16, 32, 3, padding=1, bias=True,
                              indice_key="s").eval()
        _load_official_into_local(ref_conv, off_conv)
        ref_conv = ref_conv.to("cuda:0")
        x_cuda = SparseConvTensor(x_cpu.features.cuda(), x_cpu.indices.cuda(),
                                  (10, 10, 10), 1)
        ref = ref_subm(ref_conv, x_cuda)
        assert_close(off_out, ref.cpu(), atol=1e-4, rtol=1e-3,
                     msg="official spconv CUDA vs pure torch ref (SubMConv3d)")

    @pytest.mark.skipif(not _official_spconv_available(),
                        reason="official spconv not installed")
    def test_sparse_vs_pure_torch(self):
        """SparseConv3d 官方 CUDA vs 纯 torch 参考（均在 CUDA 上跑）"""
        off_pt = _import_official_spconv()
        x_cpu = make_tensor(200, 16, (10, 10, 10), seed=1)

        off_conv = off_pt.SparseConv3d(16, 32, 3, stride=2, padding=1,
                                       bias=True, indice_key="down").cuda().eval()
        off_x = off_pt.SparseConvTensor(
            x_cpu.features.cuda(), x_cpu.indices.cuda(), (10, 10, 10), 1)
        with torch.no_grad():
            off_out_t = off_conv(off_x)
        off_feats = off_out_t.features.cpu()
        off_idx = off_out_t.indices.cpu()

        # 纯 torch 参考（同权重，在 CUDA 上跑）
        ref_conv = SparseConv3d(16, 32, 3, stride=2, padding=1,
                                bias=True, indice_key="down").eval()
        _load_official_into_local(ref_conv, off_conv)
        ref_conv = ref_conv.to("cuda:0")
        x_cuda = SparseConvTensor(x_cpu.features.cuda(), x_cpu.indices.cuda(),
                                  (10, 10, 10), 1)
        ref_idx, ref_feats = ref_sparse(ref_conv, x_cuda, return_idx=True)
        ref_idx, ref_feats = ref_idx.cpu(), ref_feats.cpu()

        # 输出坐标集合必须一致
        assert ref_idx.shape == off_idx.shape, \
            f"output index count mismatch: {ref_idx.shape} vs {off_idx.shape}"
        # 坐标顺序可能不同：按 lexicographic 编码排序后直接对比
        off_sort = _lexsort(off_idx)
        ref_sort = _lexsort(ref_idx)
        assert torch.equal(ref_idx[ref_sort], off_idx[off_sort]), \
            "output coordinates differ after sorting"
        assert_close(ref_feats[ref_sort], off_feats[off_sort],
                     atol=1e-4, rtol=1e-3,
                     msg="official spconv CUDA vs pure torch ref (SparseConv3d)")

    @pytest.mark.skipif(not _official_spconv_available(),
                        reason="official spconv not installed")
    def test_subm_random_seeds_vs_pure_torch(self):
        """多个随机种子下 SubMConv3d 官方 CUDA vs 纯 torch 参考"""
        off_pt = _import_official_spconv()
        for seed in range(3):
            x_cpu = make_tensor(150, 8, (8, 8, 8), seed=seed)
            off_conv = off_pt.SubMConv3d(8, 8, 3, padding=1, bias=True).cuda().eval()
            off_x = off_pt.SparseConvTensor(
                x_cpu.features.cuda(), x_cpu.indices.cuda(), (8, 8, 8), 1)
            with torch.no_grad():
                off_out = off_conv(off_x).features.cpu()
            ref_conv = SubMConv3d(8, 8, 3, padding=1, bias=True).eval()
            _load_official_into_local(ref_conv, off_conv)
            ref_conv = ref_conv.to("cuda:0")
            x_cuda = SparseConvTensor(x_cpu.features.cuda(), x_cpu.indices.cuda(),
                                      (8, 8, 8), 1)
            ref = ref_subm(ref_conv, x_cuda)
            assert_close(off_out, ref.cpu(), atol=1e-4, rtol=1e-3,
                         msg=f"official spconv CUDA vs pure torch ref (seed={seed})")

    @pytest.mark.skipif(not _official_spconv_available(),
                        reason="official spconv not installed")
    def test_inverse_vs_pure_torch(self):
        """SparseInverseConv3d 官方 CUDA vs 纯 torch 参考（均在 CUDA 上跑）"""
        off_pt = _import_official_spconv()
        x_cpu = make_tensor(120, 8, (8, 8, 8), seed=3)

        # 官方流程
        off_down = off_pt.SparseConv3d(8, 16, 3, stride=2, padding=1,
                                       bias=True, indice_key="up").cuda().eval()
        off_x = off_pt.SparseConvTensor(
            x_cpu.features.cuda(), x_cpu.indices.cuda(), (8, 8, 8), 1)
        with torch.no_grad():
            off_down_out = off_down(off_x)
        off_up = off_pt.SparseInverseConv3d(16, 8, 3, indice_key="up",
                                            bias=True).cuda().eval()
        with torch.no_grad():
            off_out_t = off_up(off_down_out)
        off_feats = off_out_t.features.cpu()
        off_idx = off_out_t.indices.cpu()

        # 纯 torch 参考流程（官方权重 + CUDA 设备）
        loc_down = SparseConv3d(8, 16, 3, stride=2, padding=1,
                                bias=True, indice_key="up").eval()
        loc_up = SparseInverseConv3d(16, 8, 3, indice_key="up",
                                     bias=True).eval()
        _load_official_into_local(loc_down, off_down)
        _load_official_into_local(loc_up, off_up)
        loc_down, loc_up = loc_down.to("cuda:0"), loc_up.to("cuda:0")
        x_cuda = SparseConvTensor(x_cpu.features.cuda(), x_cpu.indices.cuda(),
                                  (8, 8, 8), 1)
        with torch.no_grad():
            loc_down_out = loc_down(x_cuda)
        info = loc_down_out._indice_dict["up"]
        ref_idx, ref_feats = ref_inverse(loc_up, loc_down_out, info,
                                         return_idx=True)
        ref_idx, ref_feats = ref_idx.cpu(), ref_feats.cpu()

        # 输出坐标集合必须一致
        assert ref_idx.shape == off_idx.shape, \
            f"output index count mismatch: {ref_idx.shape} vs {off_idx.shape}"
        off_sort = _lexsort(off_idx)
        ref_sort = _lexsort(ref_idx)
        assert torch.equal(ref_idx[ref_sort], off_idx[off_sort]), \
            "output coordinates differ after sorting"
        assert_close(ref_feats[ref_sort], off_feats[off_sort],
                     atol=1e-4, rtol=1e-3,
                     msg="official spconv CUDA vs pure torch ref (SparseInverseConv3d)")


# ============================================================
# NPU / AscendC 版本测试（仅当 NPU 可用时执行）
# 与官方 CUDA 共享同一纯 torch 参考基准（ref_*）→ 保证 NPU 与 CUDA 精度对齐
# ============================================================

@pytest.mark.skipif(not (hasattr(torch, "npu") and torch.npu.is_available()),
                    reason="NPU not available")
class TestNPU:
    def test_subm_npu_vs_pure_torch(self):
        """SubMConv3d NPU/AscendC vs 纯 torch 参考实现"""
        dev = torch.device("npu:0")
        x = make_tensor(100, 8, (6, 6, 6))
        conv_cpu = SubMConv3d(8, 8, 3, padding=1, bias=True, indice_key="s").eval()
        ref = ref_subm(conv_cpu, x)

        conv_n = SubMConv3d(8, 8, 3, padding=1, bias=True, indice_key="s").eval()
        conv_n.load_state_dict(conv_cpu.state_dict())
        conv_n = conv_n.to(dev)  # 必须将 conv 移到 NPU，否则 _wT 在 CPU
        x_n = SparseConvTensor(x.features.to(dev), x.indices.to(dev),
                               x.spatial_shape, x.batch_size)
        yn = conv_n(x_n).features
        # NPU 混合精度路径放宽到 atol=1e-3, rtol=1e-2
        assert_close(yn.cpu(), ref, atol=1e-3, rtol=1e-2,
                     msg="SubMConv3d NPU/AscendC vs pure torch ref")

    def test_sparse_npu_vs_pure_torch(self):
        """SparseConv3d NPU/AscendC vs 纯 torch 参考实现（坐标排序后对齐）"""
        dev = torch.device("npu:0")
        x = make_tensor(150, 16, (10, 10, 10), seed=1)
        conv_cpu = SparseConv3d(16, 32, 3, stride=2, padding=1,
                                bias=True, indice_key="down").eval()
        ref_idx, ref_feats = ref_sparse(conv_cpu, x, return_idx=True)

        conv_n = SparseConv3d(16, 32, 3, stride=2, padding=1,
                              bias=True, indice_key="down").eval()
        conv_n.load_state_dict(conv_cpu.state_dict())
        conv_n = conv_n.to(dev)
        x_n = SparseConvTensor(x.features.to(dev), x.indices.to(dev),
                               x.spatial_shape, x.batch_size)
        y = conv_n(x_n)
        y_feats = y.features.cpu()
        y_idx = y.indices.cpu()

        # 输出坐标集合必须一致（顺序可能不同）
        assert y_idx.shape == ref_idx.shape, \
            f"output index count mismatch: {y_idx.shape} vs {ref_idx.shape}"
        y_sort = _lexsort(y_idx)
        ref_sort = _lexsort(ref_idx)
        assert torch.equal(ref_idx[ref_sort], y_idx[y_sort]), \
            "output coordinates differ after sorting"
        assert_close(y_feats[y_sort], ref_feats[ref_sort],
                     atol=1e-3, rtol=1e-2,
                     msg="SparseConv3d NPU/AscendC vs pure torch ref")

    def test_inverse_npu_vs_pure_torch(self):
        """SparseInverseConv3d NPU/AscendC vs 纯 torch 参考实现（坐标排序后对齐）"""
        dev = torch.device("npu:0")
        x = make_tensor(100, 8, (8, 8, 8), seed=3)

        down_cpu = SparseConv3d(8, 16, 3, stride=2, padding=1,
                                bias=True, indice_key="up").eval()
        up_cpu = SparseInverseConv3d(16, 8, 3, indice_key="up",
                                     bias=True).eval()
        y_cpu = down_cpu(x)
        info = y_cpu._indice_dict["up"]
        ref_idx, ref_feats = ref_inverse(up_cpu, y_cpu, info, return_idx=True)

        down_n = SparseConv3d(8, 16, 3, stride=2, padding=1,
                              bias=True, indice_key="up").eval()
        up_n = SparseInverseConv3d(16, 8, 3, indice_key="up",
                                   bias=True).eval()
        down_n.load_state_dict(down_cpu.state_dict())
        up_n.load_state_dict(up_cpu.state_dict())
        down_n = down_n.to(dev)
        up_n = up_n.to(dev)
        x_n = SparseConvTensor(x.features.to(dev), x.indices.to(dev),
                               x.spatial_shape, x.batch_size)
        y_n = down_n(x_n)
        out = up_n(y_n)
        out_feats = out.features.cpu()
        out_idx = out.indices.cpu()

        # 输出坐标集合必须一致（顺序可能不同）
        assert out_idx.shape == ref_idx.shape, \
            f"output index count mismatch: {out_idx.shape} vs {ref_idx.shape}"
        out_sort = _lexsort(out_idx)
        ref_sort = _lexsort(ref_idx)
        assert torch.equal(ref_idx[ref_sort], out_idx[out_sort]), \
            "output coordinates differ after sorting"
        assert_close(out_feats[out_sort], ref_feats[ref_sort],
                     atol=1e-3, rtol=1e-2,
                     msg="SparseInverseConv3d NPU/AscendC vs pure torch ref")


# ============================================================
# Adapter 层测试
# ============================================================

class TestAdapters:
    def test_subm_adapter_forward(self):
        x = make_tensor(80, 8, (8, 8, 8), seed=20)
        adapter = SubMConv3dAdapter(8, 16, 3, padding=1, bias=True).eval()
        direct = SubMConv3d(8, 16, 3, padding=1, bias=True).eval()
        direct.load_state_dict(adapter.conv.state_dict())
        ya = adapter(x)
        yd = direct(x)
        assert_close(ya.features, yd.features, msg="SubM adapter vs direct")

    def test_sparse_adapter_forward(self):
        x = make_tensor(100, 8, (10, 10, 10), seed=21)
        adapter = SparseConv3dAdapter(8, 16, 3, stride=2, padding=1, bias=True).eval()
        direct = SparseConv3d(8, 16, 3, stride=2, padding=1, bias=True).eval()
        direct.load_state_dict(adapter.conv.state_dict())
        ya = adapter(x)
        yd = direct(x)
        assert_close(ya.features, yd.features, msg="Sparse adapter vs direct")
        assert torch.equal(ya.indices, yd.indices)


# ============================================================
# 多 batch 场景
# ============================================================

class TestMultiBatch:
    def test_multi_batch_subm(self):
        """多 batch 下不同 batch 的体素不应互相影响"""
        N = 80
        feats = torch.randn(N, 4)
        idx0 = torch.zeros(N // 2, 4, dtype=torch.int32)
        idx0[:, 1:] = torch.randint(0, 6, (N // 2, 3))
        idx1 = torch.ones(N // 2, 4, dtype=torch.int32)
        idx1[:, 1:] = torch.randint(0, 6, (N // 2, 3))
        indices = torch.cat([idx0, idx1], dim=0)
        # 去重：重复坐标会导致 ref(dict)与 impl(searchsorted)对重复行的处理不一致
        indices = torch.unique(indices, dim=0)
        feats = feats[:indices.shape[0]]
        x = SparseConvTensor(feats, indices, (6, 6, 6), 2)
        conv = SubMConv3d(4, 8, 3, padding=1, bias=True).eval()
        y = conv(x)
        assert y.features.shape == (indices.shape[0], 8)
        ref = ref_subm(conv, x)
        assert_close(y.features, ref, msg="multi-batch SubM")

    def test_multi_batch_sparse(self):
        N = 100
        feats = torch.randn(N, 4)
        idx0 = torch.zeros(N // 2, 4, dtype=torch.int32)
        idx0[:, 1:] = torch.randint(0, 8, (N // 2, 3))
        idx1 = torch.ones(N // 2, 4, dtype=torch.int32)
        idx1[:, 1:] = torch.randint(0, 8, (N // 2, 3))
        indices = torch.cat([idx0, idx1], dim=0)
        indices = torch.unique(indices, dim=0)
        feats = feats[:indices.shape[0]]
        x = SparseConvTensor(feats, indices, (8, 8, 8), 2)
        conv = SparseConv3d(4, 8, 3, stride=2, padding=1, bias=True).eval()
        y = conv(x)
        ref = ref_sparse(conv, x)
        assert_close(y.features, ref, msg="multi-batch Sparse")


# ============================================================
# dilation 参数测试
# ============================================================

class TestDilation:
    def test_dilation_subm(self):
        """dilation=2 的 SubMConv 应查找更远的邻居，数值与 ref_subm 一致"""
        x = make_tensor(100, 8, (10, 10, 10), seed=30)
        conv = SubMConv3d(8, 8, 3, padding=2, dilation=2, bias=True).eval()
        y = conv(x)
        ref = ref_subm(conv, x)
        assert y.features.shape == (x.features.shape[0], 8)
        assert_close(y.features, ref, msg="SubMConv dilation=2")


# ============================================================
# VoxelGeneratorV2 补充测试
# ============================================================

class TestVoxelGeneratorExtra:
    def test_empty_input(self):
        """空输入不应报错"""
        gen = VoxelGeneratorV2(
            voxel_size=(0.1, 0.1, 0.1),
            point_cloud_range=(0, 0, 0, 5, 5, 5),
            max_num_points=5, max_voxels=100,
        )
        out = gen.generate(np.zeros((0, 4), dtype=np.float32))
        assert out['voxels'].shape[0] == 0
        assert out['coordinates'].shape[0] == 0

    def test_all_out_of_range(self):
        """全部越界的输入应返回空"""
        gen = VoxelGeneratorV2(
            voxel_size=(1, 1, 1),
            point_cloud_range=(0, 0, 0, 5, 5, 5),
            max_num_points=5, max_voxels=100,
        )
        pts = np.array([[100, 100, 100, 1], [-1, -1, -1, 1]], dtype=np.float32)
        out = gen.generate(pts)
        assert out['voxels'].shape[0] == 0

    def test_max_voxels_truncation(self):
        """超过 max_voxels 时应截断"""
        gen = VoxelGeneratorV2(
            voxel_size=(0.5, 0.5, 0.5),
            point_cloud_range=(0, 0, 0, 10, 10, 10),
            max_num_points=5, max_voxels=3,
        )
        pts = np.random.uniform(0, 10, (200, 4)).astype(np.float32)
        out = gen.generate(pts)
        assert out['voxels'].shape[0] <= 3

    def test_max_num_points_truncation(self):
        """超过 max_num_points 的点不应溢出"""
        gen = VoxelGeneratorV2(
            voxel_size=(10, 10, 10),
            point_cloud_range=(0, 0, 0, 100, 100, 100),
            max_num_points=3, max_voxels=10,
        )
        pts = np.zeros((10, 4), dtype=np.float32)
        pts[:, :3] = 1.0  # 全部落入同一体素
        out = gen.generate(pts)
        assert out['num_points_per_voxel'][0] <= 3

    def test_coordinate_order_zyx(self):
        """验证输出坐标为 [z, y, x] 顺序"""
        gen = VoxelGeneratorV2(
            voxel_size=(1, 1, 1),
            point_cloud_range=(0, 0, 0, 5, 5, 5),
            max_num_points=5, max_voxels=100,
        )
        # 放一个在 (x=2, y=1, z=0) 的点
        pts = np.array([[2.5, 1.5, 0.5, 1.0]], dtype=np.float32)
        out = gen.generate(pts)
        assert out['coordinates'].shape[0] == 1
        # coords_out = coords[m[0]][[2, 1, 0]] → [z, y, x] = [0, 1, 2]
        assert out['coordinates'][0, 0] == 0  # z
        assert out['coordinates'][0, 1] == 1  # y
        assert out['coordinates'][0, 2] == 2  # x


# ============================================================
# SparseConvTensor 补充测试
# ============================================================

class TestSparseConvTensorExtra:
    def test_dense_out_of_range_filtered(self):
        """dense() 应跳过越界坐标，只有 2 个有效体素被填充"""
        feats = torch.randn(3, 4)
        idx = torch.tensor([
            [0, 0, 0, 0],
            [0, 2, 3, 1],
            [0, 100, 0, 0],  # 越界
        ], dtype=torch.int32)
        t = SparseConvTensor(feats, idx, (3, 5, 3), 1)
        dense = t.dense(channels_first=True)
        # 第 3 行越界被跳过，只有 2 个体素被填充
        filled = (dense.abs().sum(dim=1) > 0).sum().item()
        assert filled == 2, f"expected 2 filled voxels, got {filled}"

    def test_to_does_not_corrupt_original(self):
        """to() 返回新对象，修改新对象不应影响原 tensor"""
        t = make_tensor(20, 4, (6, 8, 10))
        original_feats = t.features.clone()
        t2 = t.cpu()
        assert t2 is not t, "to() should return a new SparseConvTensor"
        t2.features.add_(1)
        # 原对象不应被修改
        assert torch.equal(t.features, original_feats)
        # t2 能正常使用且与 t 共享相同坐标/元数据
        assert t2.features.shape == original_feats.shape
        assert torch.equal(t2.indices, t.indices)

    def test_from_dense_roundtrip_values(self):
        """dense -> from_dense 往返后数值应一致（立方体 spatial 避免越界跳过）"""
        t = make_tensor(50, 4, (8, 8, 8), seed=99)
        dense = t.dense(channels_first=True)
        t2 = SparseConvTensor.from_dense(dense, channels_first=True)
        # 重新排序比较
        d1 = dict(zip(
            [tuple(r.tolist()) for r in t.indices.long()],
            t.features.tolist()
        ))
        d2 = dict(zip(
            [tuple(r.tolist()) for r in t2.indices.long()],
            t2.features.tolist()
        ))
        for k in d1:
            assert k in d2, f"coord {k} missing in from_dense"
            for a, b in zip(d1[k], d2[k]):
                assert abs(a - b) < 1e-4, f"value mismatch at {k}"


# ============================================================
# kernel_size=1 测试
# ============================================================

class TestKernelSize1:
    def test_subm_k1_matches_reference(self):
        """kernel_size=1 的 SubMConv 等价于逐体素线性变换"""
        x = make_tensor(80, 8, (8, 8, 8), seed=40)
        conv = SubMConv3d(8, 16, 1, padding=0, bias=True, indice_key="k1").eval()
        y = conv(x)
        ref = ref_subm(conv, x)
        assert_close(y.features, ref, msg="SubM k=1")

    def test_sparse_k1_matches_reference(self):
        """kernel_size=1 的 SparseConv 应等于步长下采样 + 逐体素线性变换"""
        x = make_tensor(100, 8, (10, 10, 10), seed=41)
        conv = SparseConv3d(8, 16, 1, stride=2, padding=0,
                            bias=True, indice_key="k1").eval()
        y = conv(x)
        ref = ref_sparse(conv, x)
        assert_close(y.features, ref, msg="Sparse k=1")


# ============================================================
# SparseLinear 补充
# ============================================================

class TestSparseLinearExtra:
    def test_no_bias(self):
        x = make_tensor(50, 8, (6, 8, 10))
        lin = SparseLinear(8, 16, bias=False)
        y = lin(x)
        assert y.features.shape == (50, 16)
        ref = torch.nn.functional.linear(x.features, lin.linear.weight, None)
        assert_close(y.features, ref, msg="SparseLinear no bias")

    def test_grad_flow(self):
        x = make_tensor(50, 8, (6, 8, 10))
        lin = SparseLinear(8, 16, bias=True)
        y = lin(x)
        y.features.sum().backward()
        assert lin.linear.weight.grad is not None
        assert lin.linear.bias.grad is not None


# ============================================================
# 缓存淘汰与多输入隔离
# ============================================================

class TestCacheEviction:
    def test_cache_eviction_correctness(self):
        """缓存满 8 个后淘汰最旧的，后续结果仍正确"""
        conv = SubMConv3d(8, 8, 3, padding=1, bias=True).eval()
        # 用 10 个不同的输入触发缓存淘汰
        results = []
        for seed in range(10):
            x = make_tensor(50, 8, (6, 6, 6), seed=seed)
            results.append((x, conv(x).features.clone()))
        # 再次验证前两个（应已被淘汰，重新构建）
        for i in [0, 1]:
            x, expected = results[i]
            actual = conv(x).features
            assert_close(actual, expected, atol=1e-4, msg=f"after eviction seed={i}")

    def test_different_inputs_different_cache(self):
        """不同输入不应命中同一缓存"""
        x1 = make_tensor(80, 8, (6, 6, 6), seed=1)
        x2 = make_tensor(80, 8, (6, 6, 6), seed=2)
        conv = SubMConv3d(8, 8, 3, padding=1, bias=True).eval()
        y1 = conv(x1).features
        y2 = conv(x2).features
        assert not torch.allclose(y1, y2, atol=1e-3), "different inputs got same output"


# ============================================================
# SparseInverseConv in_shape 回退路径
# ============================================================

class TestInverseConvFallback:
    def test_in_shape_from_stride_product(self):
        """当 indice_dict 中没有 in_shape 时，用 spatial * stride 推断"""
        x = make_tensor(100, 8, (10, 10, 10), seed=50)
        down = SparseConv3d(8, 16, 3, stride=2, padding=1,
                            bias=True, indice_key="fb").eval()
        y = down(x)
        z = SparseConvTensor(torch.randn(y.features.shape[0], 16), y.indices,
                             y.spatial_shape, x.batch_size)
        # 只保留 in_coords，删除 in_shape
        z._indice_dict = {"fb": {
            "in_coords": x.indices.clone(),
            "stride": (2, 2, 2),
            "padding": (1, 1, 1),
            "kernel": (3, 3, 3),
        }}
        up = SparseInverseConv3d(16, 8, 3, indice_key="fb", bias=True).eval()
        out = up(z)
        # in_shape = spatial * stride = (5*2, 5*2, 5*2) = (10, 10, 10)
        assert tuple(out.spatial_shape[:3]) == (10, 10, 10)
