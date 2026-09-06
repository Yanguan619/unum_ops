"""Ascend 310P BevPool 算子正确性测试。

覆盖 unum_ops.bev_pool 接口，与 torch-native index_add_ 参考实现对比。

运行方式:
    cd /data/workspace/unum_ops
    PYTHONPATH=".venv/lib/python3.11/site-packages:$PYTHONPATH" \
        pytest test/test_bev_pool.py -v
"""
import os
import sys

import pytest
import torch
import torch_npu

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(0)

from unum_ops.bev_pool import bev_pool, bev_pool_torch


# ── Reference: bev_pool_torch (pure-PyTorch scatter_add, CPU) ──────────────

def bev_pool_ref(feats, coords, B, D, H, W):
    return bev_pool_torch(feats, coords, B, D, H, W).out


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_random_points(N, B, D, H, W, C, seed=42):
    rng = torch.Generator()
    rng.manual_seed(seed)
    feats = torch.randn(N, C, generator=rng, dtype=torch.float32)
    coords = torch.zeros(N, 4, dtype=torch.int64)
    coords[:, 0] = torch.randint(0, W, (N,), generator=rng, dtype=torch.int64)
    coords[:, 1] = torch.randint(0, H, (N,), generator=rng, dtype=torch.int64)
    coords[:, 2] = torch.randint(0, D, (N,), generator=rng, dtype=torch.int64)
    coords[:, 3] = torch.randint(0, B, (N,), generator=rng, dtype=torch.int64)
    return feats, coords


def _run(feats, coords, B, D, H, W):
    pts = feats.npu().contiguous()
    cs = coords.npu().contiguous()
    torch.npu.synchronize()
    out = bev_pool(pts, cs, B, D, H, W)
    torch.npu.synchronize()
    return out.out


# ── Warm-up ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _warm_up_npu():
    feats, coords = _make_random_points(100, 1, 2, 4, 4, 8, seed=1)
    bev_pool(feats.npu().contiguous(), coords.npu().contiguous(), 1, 2, 4, 4)
    torch.npu.synchronize()


# ── Tests ──────────────────────────────────────────────────────────────────

def test_small_random():
    """小规模随机数据，与 CPU index_add_ 参考对比。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 50
    feats, coords = _make_random_points(N, B, D, H, W, C)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5, rtol=1e-5), \
        f"small random mismatch: max diff={torch.abs(out.cpu() - ref).max().item()}"


def test_medium_random():
    """中等规模随机数据。"""
    B, D, H, W, C = 2, 4, 8, 8, 16
    N = 500
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=123)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5, rtol=1e-5), \
        f"medium random mismatch"


def test_multibatch():
    """多 batch 测试。"""
    B, D, H, W, C = 2, 3, 5, 5, 8
    N = 200
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=456)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5, rtol=1e-5)


def test_single_point():
    """单点输入。"""
    B, D, H, W, C = 1, 1, 1, 1, 4
    feats = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
    out = _run(feats, coords, B, D, H, W)
    expected = feats.reshape(B, C, D, H, W)
    assert torch.allclose(out.cpu(), expected, atol=1e-5)

    # 解码验证：out[0, :, 0, 0, 0] == feats[0]
    assert torch.allclose(out[0, :, 0, 0, 0].cpu(), feats[0], atol=1e-5)


def test_multiple_points_same_voxel():
    """多点落入同一 voxel，应累加。"""
    B, D, H, W, C = 1, 1, 1, 1, 4
    feats = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                          [2.0, 2.0, 2.0, 2.0],
                          [3.0, 3.0, 3.0, 3.0]], dtype=torch.float32)
    coords = torch.tensor([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=torch.int64)
    out = _run(feats, coords, B, D, H, W)
    expected = feats.sum(dim=0).reshape(B, C, D, H, W)
    assert torch.allclose(out.cpu(), expected, atol=1e-5)


def test_multiple_voxels():
    """多点分布到不同 voxel。"""
    B, D, H, W, C = 1, 1, 2, 2, 2
    feats = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                         dtype=torch.float32)
    coords = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0],
                           [0, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.int64)
    out = _run(feats, coords, B, D, H, W)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5)


def test_empty_points():
    """空点云（N=0）。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    feats = torch.empty(0, C, dtype=torch.float32)
    coords = torch.empty(0, 4, dtype=torch.int64)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5)


def test_large_channels():
    """较多通道数（C=80，BEVFusion 典型值）。"""
    B, D, H, W, C = 1, 2, 4, 4, 80
    N = 100
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=789)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-4, rtol=1e-4), \
        f"large channels mismatch"


# ── Output invariants ──────────────────────────────────────────────────────

def test_output_shape():
    """验证输出 shape 和 dtype。"""
    B, D, H, W, C = 2, 3, 5, 5, 16
    N = 200
    feats = torch.randn(N, C, dtype=torch.float32)
    coords = torch.zeros(N, 4, dtype=torch.int64)
    coords[:, 0] = torch.randint(0, W, (N,), dtype=torch.int64)
    coords[:, 1] = torch.randint(0, H, (N,), dtype=torch.int64)
    coords[:, 2] = torch.randint(0, D, (N,), dtype=torch.int64)
    coords[:, 3] = torch.randint(0, B, (N,), dtype=torch.int64)
    out = _run(feats, coords, B, D, H, W)
    assert out.shape == (B, C, D, H, W)
    assert out.dtype == torch.float32


def test_output_device():
    """验证输出在 NPU 上。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 50
    feats = torch.randn(N, C, dtype=torch.float32)
    coords = torch.zeros(N, 4, dtype=torch.int64)
    coords[:, 0] = torch.randint(0, W, (N,), dtype=torch.int64)
    coords[:, 1] = torch.randint(0, H, (N,), dtype=torch.int64)
    coords[:, 2] = torch.randint(0, D, (N,), dtype=torch.int64)
    coords[:, 3] = torch.randint(0, B, (N,), dtype=torch.int64)
    out = _run(feats, coords, B, D, H, W)
    assert out.device.type == "npu"


# ── Edge cases ──────────────────────────────────────────────────────────────

def _make_points(N, B, D, H, W, C, seed=42):
    """生成随机点，与 _make_random_points 一致。"""
    return _make_random_points(N, B, D, H, W, C, seed=seed)


def test_all_oob_points():
    """所有点都越界 → 输出全零。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 20
    feats, coords = _make_points(N, B, D, H, W, C, seed=6)
    coords_oob = coords.clone()
    coords_oob[:, 0] = -2  # 全部 x 越界
    out = _run(feats, coords_oob, B, D, H, W)
    assert torch.all(out.cpu() == 0.0), "all OOB points should produce zero output"


def test_all_oob_points():
    """所有点都越界 → 输出全零。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 20
    feats, coords = _make_points(N, B, D, H, W, C, seed=6)
    coords_oob = coords.clone()
    coords_oob[:, 0] = -2  # 全部 x 越界
    out = _run(feats, coords_oob, B, D, H, W)
    assert torch.all(out.cpu() == 0.0)


def test_rank_int64_no_float_collision():
    """rank 用 int64 排序，大网格下不因 float32 精度碰撞而错。"""
    # 构造 W < H 且 rank 值很大的情况（旧公式 float 会碰撞）
    B, D, H, W, C = 2, 3, 10, 4, 8
    N = 500
    feats, coords = _make_points(N, B, D, H, W, C, seed=11)
    # 随机打乱输入顺序（Python 层负责排序）
    perm = torch.randperm(N)
    feats_shuf, coords_shuf = feats[perm], coords[perm]
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats_shuf, coords_shuf, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5), \
        f"int64 rank mismatch: max diff={torch.abs(out.cpu() - ref).max().item()}"


def test_non_contiguous_input():
    """非连续输入应被拒绝（binding TORCH_CHECK 抛错）。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 50
    feats, coords = _make_points(N, B, D, H, W, C, seed=7)
    pts = feats.npu().contiguous()
    cs = coords.npu().contiguous()
    # 构造 rank 排序所需的输入（与 wrapper 一致），再制造非连续 feats
    ranks = (cs[:, 0] + cs[:, 1] * W + cs[:, 2] * (W * H) + cs[:, 3] * (W * H * D))
    indices = ranks.argsort()
    feats_sorted = pts[indices].contiguous()
    coords_sorted = cs[indices].int().contiguous()
    ranks_sorted = ranks[indices]
    kept = torch.ones(N, device=feats_sorted.device, dtype=torch.bool)
    kept[1:] = ranks_sorted[1:] != ranks_sorted[:-1]
    interval_starts = torch.where(kept)[0].int().contiguous()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = N - interval_starts[-1]

    # 制造非连续 feats：cat 后取偏移切片（stride 非 1）
    feats_big = torch.cat([feats_sorted, feats_sorted], dim=1)  # (N, 2C)
    feats_nc = feats_big[:, 1:C + 1]                            # (N, C) 非连续
    assert not feats_nc.is_contiguous()
    with pytest.raises(RuntimeError):
        torch.ops.unum.bev_pool(feats_nc, coords_sorted, interval_starts,
                                interval_lengths, int(B), int(D), int(H), int(W))


def test_wrong_dtype_input():
    """错误 dtype 输入应被拒绝（直接调底层算子）。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 50
    feats, coords = _make_points(N, B, D, H, W, C, seed=8)
    pts = feats.npu().contiguous()
    cs = coords.npu().contiguous()
    ranks = (cs[:, 0] + cs[:, 1] * W + cs[:, 2] * (W * H) + cs[:, 3] * (W * H * D))
    indices = ranks.argsort()
    feats_sorted = pts[indices].contiguous()
    coords_sorted = cs[indices].int().contiguous()
    ranks_sorted = ranks[indices]
    kept = torch.ones(N, device=feats_sorted.device, dtype=torch.bool)
    kept[1:] = ranks_sorted[1:] != ranks_sorted[:-1]
    interval_starts = torch.where(kept)[0].int().contiguous()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = N - interval_starts[-1]

    # feats 用 float16 → 拒绝（NPU 上 double 会被替换为 float，故用 half）
    with pytest.raises(RuntimeError):
        torch.ops.unum.bev_pool(feats_sorted.half(), coords_sorted,
                                interval_starts, interval_lengths,
                                int(B), int(D), int(H), int(W))
    # coords 用 float32 → 拒绝
    with pytest.raises(RuntimeError):
        torch.ops.unum.bev_pool(feats_sorted, coords_sorted.float(),
                                interval_starts, interval_lengths,
                                int(B), int(D), int(H), int(W))
    # interval_starts 用 float32 → 拒绝
    with pytest.raises(RuntimeError):
        torch.ops.unum.bev_pool(feats_sorted, coords_sorted,
                                interval_starts.float(), interval_lengths,
                                int(B), int(D), int(H), int(W))


def test_invalid_grid_dims():
    """非正网格维度应被拒绝。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 50
    feats, coords = _make_points(N, B, D, H, W, C, seed=9)
    pts = feats.npu().contiguous()
    cs = coords.npu().contiguous()
    with pytest.raises(RuntimeError):
        bev_pool(pts, cs, 0, D, H, W)   # B=0
    with pytest.raises(RuntimeError):
        bev_pool(pts, cs, B, 0, H, W)   # D=0
    with pytest.raises(RuntimeError):
        bev_pool(pts, cs, B, D, 0, W)   # H=0
    with pytest.raises(RuntimeError):
        bev_pool(pts, cs, B, D, H, 0)   # W=0


def test_coords_wrong_columns():
    """coords 列数不为 4 应被拒绝（wrapper 在 rank 计算前检查）。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 50
    feats, coords = _make_points(N, B, D, H, W, C, seed=10)
    pts = feats.npu().contiguous()
    cs = coords.npu().contiguous()
    bad_coords = cs[:, :3]  # 只有 3 列
    with pytest.raises((RuntimeError, IndexError)):
        bev_pool(pts, bad_coords, B, D, H, W)