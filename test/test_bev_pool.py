"""Ascend 310P BevPool 算子正确性测试。

覆盖 unum_ops.bev_pool 接口，与 torch-native index_add_ 参考实现对比。

本文件同时支持两种运行环境：
  - NPU 环境：执行 AscendC kernel 正确性测试（TestBEVPoolNPU）
  - 纯 CUDA 环境：执行 scatter_add 参考实现的 CUDA/CPU 交叉验证（TestBEVPoolVsCUDA）

运行方式:
    cd /data/workspace/unum_ops
    PYTHONPATH=".venv/lib/python3.11/site-packages:$PYTHONPATH" \
        pytest test/test_bev_pool.py -v
"""
import gc
import time

import numpy as np
import pytest
import torch

try:
    import torch_npu
    NPU_AVAIL = torch.npu.is_available()
except Exception:
    NPU_AVAIL = False

if NPU_AVAIL:
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(0)

from unum_ops.bev_pool import bev_pool, bev_pool_torch


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
    if not NPU_AVAIL:
        yield
        return
    feats, coords = _make_random_points(100, 1, 2, 4, 4, 8, seed=1)
    bev_pool(feats.npu().contiguous(), coords.npu().contiguous(), 1, 2, 4, 4)
    torch.npu.synchronize()
    yield


NPU_SKIP = pytest.mark.skipif(not NPU_AVAIL, reason="NPU not available")


# ── AscendC kernel 测试（NPU 环境） ───────────────────────────────────────────

@NPU_SKIP
def test_small_random():
    """小规模随机数据，与 CPU index_add_ 参考对比。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 50
    feats, coords = _make_random_points(N, B, D, H, W, C)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5, rtol=1e-5), \
        f"small random mismatch: max diff={torch.abs(out.cpu() - ref).max().item()}"


@NPU_SKIP
def test_medium_random():
    """中等规模随机数据。"""
    B, D, H, W, C = 2, 4, 8, 8, 16
    N = 500
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=123)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5, rtol=1e-5), \
        "medium random mismatch"


@NPU_SKIP
def test_multibatch():
    """多 batch 测试。"""
    B, D, H, W, C = 2, 3, 5, 5, 8
    N = 200
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=456)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5, rtol=1e-5)


@NPU_SKIP
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


@NPU_SKIP
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


@NPU_SKIP
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


@NPU_SKIP
def test_empty_points():
    """空点云（N=0）。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    feats = torch.empty(0, C, dtype=torch.float32)
    coords = torch.empty(0, 4, dtype=torch.int64)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5)


@NPU_SKIP
def test_large_channels():
    """较多通道数（C=80，BEVFusion 典型值）。"""
    B, D, H, W, C = 1, 2, 4, 4, 80
    N = 100
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=789)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-4, rtol=1e-4), \
        "large channels mismatch"


# ── Output invariants ──────────────────────────────────────────────────────

@NPU_SKIP
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


@NPU_SKIP
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

@NPU_SKIP
def test_all_oob_points():
    """所有点都越界 → 输出全零。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 20
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=6)
    coords_oob = coords.clone()
    coords_oob[:, 0] = -2  # 全部 x 越界
    out = _run(feats, coords_oob, B, D, H, W)
    assert torch.all(out.cpu() == 0.0), "all OOB points should produce zero output"


@NPU_SKIP
def test_mixed_oob_points():
    """部分点越界：越界点的坐标不应污染有效 voxel 的累加。"""
    B, D, H, W, C = 1, 1, 2, 2, 2
    feats = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                         dtype=torch.float32)
    coords = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0],
                           [-1, 0, 0, 0], [5, 0, 0, 0]], dtype=torch.int64)  # 后两行越界
    out = _run(feats, coords, B, D, H, W)
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5)


@NPU_SKIP
def test_rank_int64_no_float_collision():
    """rank 用 int64 排序，大网格下不因 float32 精度碰撞而错。"""
    # 构造 W < H 且 rank 值很大的情况（旧公式 float 会碰撞）
    B, D, H, W, C = 2, 3, 10, 4, 8
    N = 500
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=11)
    # 随机打乱输入顺序（Python 层负责排序）
    perm = torch.randperm(N)
    feats_shuf, coords_shuf = feats[perm], coords[perm]
    ref = bev_pool_ref(feats, coords, B, D, H, W)
    out = _run(feats_shuf, coords_shuf, B, D, H, W)
    assert torch.allclose(out.cpu(), ref, atol=1e-5), \
        f"int64 rank mismatch: max diff={torch.abs(out.cpu() - ref).max().item()}"


@NPU_SKIP
def test_non_contiguous_input():
    """非连续输入应被拒绝（binding TORCH_CHECK 抛错）。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 50
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=7)
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


@NPU_SKIP
def test_wrong_dtype_input():
    """错误 dtype 输入应被拒绝（直接调底层算子）。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 50
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=8)
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


@NPU_SKIP
def test_invalid_grid_dims():
    """非正网格维度应被拒绝。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 50
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=9)
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


@NPU_SKIP
def test_coords_wrong_columns():
    """coords 列数不为 4 应被拒绝（wrapper 在 rank 计算前检查）。"""
    B, D, H, W, C = 1, 2, 4, 4, 8
    N = 50
    feats, coords = _make_random_points(N, B, D, H, W, C, seed=10)
    pts = feats.npu().contiguous()
    cs = coords.npu().contiguous()
    bad_coords = cs[:, :3]  # 只有 3 列
    with pytest.raises((RuntimeError, IndexError)):
        bev_pool(pts, bad_coords, B, D, H, W)


# ============================================================
# bev_pool 参考实现 CUDA/CPU 交叉验证（纯 CUDA 环境可运行）
# 验证 scatter_add 参考与 unum_ops bev_pool 使用的语义一致
# ============================================================

def bev_pool_ref_cuda(feats, coords, B, D, H, W):
    """CUDA 参考实现：torch.scatter_add_ 按 voxel 索引累加。

    输出布局与 bev_pool_torch / AscendC kernel 一致：(B, C, D, H, W)。
    """
    feats = feats.cuda().contiguous()
    coords = coords.cuda().contiguous()
    N, C = feats.shape
    rank = (coords[:, 0] + coords[:, 1] * W
            + coords[:, 2] * (W * H) + coords[:, 3] * (W * H * D))
    out = torch.zeros(B * D * H * W, C, dtype=torch.float32, device="cuda")
    out.index_add_(0, rank, feats)
    return out.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBEVPoolVsCUDA:
    def test_cuda_matches_cpu_reference(self):
        """CUDA scatter_add 与纯 torch 参考（CPU scatter_add）一致"""
        B, D, H, W, C = 1, 2, 4, 4, 8
        N = 50
        feats, coords = _make_random_points(N, B, D, H, W, C)
        ref_cpu = bev_pool_ref(feats, coords, B, D, H, W)
        ref_cuda = bev_pool_ref_cuda(feats, coords, B, D, H, W)
        assert torch.allclose(ref_cpu, ref_cuda.cpu(), atol=1e-5, rtol=1e-5), \
            "CUDA scatter_add should match CPU reference"

    def test_medium_random(self):
        """中等规模随机数据"""
        B, D, H, W, C = 2, 4, 8, 8, 16
        N = 500
        feats, coords = _make_random_points(N, B, D, H, W, C, seed=123)
        ref_cpu = bev_pool_ref(feats, coords, B, D, H, W)
        ref_cuda = bev_pool_ref_cuda(feats, coords, B, D, H, W)
        assert torch.allclose(ref_cpu, ref_cuda.cpu(), atol=1e-5, rtol=1e-5)

    def test_multiple_points_same_voxel(self):
        """多点同一 voxel 累加"""
        B, D, H, W, C = 1, 1, 1, 1, 4
        feats = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                              [2.0, 2.0, 2.0, 2.0],
                              [3.0, 3.0, 3.0, 3.0]], dtype=torch.float32)
        coords = torch.tensor([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]], dtype=torch.int64)
        ref = bev_pool_ref_cuda(feats, coords, B, D, H, W)
        expected = feats.sum(dim=0).reshape(B, C, D, H, W)
        assert torch.allclose(ref.cpu(), expected, atol=1e-5)

    def test_random_seeds(self):
        """多种子下 CUDA scatter_add 与 CPU 参考一致"""
        for seed in range(3):
            B, D, H, W, C = 1, 2, 4, 4, 8
            N = 100
            feats, coords = _make_random_points(N, B, D, H, W, C, seed=seed)
            ref_cpu = bev_pool_ref(feats, coords, B, D, H, W)
            ref_cuda = bev_pool_ref_cuda(feats, coords, B, D, H, W)
            assert torch.allclose(ref_cpu, ref_cuda.cpu(), atol=1e-5, rtol=1e-5), \
                f"seed={seed} mismatch"

    def test_empty_points(self):
        """空点云应全零"""
        B, D, H, W, C = 1, 2, 4, 4, 8
        feats = torch.empty(0, C, dtype=torch.float32)
        coords = torch.empty(0, 4, dtype=torch.int64)
        ref = bev_pool_ref_cuda(feats, coords, B, D, H, W)
        assert torch.all(ref.cpu() == 0.0)


# ============================================================
# 高压精度对比（高碰撞率场景）
# ============================================================

def _bev_pool_ref_index_add(feats, coords, B, D, H, W):
    """Atomic scatter-add reference (matches v3 AscendC summation order)."""
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


@pytest.mark.skipif(not NPU_AVAIL, reason="NPU required")
class TestBEVPoolHighCollision:
    """High-collision precision comparison.

    设计：
      - 网格小 → voxel 数少 → 每 voxel 平均点多 → 强制累积顺序发生作用
      - N 大 → 同样意义
      - C 多 → channel 维度也参与累加

    如果 max_diff 仍然 0，说明 unum_ops AscendC 的 UB 顺序累加恰好等价于
    atomic add（不是偶然，而是 N·ε 在 fp32 的舍入路径上对称分布）。
    """

    @staticmethod
    def make_input(N, B, D, H, W, C, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        feats_cpu = torch.randn(N, C, dtype=torch.float32, generator=g)
        coords_cpu = torch.zeros(N, 4, dtype=torch.int32)
        coords_cpu[:, 0] = torch.randint(0, W, (N,), generator=g, dtype=torch.int32)
        coords_cpu[:, 1] = torch.randint(0, H, (N,), generator=g, dtype=torch.int32)
        coords_cpu[:, 2] = torch.randint(0, D, (N,), generator=g, dtype=torch.int32)
        coords_cpu[:, 3] = torch.randint(0, B, (N,), generator=g, dtype=torch.int32)
        return feats_cpu.to("npu"), coords_cpu.to("npu")

    @pytest.mark.parametrize("cfg", [
        # (name, B, D, H, W, C, N, seed)
        # 实际 BEV fusion 场景：avg/voxel ≤ 50，voxel 数 1k~16k
        ("mid_grid_100k",   2,  8, 32,  32,  80, 100000, 2),  # 16384 voxels, ~6 avg/voxel
        ("mid_grid_200k",   2,  8, 32,  32,  80, 200000, 3),  # ~12 avg/voxel
        ("dense_voxel_50k", 1,  4, 16,  16,  80,  50000, 1),  # 1024 voxels, ~49 avg/voxel
        ("dense_large_C",   1,  4, 16,  16, 256,  50000, 4),  # 1024 voxels, ~49 avg/voxel, C=256
    ])
    def test_high_collision(self, cfg):
        name, B, D, H, W, C, N, seed = cfg
        feats, coords = self.make_input(N, B, D, H, W, C, seed)
        ref = _bev_pool_ref_index_add(feats.cpu(), coords.cpu(), B, D, H, W).to("npu")
        u = bev_pool(feats, coords, B, D, H, W).out
        d = (u.float() - ref.float()).abs()
        n_voxel = B * D * H * W
        avg_per_voxel = N / n_voxel
        n_diff = (d > 0).sum().item()
        print(
            f"\n[{name:>20}] N={N:>6} voxels={n_voxel:>6} avg/voxel={avg_per_voxel:>6.1f} C={C:>3}"
            f"  | max_diff={d.max().item():.3e}  mean_diff={d.mean().item():.3e}"
            f"  | nonzero_voxels={n_diff:>6}/{ref.numel()//C}"
        )
        # 硬性断言：即使是 fp32 atomic 差异，也应该 < 1e-4 数量级
        # （unum vs torch 实测 0；real-world 应至少 6 个数量级精度）
        assert d.max().item() < 1e-4, (
            f"[{name}] max_diff {d.max().item():.3e} 超出 fp32 累加误差上限，"
            f"可能是 correctness bug 而非 precision"
        )


# ============================================================
# 连续调用稳定性（bev_pool 长稳压测）
# ============================================================

def _bev_pool_once(B, D, H, W, C, N, seed):
    """单次 bev_pool 调用，返回正确性 max_diff。"""
    torch.manual_seed(seed)
    feats = torch.randn(N, C, dtype=torch.float32)
    coords = torch.zeros(N, 4, dtype=torch.int64)
    coords[:, 0] = torch.randint(0, W, (N,))
    coords[:, 1] = torch.randint(0, H, (N,))
    coords[:, 2] = torch.randint(0, D, (N,))
    coords[:, 3] = torch.randint(0, B, (N,))
    ref = bev_pool_torch(feats, coords, B, D, H, W).out
    pts = feats.npu().contiguous()
    cs = coords.npu().contiguous()
    out = bev_pool(pts, cs, B, D, H, W)
    torch.npu.synchronize()
    return (out.out.cpu() - ref).abs().max().item()


@pytest.mark.skipif(not NPU_AVAIL, reason="NPU required")
class TestBEVPoolStability:
    """bev_pool 连续多轮调用稳定性（无设备错误、正确性不退化、显存不泄漏）。"""

    def test_small_loop(self):
        """小参数长稳：500 次随机参数连续调用。"""
        errors = []
        for i in range(1, 501):
            try:
                B = np.random.randint(1, 3)
                D = np.random.randint(1, 4)
                H = np.random.randint(4, 16)
                W = np.random.randint(4, 16)
                C = int(np.random.choice([8, 16, 32, 64, 80]))
                N = int(np.random.randint(10, 500))
                diff = _bev_pool_once(B, D, H, W, C, N, seed=i * 100)
                if diff > 1e-5:
                    errors.append((i, f"diff={diff:.3e}"))
            except Exception as e:
                errors.append((i, str(e)[:80]))
            if i % 200 == 0:
                gc.collect()
                torch.npu.synchronize()
                torch.npu.empty_cache()
        assert not errors, f"bev_pool 长稳压测失败：{len(errors)} 个错误, 前5: {errors[:5]}"

    def test_large_loop(self):
        """大负载长稳：250 次，含 512k 点 bev_pool。"""
        errors = []
        for i in range(1, 251):
            try:
                B, D, H, W, C = 1, 8, 100, 100, 80
                N = 512000 if i % 4 == 0 else 50000
                diff = _bev_pool_once(B, D, H, W, C, N, seed=i)
                tol = 1e-4 if N == 512000 else 1e-5
                if diff > tol:
                    errors.append((i, f"diff={diff:.3e}"))
            except Exception as e:
                errors.append((i, str(e)[:80]))
            if i % 100 == 0:
                gc.collect()
                torch.npu.synchronize()
                torch.npu.empty_cache()
        assert not errors, f"bev_pool 大负载长稳压测失败：{len(errors)} 个错误, 前5: {errors[:5]}"


if __name__ == "__main__":
    t0 = time.perf_counter()
    TestBEVPoolStability().test_small_loop()
    TestBEVPoolStability().test_large_loop()
    print(f"bev_pool 长稳压测全部通过, 总耗时 {time.perf_counter() - t0:.0f}s")
