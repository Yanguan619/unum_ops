"""Ascend 310P BevPool 性能基准测试。

对比 AscendC 算子与纯 torch scatter_add 参考实现耗时。

运行方式:
    python -m pytest benchmark/bench_bev_pool.py -v
"""
import os
import time

import numpy as np
import pytest
import torch
import torch_npu

from unum_ops.bev_pool import bev_pool, bev_pool_torch

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(0)

WARMUP = 5
REPEAT = 20
_RESULTS_FILE = os.path.join(os.path.dirname(__file__), "output", "bev_pool_results.txt")
B, D, H, W, C = 1, 8, 100, 100, 80
_N_POINTS = [2000, 8000, 32000, 128000, 512000]
# 维度扫描：不同 C / 网格 / batch 组合
_DIM_SWEEPS = [
    ("C=16",    1, 8, 100, 100, 16),
    ("C=80",    1, 8, 100, 100, 80),
    ("C=256",   1, 8, 100, 100, 256),
    ("小网格",   1, 8, 50, 50, 80),
    ("大网格",   1, 8, 200, 200, 80),
    ("D=16",    1, 16, 100, 100, 80),
    ("多batch", 4, 8, 100, 100, 80),
]


def _time_ms(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def test_bev_pool_bench():
    """BevPool 算子性能基准（AscendC vs torch scatter_add）。"""
    rng = torch.Generator()
    rng.manual_seed(7)
    results = []

    print(f"\n{'num_points':>10} {'torch_ms':>12} {'ascendc_ms':>12}")
    print("-" * 40)

    for N in _N_POINTS:
        feats = torch.randn(N, C, generator=rng, dtype=torch.float32)
        coords = torch.stack([
            torch.randint(0, W, (N,), generator=rng, dtype=torch.int64),
            torch.randint(0, H, (N,), generator=rng, dtype=torch.int64),
            torch.randint(0, D, (N,), generator=rng, dtype=torch.int64),
            torch.randint(0, B, (N,), generator=rng, dtype=torch.int64),
        ], dim=1)
        pts = feats.npu().contiguous()
        cs = coords.npu().contiguous()

        asc_mean, asc_std = _time_ms(lambda: bev_pool(pts, cs, B, D, H, W))
        torch_mean, torch_std = _time_ms(lambda: bev_pool_torch(feats, coords, B, D, H, W))

        results.append((N, asc_mean, asc_std, torch_mean, torch_std))
        print(f"{N:>10} {torch_mean:>10.3f} ±{torch_std:.3f}  {asc_mean:>10.3f} ±{asc_std:.3f}")

    os.makedirs(os.path.dirname(_RESULTS_FILE), exist_ok=True)
    with open(_RESULTS_FILE, "w") as f:
        f.write(f"B={B} D={D} H={H} W={W} C={C}\n")
        f.write(f"{'num_points':>10} {'torch_ms':>12} {'torch_std':>12} {'ascendc_ms':>12} {'ascendc_std':>12}\n")
        for N, am, as_, tm, ts in results:
            f.write(f"{N:>10} {tm:>12.3f} {ts:>12.3f} {am:>12.3f} {as_:>12.3f}\n")
    print(f"\nResults saved to {_RESULTS_FILE}")


def test_bev_pool_bench_dim_sweep():
    """维度扫描：不同 C/D/H/W/batch 组合下的性能。"""
    rng = torch.Generator()
    rng.manual_seed(11)
    N = 32000
    print(f"\n{'配置':<10} {'B×D×H×W×C':<22} {'ascendc_ms':>12} {'torch_ms':>12}")
    print("-" * 56)

    for label, b, d, h, w, c in _DIM_SWEEPS:
        feats = torch.randn(N, c, generator=rng, dtype=torch.float32)
        coords = torch.stack([
            torch.randint(0, w, (N,), generator=rng, dtype=torch.int64),
            torch.randint(0, h, (N,), generator=rng, dtype=torch.int64),
            torch.randint(0, d, (N,), generator=rng, dtype=torch.int64),
            torch.randint(0, b, (N,), generator=rng, dtype=torch.int64),
        ], dim=1)
        pts = feats.npu().contiguous()
        cs = coords.npu().contiguous()

        asc_mean, asc_std = _time_ms(lambda: bev_pool(pts, cs, b, d, h, w))
        torch_mean, torch_std = _time_ms(lambda: bev_pool_torch(feats, coords, b, d, h, w))

        print(f"{label:<10} {b}×{d}×{h}×{w}×{c:<10} {asc_mean:>10.3f} ±{asc_std:.3f} {torch_mean:>10.3f} ±{torch_std:.3f}")