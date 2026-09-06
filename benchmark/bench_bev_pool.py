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