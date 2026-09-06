"""Ascend 310P Voxelization 性能基准测试。

对比 AscendC 算子与 numpy VoxelGeneratorV2 参考实现的耗时。

运行方式:
    python -m pytest benchmark/bench_voxelization.py -v
"""
import os
import time

import numpy as np
import pytest
import torch
import torch_npu

_HERE = os.path.dirname(os.path.abspath(__file__))

from unum_ops.voxelization import voxelization
from unum_ops.spconv.utils import VoxelGeneratorV2

DEFAULT_VOXEL_SIZE = (0.16, 0.16, 4.0)
DEFAULT_PCR = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
WARMUP = 5
REPEAT = 20

_N_POINTS = [5000, 20000, 50000, 100000]
_RESULTS_FILE = os.path.join(_HERE, "output", "voxelization_results.txt")

# 不同参数组合
_PARAM_SWEEPS = [
    ("默认",    DEFAULT_VOXEL_SIZE, DEFAULT_PCR),
    ("大voxel", (0.5, 0.5, 4.0),   DEFAULT_PCR),
    ("窄范围",  (0.16, 0.16, 4.0), (0, -20, -3, 40, 20, 1)),
]

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(0)


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


def test_voxelization_bench():
    """Voxelization 算子性能基准（AscendC vs numpy）。"""
    gen = VoxelGeneratorV2(DEFAULT_VOXEL_SIZE, DEFAULT_PCR, 32, 40000)
    results = []

    print(f"\n{'num_points':>12} {'ascendc_ms':>12} {'numpy_ms':>12}")
    print("-" * 40)

    for N in _N_POINTS:
        np.random.seed(N)
        points = np.random.randn(N, 4).astype(np.float32)
        pts = torch.from_numpy(points.copy()).npu()

        ac_mean, ac_std = _time_ms(lambda: voxelization(pts))
        np_repeat = max(3, min(REPEAT, 20 * 5000 // N))
        np_mean, np_std = _time_ms(lambda: gen.generate(points), warmup=2, repeat=np_repeat)

        results.append((N, ac_mean, ac_std, np_mean, np_std))
        print(f"{N:>12} {ac_mean:>10.3f} ±{ac_std:.3f}  {np_mean:>10.3f} ±{np_std:.3f}")

    os.makedirs(os.path.dirname(_RESULTS_FILE), exist_ok=True)
    with open(_RESULTS_FILE, "w") as f:
        f.write(f"{'num_points':>12} {'ascendc_ms':>12} {'ascendc_std':>12} {'numpy_ms':>12} {'numpy_std':>12}\n")
        for N, ac_mean, ac_std, np_mean, np_std in results:
            f.write(f"{N:>12} {ac_mean:>12.3f} {ac_std:>12.3f} {np_mean:>12.3f} {np_std:>12.3f}\n")


def test_voxelization_bench_param_sweep():
    """参数扫描：不同 voxel_size / PCR 下的性能。"""
    N = 50000
    print(f"\n{'配置':<12} {'ascendc_ms':>12}")
    print("-" * 28)

    for label, vs, pcr in _PARAM_SWEEPS:
        np.random.seed(42)
        points = np.random.randn(N, 4).astype(np.float32)
        pts = torch.from_numpy(points.copy()).npu()
        ac_mean, ac_std = _time_ms(lambda: voxelization(pts, voxel_size=vs, pcr=pcr))
        print(f"{label:<12} {ac_mean:>10.3f} ±{ac_std:.3f}")