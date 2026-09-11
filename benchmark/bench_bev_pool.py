"""Ascend 310P BevPool 性能基准测试（Triton-free + tabulate 输出）。

对比 AscendC 算子与纯 torch scatter_add 参考实现耗时。
torch / ascendc 作为表头列，num_points 或配置作为行。

运行方式:
    python benchmark/bench_bev_pool.py
    python -m pytest benchmark/bench_bev_pool.py -v
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch_npu

from bench_utils import Benchmark, do_bench, perf_report
from unum_ops.bev_pool import bev_pool, bev_pool_torch

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("UNUM_BENCH_DEVICE", "0")))

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
_QUANTILES = [0.5, 0.2, 0.8]


def _make_inputs(N, b, d, h, w, c, seed):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(N, c, generator=g, dtype=torch.float32)
    coords = torch.stack([
        torch.randint(0, w, (N,), generator=g, dtype=torch.int64),
        torch.randint(0, h, (N,), generator=g, dtype=torch.int64),
        torch.randint(0, d, (N,), generator=g, dtype=torch.int64),
        torch.randint(0, b, (N,), generator=g, dtype=torch.int64),
    ], dim=1)
    return feats, coords


def _bench(feats, coords, b, d, h, w):
    torch_ms = do_bench(lambda: bev_pool_torch(feats, coords, b, d, h, w),
                        quantiles=_QUANTILES)
    pts = feats.npu().contiguous()
    cs = coords.npu().contiguous()
    asc_ms = do_bench(lambda: bev_pool(pts, cs, b, d, h, w),
                      grad_to_none=[pts, cs], quantiles=_QUANTILES)
    return {"torch(ms)": torch_ms, "ascendc(ms)": asc_ms}


@perf_report(
    Benchmark(
        x_names=["num_points"],
        x_vals=_N_POINTS,
        x_log=True,
        plot_name="bev_pool",
        ylabel="Latency (ms)",
    ),
)
def bench_npoints(num_points):
    feats, coords = _make_inputs(num_points, B, D, H, W, C, seed=7)
    return _bench(feats, coords, B, D, H, W)


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[label for label, *_ in _DIM_SWEEPS],
        plot_name="bev_pool_dim_sweep",
        ylabel="Latency (ms)",
    ),
)
def bench_dimsweep(config):
    N = 32000
    b, d, h, w, c = next(dims for label, *dims in _DIM_SWEEPS if label == config)
    feats, coords = _make_inputs(N, b, d, h, w, c, seed=11)
    return _bench(feats, coords, b, d, h, w)


def _run_all():
    bench_npoints.run(print_data=True, show_plots=True)
    bench_dimsweep.run(print_data=True, show_plots=True)


def test_bev_pool_bench():
    bench_npoints.run(print_data=True, show_plots=True)


def test_bev_pool_bench_dim_sweep():
    bench_dimsweep.run(print_data=True, show_plots=True)


if __name__ == "__main__":
    _run_all()