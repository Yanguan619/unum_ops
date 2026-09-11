"""Ascend 310P Voxelization 性能基准测试（Triton-free + tabulate 输出）。

对比 AscendC 算子与 numpy VoxelGeneratorV2 参考实现的耗时。
ascendc / numpy 作为表头列，num_points 或配置作为行。

运行方式:
    python benchmark/bench_voxelization.py
    python -m pytest benchmark/bench_voxelization.py -v
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch_npu

from bench_utils import Benchmark, do_bench, perf_report
from unum_ops.voxelization import voxelization
from unum_ops.spconv.utils import VoxelGeneratorV2

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("UNUM_BENCH_DEVICE", "0")))

DEFAULT_VOXEL_SIZE = (0.16, 0.16, 4.0)
DEFAULT_PCR = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
_N_POINTS = [5000, 20000, 50000, 100000]
_PARAM_SWEEPS = [
    ("默认",    DEFAULT_VOXEL_SIZE, DEFAULT_PCR),
    ("大voxel", (0.5, 0.5, 4.0),   DEFAULT_PCR),
    ("窄范围",  (0.16, 0.16, 4.0), (0.0, -20.0, -3.0, 40.0, 20.0, 1.0)),
]
_QUANTILES = [0.5, 0.2, 0.8]


def _make_points(N, seed):
    np.random.seed(seed)
    return np.random.randn(N, 4).astype(np.float32)


@perf_report(
    Benchmark(
        x_names=["num_points"],
        x_vals=_N_POINTS,
        x_log=True,
        plot_name="voxelization",
        ylabel="Latency (ms)",
    ),
)
def bench_npoints(num_points):
    points = _make_points(num_points, seed=num_points)
    pts = torch.from_numpy(points.copy()).npu()
    ac_ms = do_bench(lambda: voxelization(pts), grad_to_none=[pts], quantiles=_QUANTILES)
    gen = VoxelGeneratorV2(DEFAULT_VOXEL_SIZE, DEFAULT_PCR, 32, 40000)
    np_ms = do_bench(lambda: gen.generate(points), quantiles=_QUANTILES)
    return {"ascendc(ms)": ac_ms, "numpy(ms)": np_ms}


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[label for label, *_ in _PARAM_SWEEPS],
        plot_name="voxelization_param_sweep",
        ylabel="Latency (ms)",
    ),
)
def bench_paramsweep(config):
    N = 50000
    vs, pcr = next((vs, pr) for label, vs, pr in _PARAM_SWEEPS if label == config)
    points = _make_points(N, seed=42)
    pts = torch.from_numpy(points.copy()).npu()
    ac_ms = do_bench(lambda: voxelization(pts, voxel_size=vs, pcr=pcr),
                     grad_to_none=[pts], quantiles=_QUANTILES)
    return {"ascendc(ms)": ac_ms}


def _run_all():
    bench_npoints.run(print_data=True, show_plots=True)
    bench_paramsweep.run(print_data=True, show_plots=True)


def test_voxelization_bench():
    bench_npoints.run(print_data=True, show_plots=True)


def test_voxelization_bench_param_sweep():
    bench_paramsweep.run(print_data=True, show_plots=True)


if __name__ == "__main__":
    _run_all()