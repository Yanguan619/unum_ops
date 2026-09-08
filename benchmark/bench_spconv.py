"""spconv 算子性能回归基准（Triton-free + tabulate 输出）。

对比 SubMConv3d / SparseConv3d / SparseInverseConv3d 在不同
体素数/通道数/空间尺寸下的前向耗时。对比项作为表头列，config 作为行。

CPU：首次构建 / 缓存命中的 torch 实现耗时。
NPU：torch einsum vs AscendC 内核（完整卷积前向）。

运行方式:
    python benchmark/bench_spconv.py
    python -m pytest benchmark/bench_spconv.py -v
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "unum_ops"))
sys.path.insert(0, _HERE)

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_npu

from bench_utils import Benchmark, do_bench, perf_report
from spconv import ascendc
import spconv.conv as conv_mod
from spconv.conv import SubMConv3d, SparseConv3d, SparseInverseConv3d
from spconv.sparse_modules import SparseConvTensor, SparseSequential

NPU_AVAIL = hasattr(torch, "npu") and torch.npu.is_available()
if NPU_AVAIL:
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.set_device(int(os.environ.get("UNUM_BENCH_DEVICE", "0")))
    ascendc.available()

WARMUP = 3
REPEAT = 10
_QUANTILES = [0.5, 0.2, 0.8]

_SUBM_CONFIGS = [
    ("N=200 C=16 (8,8,8)",     200, 16, (8, 8, 8)),
    ("N=500 C=32 (16,16,16)",  500, 32, (16, 16, 16)),
    ("N=1000 C=64 (32,32,32)", 1000, 64, (32, 32, 32)),
]
_SPARSE_CONFIGS = [
    ("N=200 C=16 (8,8,8)",     200, 16, (8, 8, 8)),
    ("N=500 C=32 (16,16,16)",  500, 32, (16, 16, 16)),
]
_INVERSE_CONFIGS = [
    ("N=200 C=16 (8,8,8)",     200, 16, (8, 8, 8)),
    ("N=500 C=32 (16,16,16)",  500, 32, (16, 16, 16)),
]
_NPU_CONFIGS = [
    ("N=200 C=16 (8,8,8)",     200, 16, (8, 8, 8)),
    ("N=500 C=32 (16,16,16)",  500, 32, (16, 16, 16)),
    ("N=1000 C=64 (32,32,32)", 1000, 64, (32, 32, 32)),
]


def make_tensor(N, C, spatial, batch_size=1, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    x_dim, y_dim, z_dim = spatial
    coords = set()
    while len(coords) < N:
        xs = torch.randint(0, x_dim, (N * 4,), generator=g).tolist()
        ys = torch.randint(0, y_dim, (N * 4,), generator=g).tolist()
        zs = torch.randint(0, z_dim, (N * 4,), generator=g).tolist()
        for i in range(len(xs)):
            coords.add((0, xs[i], ys[i], zs[i]))
            if len(coords) >= N:
                break
    indices = torch.tensor(sorted(coords)[:N], dtype=torch.int32).to(device)
    features = torch.randn(indices.shape[0], C, generator=g).to(device)
    return SparseConvTensor(features, indices, spatial, batch_size)


# ----------------------------------------------------------------------
# CPU: 首次构建 / 缓存命中（torch 实现）
# ----------------------------------------------------------------------

@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _SUBM_CONFIGS],
        plot_name="spconv_subm",
        ylabel="Latency (ms)",
    ),
)
def bench_subm(config):
    N, C, spatial = next(c[1:] for c in _SUBM_CONFIGS if c[0] == config)
    x = make_tensor(N, C, spatial, seed=7)
    conv = SubMConv3d(C, C, 3, padding=1, bias=True).eval()
    conv._nb_cache = {}
    first_ms = do_bench(lambda: conv(x).features, warmup=1, rep=5, quantiles=_QUANTILES)
    conv(x)
    cache_ms = do_bench(lambda: conv(x).features, warmup=1, rep=REPEAT, quantiles=_QUANTILES)
    return {"first(ms)": first_ms, "cache(ms)": cache_ms}


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _SPARSE_CONFIGS],
        plot_name="spconv_sparse_conv",
        ylabel="Latency (ms)",
    ),
)
def bench_sparse_conv(config):
    N, C, spatial = next(c[1:] for c in _SPARSE_CONFIGS if c[0] == config)
    x = make_tensor(N, C, spatial, seed=8)
    conv = SparseConv3d(C, 2 * C, 3, stride=2, padding=1, bias=True).eval()
    conv._nb_cache = {}
    first_ms = do_bench(lambda: conv(x).features, warmup=1, rep=5, quantiles=_QUANTILES)
    conv(x)
    cache_ms = do_bench(lambda: conv(x).features, warmup=1, rep=REPEAT, quantiles=_QUANTILES)
    return {"first(ms)": first_ms, "cache(ms)": cache_ms}


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _INVERSE_CONFIGS],
        plot_name="spconv_inverse_conv",
        ylabel="Latency (ms)",
    ),
)
def bench_inverse_conv(config):
    N, C, spatial = next(c[1:] for c in _INVERSE_CONFIGS if c[0] == config)
    x = make_tensor(N, C, spatial, seed=9)
    down = SparseConv3d(C, 2 * C, 3, stride=2, padding=1, bias=True, indice_key="up").eval()
    y = down(x)
    z = SparseConvTensor(torch.randn(y.features.shape[0], 2 * C), y.indices,
                         y.spatial_shape, x.batch_size)
    z._indice_dict = dict(y._indice_dict)
    up = SparseInverseConv3d(2 * C, C, 3, indice_key="up", bias=True).eval()
    fwd_ms = do_bench(lambda: up(z).features, warmup=1, rep=REPEAT, quantiles=_QUANTILES)
    return {"fwd(ms)": fwd_ms}


# ----------------------------------------------------------------------
# NPU: torch einsum vs AscendC 内核（完整卷积前向）
# ----------------------------------------------------------------------

_NPU_ASCENDC = NPU_AVAIL and ascendc.available()


def _bench_npu(config, conv_type, seed):
    """测量 SubMConv3d / SparseConv3d 在 NPU 上的 torch 和 AscendC 耗时。"""
    N, C, spatial = next(c[1:] for c in _NPU_CONFIGS if c[0] == config)
    x = make_tensor(N, C, spatial, seed=seed, device="npu")
    if conv_type == "subm":
        conv = SubMConv3d(C, C, 3, padding=1, bias=True).eval().to("npu")
    else:
        conv = SparseConv3d(C, 2 * C, 3, stride=2, padding=1, bias=True).eval().to("npu")
    conv(x)

    conv_mod._USE_ASCENDC = False
    feats = conv(x).features
    torch_ms = do_bench(lambda: conv(x).features, grad_to_none=[feats],
                        quantiles=_QUANTILES)

    result = {"torch(ms)": torch_ms}
    if _NPU_ASCENDC:
        conv_mod._USE_ASCENDC = True
        feats = conv(x).features
        ascendc_ms = do_bench(lambda: conv(x).features, grad_to_none=[feats],
                              quantiles=_QUANTILES)
        result["ascendc(ms)"] = ascendc_ms
    conv_mod._USE_ASCENDC = False
    return result


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _NPU_CONFIGS],
        plot_name="spconv_subm_npu",
        ylabel="Latency (ms)",
    ),
)
def bench_subm_npu(config):
    return _bench_npu(config, "subm", seed=7)


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _NPU_CONFIGS],
        plot_name="spconv_sparse_conv_npu",
        ylabel="Latency (ms)",
    ),
)
def bench_sparse_conv_npu(config):
    return _bench_npu(config, "sparse", seed=8)


def _run_all():
    bench_subm.run(print_data=True, show_plots=True)
    bench_sparse_conv.run(print_data=True, show_plots=True)
    bench_inverse_conv.run(print_data=True, show_plots=True)
    if NPU_AVAIL:
        bench_subm_npu.run(print_data=True, show_plots=True)
        bench_sparse_conv_npu.run(print_data=True, show_plots=True)


def test_subm_bench():
    bench_subm.run(print_data=True, show_plots=True)


def test_sparse_conv_bench():
    bench_sparse_conv.run(print_data=True, show_plots=True)


def test_inverse_conv_bench():
    bench_inverse_conv.run(print_data=True, show_plots=True)


@pytest.mark.skipif(not NPU_AVAIL, reason="NPU not available")
def test_subm_npu_bench():
    bench_subm_npu.run(print_data=True, show_plots=True)


@pytest.mark.skipif(not NPU_AVAIL, reason="NPU not available")
def test_sparse_conv_npu_bench():
    bench_sparse_conv_npu.run(print_data=True, show_plots=True)


if __name__ == "__main__":
    _run_all()