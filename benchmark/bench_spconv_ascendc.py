"""spconv AscendC 内核性能基准（Triton-free + tabulate 输出）。

当前状态（310P）：
  - 标量 Axpy 内核已被 Cube 内核替换（op_extension 已编译安装，op_kernel 使用
    Matmul<GM,ND,half> 高层 API）。
  - 已知问题：Cube 内核输出写入 stride 为 baseM(1024) 而非 1，导致仅 49 行正确
    （每 1024 行块的首行）。需改用低层 Mad/mmad 原语或解决 Matmul API 的
    CLayout 配置。
  - 实际生产路径：torch aclnn 2D GEMM 已使用 Cube 加速（7ms @ 5 万体素），
    设置 UNUM_SPCONV_USE_ASCENDC=1 启用 Cube 内核（当前未默认开启）。

运行方式:
    python benchmark/bench_spconv_ascendc.py
    python -m pytest benchmark/bench_spconv_ascendc.py -v
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
from spconv.conv import SubMConv3d, SparseConv3d, SparseInverseConv3d
from spconv.sparse_modules import SparseConvTensor

torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.set_device(int(os.environ.get("UNUM_BENCH_DEVICE", "0")))

NPU_AVAIL = hasattr(torch, "npu") and torch.npu.is_available()
if NPU_AVAIL:
    ascendc.available()

_REPEAT = 10
_QUANTILES = [0.5, 0.2, 0.8]

_GEMM_CONFIGS = [
    ("K=1 N=1000 Cin=64 Cout=64",     1, 1000, 64, 64),
    ("K=1 N=5000 Cin=64 Cout=64",     1, 5000, 64, 64),
    ("K=3 N=1000 Cin=64 Cout=64",     3, 1000, 64, 64),
    ("K=27 N=500 Cin=64 Cout=64",     27, 500, 64, 64),
    ("K=27 N=1000 Cin=64 Cout=64",    27, 1000, 64, 64),
    ("K=27 N=5000 Cin=64 Cout=64",    27, 5000, 64, 64),
    ("K=125 N=200 Cin=16 Cout=16",    125, 200, 16, 16),
    ("K=8 N=500 Cin=128 Cout=128",    8, 500, 128, 128),
]
_SUBM_CONFIGS = [
    ("N=200 C=16 (8,8,8)",     200, 16, (8, 8, 8)),
    ("N=500 C=32 (16,16,16)",  500, 32, (16, 16, 16)),
]


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _GEMM_CONFIGS],
        line_arg="provider",
        line_vals=["ascendc", "einsum"],
        line_names=["ascendc(ms)", "einsum(ms)"],
        plot_name="spconv_gemm",
        ylabel="Latency (ms)",
    ),
)
def bench_gemm(config, provider):
    if not NPU_AVAIL:
        return 0.0
    K, N, Cin, Cout = next(c for c in _GEMM_CONFIGS if c[0] == config)
    g = torch.Generator().manual_seed(42)
    feats = torch.randn(K, N, Cin, generator=g).npu()
    weight = torch.randn(K, Cin, Cout, generator=g).npu()
    bias = torch.randn(Cout, generator=g).npu()
    params = torch.tensor([K, N, Cin, Cout, 1, 8], dtype=torch.int32, device="npu")
    grad_to_none = [feats, weight, bias, params]

    if provider == "ascendc":
        return do_bench(lambda: torch.ops.unum.spconv_gemm(feats, weight, bias, params),
                        grad_to_none=grad_to_none, quantiles=_QUANTILES)
    return do_bench(lambda: torch.einsum('kni,kio->no', feats, weight) + bias,
                    grad_to_none=grad_to_none, quantiles=_QUANTILES)


def _make_subm_input(N, C, spatial, seed, dev):
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
    indices = torch.tensor(sorted(coords)[:N], dtype=torch.int32)
    features = torch.randn(indices.shape[0], C, generator=g)
    x = SparseConvTensor(features, indices, spatial, 1)
    x_n = SparseConvTensor(x.features.to(dev), x.indices.to(dev), x.spatial_shape, 1)
    return x_n


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _SUBM_CONFIGS],
        line_arg="provider",
        line_vals=["subm"],
        line_names=["subm(ms)"],
        plot_name="spconv_subm_ascendc",
        ylabel="Latency (ms)",
    ),
)
def bench_subm(config, provider):
    if not NPU_AVAIL:
        return 0.0
    dev = torch.device("npu:0")
    N, C, spatial = next(c for c in _SUBM_CONFIGS if c[0] == config)
    x_n = _make_subm_input(N, C, spatial, seed=7, dev=dev)
    conv = SubMConv3d(C, C, 3, padding=1, bias=True).eval().to(dev)
    conv(x_n)
    feats = conv(x_n).features
    return do_bench(lambda: conv(x_n).features, grad_to_none=[feats], quantiles=_QUANTILES)


def _run_all():
    bench_gemm.run(print_data=True, show_plots=True)
    bench_subm.run(print_data=True, show_plots=True)


@pytest.mark.skipif(not NPU_AVAIL, reason="NPU not available")
def test_gemm_bench():
    bench_gemm.run(print_data=True, show_plots=True)


@pytest.mark.skipif(not NPU_AVAIL, reason="NPU not available")
def test_subm_bench():
    bench_subm.run(print_data=True, show_plots=True)


if __name__ == "__main__":
    _run_all()