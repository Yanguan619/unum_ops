"""spconv 算子性能回归基准。

对比 Conv3d / SubMConv3d / SparseConv3d / SparseInverseConv3d 在不同
体素数/通道数/空间尺寸下的前向耗时（含首次构建 + 缓存命中）。

运行方式:
    python -m pytest benchmark/bench_spconv.py -v
"""
import os
import time
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "unum_ops"))

import spconv
from spconv.conv import SubMConv3d, SparseConv3d, SparseInverseConv3d
from spconv.sparse_modules import SparseConvTensor, SparseSequential

WARMUP = 3
REPEAT = 10
_RESULTS_FILE = os.path.join(_HERE, "output", "spconv_results.txt")


def make_tensor(N, C, spatial, batch_size=1, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    D, H, W = spatial
    coords = set()
    while len(coords) < N:
        xs = torch.randint(0, W, (N * 4,), generator=g).tolist()
        ys = torch.randint(0, H, (N * 4,), generator=g).tolist()
        zs = torch.randint(0, D, (N * 4,), generator=g).tolist()
        for i in range(len(xs)):
            coords.add((0, xs[i], ys[i], zs[i]))
            if len(coords) >= N:
                break
    indices = torch.tensor(sorted(coords)[:N], dtype=torch.int32)
    features = torch.randn(indices.shape[0], C, generator=g)
    return SparseConvTensor(features, indices, spatial, batch_size)


def _time_ms(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


@pytest.mark.parametrize("N,C,spatial", [
    (200, 16, (8, 8, 8)),
    (500, 32, (16, 16, 16)),
    (1000, 64, (32, 32, 32)),
])
def test_subm_bench(N, C, spatial):
    """SubMConv3d 首次构建 + 缓存命中耗时。"""
    x = make_tensor(N, C, spatial, seed=7)
    conv = SubMConv3d(C, C, 3, padding=1, bias=True).eval()
    # 首次（含构建）
    conv._nb_cache = {}
    t_first, s_first = _time_ms(lambda: conv(x).features, warmup=1, repeat=5)
    # 缓存命中
    conv(x)
    t_hit, s_hit = _time_ms(lambda: conv(x).features, warmup=1, repeat=REPEAT)
    print(f"\nN={N} C={C} spatial={spatial}: 首次={t_first:.2f}±{s_first:.2f}ms  命中={t_hit:.2f}±{s_hit:.2f}ms")
    assert t_first > 0 and t_hit > 0


@pytest.mark.parametrize("N,C,spatial", [
    (200, 16, (8, 8, 8)),
    (500, 32, (16, 16, 16)),
])
def test_sparse_conv_bench(N, C, spatial):
    """SparseConv3d 首次构建 + 缓存命中耗时。"""
    x = make_tensor(N, C, spatial, seed=8)
    conv = SparseConv3d(C, 2 * C, 3, stride=2, padding=1, bias=True).eval()
    conv._nb_cache = {}
    t_first, s_first = _time_ms(lambda: conv(x).features, warmup=1, repeat=5)
    conv(x)
    t_hit, s_hit = _time_ms(lambda: conv(x).features, warmup=1, repeat=REPEAT)
    print(f"\nN={N} C={C} spatial={spatial}: 首次={t_first:.2f}±{s_first:.2f}ms  命中={t_hit:.2f}±{s_hit:.2f}ms")
    assert t_first > 0 and t_hit > 0


@pytest.mark.parametrize("N,C,spatial", [
    (200, 16, (8, 8, 8)),
    (500, 32, (16, 16, 16)),
])
def test_inverse_conv_bench(N, C, spatial):
    """SparseInverseConv3d 耗时。"""
    x = make_tensor(N, C, spatial, seed=9)
    down = SparseConv3d(C, 2 * C, 3, stride=2, padding=1, bias=True, indice_key="up").eval()
    y = down(x)
    z = SparseConvTensor(torch.randn(y.features.shape[0], 2 * C), y.indices,
                         y.spatial_shape, x.batch_size)
    z._indice_dict = dict(y._indice_dict)
    up = SparseInverseConv3d(2 * C, C, 3, indice_key="up", bias=True).eval()
    t_mean, t_std = _time_ms(lambda: up(z).features, warmup=1, repeat=REPEAT)
    print(f"\nN={N} C={C} spatial={spatial}: {t_mean:.2f}±{t_std:.2f}ms")
    assert t_mean > 0