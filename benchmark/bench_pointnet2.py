"""pointnet2 算子性能基准（Triton-free + tabulate 输出）。

覆盖 src/unum_ops/pointnet2/ 中的算子在不同规模下的耗时，
规模参考 graspnet-baseline 实际输入（N=20000, npoint=1024, nsample=32）。

对比项（作为表头列）：
  - cylinder_query（python 循环版）vs cylinder_query_torch（全向量化版）
  - grouping_operation vs grouping_operation_torch
  - three_interpolate vs three_interpolate_torch

运行方式:
    python benchmark/bench_pointnet2.py
    python -m pytest benchmark/bench_pointnet2.py -v

结果写入 benchmark/output/<plot_name>.txt
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "unum_ops"))
sys.path.insert(0, _HERE)

import pytest
import torch

from bench_utils import Benchmark, do_bench, perf_report
from pointnet2 import (
    CylinderQueryAndGroup,
    QueryAndGroup,
    ball_query,
    cylinder_query,
    cylinder_query_torch,
    furthest_point_sample,
    gather_operation,
    grouping_operation,
    grouping_operation_torch,
    knn,
    three_interpolate,
    three_interpolate_torch,
    three_nn,
)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    return torch.device("cpu")


DEVICE = _device()

_QUANTILES = [0.5, 0.2, 0.8]
# do_bench 的 warmup/rep 为时间预算（ms），按算子耗时量级分档
_BUDGET_FAST = dict(warmup=50, rep=500)      # 索引/插值类（<10ms）
_BUDGET_MED = dict(warmup=100, rep=1000)     # 查询类（10~100ms）
_BUDGET_SLOW = dict(warmup=100, rep=2000)    # FPS / 大规模查询（>100ms）

_FPS_CONFIGS = [
    ("N=2048,npoint=256", 2048, 256),
    ("N=20000,npoint=1024", 20000, 1024),  # graspnet 实际规模
]
_BALL_CONFIGS = [
    ("N=2048,npoint=256,nsample=16", 2048, 256, 16),
    ("N=20000,npoint=1024,nsample=32", 20000, 1024, 32),  # graspnet 实际规模
]
_CYL_torch_CONFIGS = [
    ("N=2048,npoint=256,nsample=32", 2048, 256, 32),
    ("N=20000,npoint=1024,nsample=32", 20000, 1024, 32),
]
_GATHER_CONFIGS = [
    ("C=64,N=2048,npoint=256", 64, 2048, 256),
    ("C=128,N=20000,npoint=1024", 128, 20000, 1024),
]
_GROUPING_CONFIGS = [
    ("C=64,N=2048,npoint=256,nsample=16", 64, 2048, 256, 16),
    ("C=128,N=20000,npoint=1024,nsample=32", 128, 20000, 1024, 32),
]
_THREE_NN_CONFIGS = [
    ("n=256,m=2048", 256, 2048),
    ("n=1024,m=20000", 1024, 20000),  # graspnet FP 层实际规模
]
_KNN_CONFIGS = [
    ("M=2048,N=256,k=3", 2048, 256, 3),
    ("M=20000,N=1024,k=3", 20000, 1024, 3),
]


def _pick(configs, config):
    return next(c[1:] for c in configs if c[0] == config)


# ============================================================
# furthest_point_sample：python 迭代循环，CPU 上是热点
# ============================================================
@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _FPS_CONFIGS],
        plot_name="pointnet2_fps",
        ylabel="Latency (ms)",
    ),
)
def bench_fps(config):
    N, npoint = _pick(_FPS_CONFIGS, config)
    xyz = torch.randn(1, N, 3, device=DEVICE)
    out = furthest_point_sample(xyz, npoint)
    ms = do_bench(lambda: furthest_point_sample(xyz, npoint),
                  grad_to_none=[out], quantiles=_QUANTILES, **_BUDGET_SLOW)
    return {"fps(ms)": ms}


# ============================================================
# ball_query：cdist + 逐中心 python 循环
# ============================================================
@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _BALL_CONFIGS],
        plot_name="pointnet2_ball_query",
        ylabel="Latency (ms)",
    ),
)
def bench_ball_query(config):
    N, npoint, nsample = _pick(_BALL_CONFIGS, config)
    xyz = torch.randn(1, N, 3, device=DEVICE)
    new_xyz = torch.randn(1, npoint, 3, device=DEVICE)
    out = ball_query(0.4, nsample, xyz, new_xyz)
    ms = do_bench(lambda: ball_query(0.4, nsample, xyz, new_xyz),
                  grad_to_none=[out], quantiles=_QUANTILES, **_BUDGET_MED)
    return {"ball_query(ms)": ms}


# ============================================================
# cylinder_query_torch：全向量化版（可跑大规模）
# ============================================================
@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _CYL_torch_CONFIGS],
        plot_name="pointnet2_cylinder_query_torch",
        ylabel="Latency (ms)",
    ),
)
def bench_cylinder_query_torch(config):
    N, npoint, nsample = _pick(_CYL_torch_CONFIGS, config)
    xyz = torch.randn(1, N, 3, device=DEVICE)
    new_xyz = torch.randn(1, npoint, 3, device=DEVICE)
    rot = torch.eye(3, device=DEVICE).expand(1, npoint, 3, 3).contiguous()
    out = cylinder_query_torch(0.5, -0.3, 0.3, nsample, xyz, new_xyz, rot)
    ms = do_bench(lambda: cylinder_query_torch(0.5, -0.3, 0.3, nsample, xyz, new_xyz, rot),
                  grad_to_none=[out], quantiles=_QUANTILES, **_BUDGET_SLOW)
    return {"torch(ms)": ms}


# ============================================================
# cylinder_query：循环版 vs 向量化版（循环版内存 O(npoint*N)，规模受限）
# ============================================================
_CYL_CMP_CONFIG = ("N=2048,npoint=256,nsample=32", 2048, 256, 32)


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[_CYL_CMP_CONFIG[0]],
        plot_name="pointnet2_cylinder_query_compare",
        ylabel="Latency (ms)",
    ),
)
def bench_cylinder_query_compare(config):
    N, npoint, nsample = _CYL_CMP_CONFIG[1:]
    xyz = torch.randn(1, N, 3, device=DEVICE)
    new_xyz = torch.randn(1, npoint, 3, device=DEVICE)
    rot = torch.eye(3, device=DEVICE).expand(1, npoint, 3, 3).contiguous()
    rot_flat = rot.view(1, npoint, 9)
    out_loop = cylinder_query(0.5, -0.3, 0.3, nsample, xyz, new_xyz, rot_flat)
    out_torch = cylinder_query_torch(0.5, -0.3, 0.3, nsample, xyz, new_xyz, rot)
    loop_ms = do_bench(lambda: cylinder_query(0.5, -0.3, 0.3, nsample, xyz, new_xyz, rot_flat),
                       grad_to_none=[out_loop], quantiles=_QUANTILES, **_BUDGET_MED)
    torch_ms = do_bench(lambda: cylinder_query_torch(0.5, -0.3, 0.3, nsample, xyz, new_xyz, rot),
                       grad_to_none=[out_torch], quantiles=_QUANTILES, **_BUDGET_MED)
    return {"loop(ms)": loop_ms, "torch(ms)": torch_ms}


# ============================================================
# gather / grouping：索引类算子
# ============================================================
@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _GATHER_CONFIGS],
        plot_name="pointnet2_gather",
        ylabel="Latency (ms)",
    ),
)
def bench_gather(config):
    C, N, npoint = _pick(_GATHER_CONFIGS, config)
    features = torch.randn(1, C, N, device=DEVICE)
    idx = torch.randint(0, N, (1, npoint), device=DEVICE)
    out = gather_operation(features, idx)
    ms = do_bench(lambda: gather_operation(features, idx),
                  grad_to_none=[out], quantiles=_QUANTILES, **_BUDGET_FAST)
    return {"gather(ms)": ms}


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _GROUPING_CONFIGS],
        plot_name="pointnet2_grouping",
        ylabel="Latency (ms)",
    ),
)
def bench_grouping(config):
    C, N, npoint, nsample = _pick(_GROUPING_CONFIGS, config)
    features = torch.randn(1, C, N, device=DEVICE)
    idx = torch.randint(0, N, (1, npoint, nsample), device=DEVICE)
    out = grouping_operation(features, idx)
    ms = do_bench(lambda: grouping_operation(features, idx),
                  grad_to_none=[out], quantiles=_QUANTILES, **_BUDGET_MED)
    return {"grouping(ms)": ms}


_GROUPING_CMP_CONFIG = ("C=128,N=20000,npoint=1024,nsample=32", 128, 20000, 1024, 32)


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[_GROUPING_CMP_CONFIG[0]],
        plot_name="pointnet2_grouping_compare",
        ylabel="Latency (ms)",
    ),
)
def bench_grouping_compare(config):
    C, N, npoint, nsample = _GROUPING_CMP_CONFIG[1:]
    features = torch.randn(1, C, N, device=DEVICE)
    idx = torch.randint(0, N, (1, npoint, nsample), device=DEVICE)
    out_loop = grouping_operation(features, idx)
    out_torch = grouping_operation_torch(features, idx)
    loop_ms = do_bench(lambda: grouping_operation(features, idx),
                       grad_to_none=[out_loop], quantiles=_QUANTILES, **_BUDGET_MED)
    torch_ms = do_bench(lambda: grouping_operation_torch(features, idx),
                       grad_to_none=[out_torch], quantiles=_QUANTILES, **_BUDGET_MED)
    return {"loop(ms)": loop_ms, "torch(ms)": torch_ms}


# ============================================================
# three_nn / three_interpolate
# ============================================================
@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _THREE_NN_CONFIGS],
        plot_name="pointnet2_three_nn",
        ylabel="Latency (ms)",
    ),
)
def bench_three_nn(config):
    n, m = _pick(_THREE_NN_CONFIGS, config)
    unknown = torch.randn(1, n, 3, device=DEVICE)
    known = torch.randn(1, m, 3, device=DEVICE)
    dist, idx = three_nn(unknown, known)
    ms = do_bench(lambda: three_nn(unknown, known),
                  grad_to_none=[dist, idx], quantiles=_QUANTILES, **_BUDGET_MED)
    return {"three_nn(ms)": ms}


_TI_CMP_CONFIG = ("c=128,m=20000,n=1024", 128, 20000, 1024)


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[_TI_CMP_CONFIG[0]],
        plot_name="pointnet2_three_interpolate_compare",
        ylabel="Latency (ms)",
    ),
)
def bench_three_interpolate_compare(config):
    c, m, n = _TI_CMP_CONFIG[1:]
    features = torch.randn(1, c, m, device=DEVICE)
    unknown = torch.randn(1, n, 3, device=DEVICE)
    known = torch.randn(1, m, 3, device=DEVICE)
    dist, idx = three_nn(unknown, known)
    weight = torch.rand(1, n, 3, device=DEVICE)
    out_loop = three_interpolate(features, idx, weight)
    out_torch = three_interpolate_torch(features, idx, weight)
    loop_ms = do_bench(lambda: three_interpolate(features, idx, weight),
                       grad_to_none=[out_loop], quantiles=_QUANTILES, **_BUDGET_FAST)
    torch_ms = do_bench(lambda: three_interpolate_torch(features, idx, weight),
                       grad_to_none=[out_torch], quantiles=_QUANTILES, **_BUDGET_FAST)
    return {"loop(ms)": loop_ms, "torch(ms)": torch_ms}


# ============================================================
# knn
# ============================================================
@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[c[0] for c in _KNN_CONFIGS],
        plot_name="pointnet2_knn",
        ylabel="Latency (ms)",
    ),
)
def bench_knn(config):
    M, N, k = _pick(_KNN_CONFIGS, config)
    ref = torch.randn(1, 3, M, device=DEVICE)
    query = torch.randn(1, 3, N, device=DEVICE)
    out = knn(ref, query, k)
    ms = do_bench(lambda: knn(ref, query, k),
                  grad_to_none=[out], quantiles=_QUANTILES, **_BUDGET_MED)
    return {"knn(ms)": ms}


# ============================================================
# nn.Module 封装：端到端 grouping 路径
# ============================================================
_QAG_CONFIG = ("N=20000,npoint=1024,nsample=32,C=128", 20000, 1024, 32, 128)


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[_QAG_CONFIG[0]],
        plot_name="pointnet2_query_and_group",
        ylabel="Latency (ms)",
    ),
)
def bench_query_and_group(config):
    N, npoint, nsample, C = _QAG_CONFIG[1:]
    xyz = torch.randn(1, N, 3, device=DEVICE)
    new_xyz = torch.randn(1, npoint, 3, device=DEVICE)
    features = torch.randn(1, C, N, device=DEVICE)
    qg = QueryAndGroup(radius=0.4, nsample=nsample)
    out = qg(xyz, new_xyz, features)
    ms = do_bench(lambda: qg(xyz, new_xyz, features),
                  grad_to_none=[out], quantiles=_QUANTILES, **_BUDGET_SLOW)
    return {"query_and_group(ms)": ms}


_CQAG_CONFIG = ("N=4096,npoint=256,nsample=32,C=64", 4096, 256, 32, 64)


@perf_report(
    Benchmark(
        x_names=["config"],
        x_vals=[_CQAG_CONFIG[0]],
        plot_name="pointnet2_cylinder_query_and_group",
        ylabel="Latency (ms)",
    ),
)
def bench_cylinder_query_and_group(config):
    N, npoint, nsample, C = _CQAG_CONFIG[1:]
    xyz = torch.randn(1, N, 3, device=DEVICE)
    new_xyz = torch.randn(1, npoint, 3, device=DEVICE)
    rot = torch.eye(3, device=DEVICE).expand(1, npoint, 3, 3).contiguous()
    features = torch.randn(1, C, N, device=DEVICE)
    cqg = CylinderQueryAndGroup(radius=0.5, hmin=-0.3, hmax=0.3, nsample=nsample)
    out = cqg(xyz, new_xyz, rot, features)
    ms = do_bench(lambda: cqg(xyz, new_xyz, rot, features),
                  grad_to_none=[out], quantiles=_QUANTILES, **_BUDGET_MED)
    return {"cylinder_query_and_group(ms)": ms}


# ============================================================
# 入口
# ============================================================
_BENCHES = [
    bench_fps,
    bench_ball_query,
    bench_cylinder_query_torch,
    bench_cylinder_query_compare,
    bench_gather,
    bench_grouping,
    bench_grouping_compare,
    bench_three_nn,
    bench_three_interpolate_compare,
    bench_knn,
    bench_query_and_group,
    bench_cylinder_query_and_group,
]


def _run_all():
    for bench in _BENCHES:
        bench.run(print_data=True, show_plots=True)


def test_fps_bench():
    bench_fps.run(print_data=True, show_plots=True)


def test_ball_query_bench():
    bench_ball_query.run(print_data=True, show_plots=True)


def test_cylinder_query_torch_bench():
    bench_cylinder_query_torch.run(print_data=True, show_plots=True)


def test_cylinder_query_loop_vs_torch_bench():
    bench_cylinder_query_compare.run(print_data=True, show_plots=True)


def test_gather_bench():
    bench_gather.run(print_data=True, show_plots=True)


def test_grouping_bench():
    bench_grouping.run(print_data=True, show_plots=True)


def test_grouping_loop_vs_torch_bench():
    bench_grouping_compare.run(print_data=True, show_plots=True)


def test_three_nn_bench():
    bench_three_nn.run(print_data=True, show_plots=True)


def test_three_interpolate_loop_vs_torch_bench():
    bench_three_interpolate_compare.run(print_data=True, show_plots=True)


def test_knn_bench():
    bench_knn.run(print_data=True, show_plots=True)


def test_query_and_group_bench():
    bench_query_and_group.run(print_data=True, show_plots=True)


def test_cylinder_query_and_group_bench():
    bench_cylinder_query_and_group.run(print_data=True, show_plots=True)


if __name__ == "__main__":
    _run_all()
