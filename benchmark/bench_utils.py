"""Triton-free 替代 @triton.testing.perf_report。

使用方式（简洁模式，推荐）：

    from bench_utils import perf_report, Benchmark, do_bench, throughput

    @perf_report(Benchmark(
        x_names=["N"], x_vals=[128, 256, 512], x_log=True,
        plot_name="softmax-performance", ylabel="GB/s",
    ))
    def bench(N):
        x = torch.randn(4096, N, device="cuda")
        ms = do_bench(lambda: torch.softmax(x, axis=-1), quantiles=[0.5,0.2,0.8])
        return {"Triton (GB/s)": throughput(ms, x.numel(), x.element_size()),
                "Torch (GB/s)": throughput(ms, x.numel(), x.element_size())}

    bench.run(print_data=True, show_plots=True)

Triton 兼容模式（line_arg/line_vals/line_names 驱动多行）：

    from triton.testing import perf_report, Benchmark
    → from bench_utils import perf_report, Benchmark

兼容性：
- Benchmark 支持 styles / ylabel / x_log / y_log / args（dict 或 tuple）
- perf_report 返回带 .run() 方法的对象
- 函数允许返回 (mean, min, max) 或 (mean, std) 或 float
- 返回 dict {col: (mean,std)} 时，每列作为独立表头（多列模式）
- throughput(ms, numel, element_size, factor=2) → GB/s
"""

import os
import time
from tabulate import tabulate

_HERE = os.path.dirname(os.path.abspath(__file__))


def throughput(ms, numel, element_size, factor=2):
    """将 ms 耗时转换为 GB/s 吞吐量。

    与 Triton 教程一致：
        gbps = factor * numel * element_size * 1e-9 / (ms * 1e-3)
    默认 factor=2（读+写），纯计算可设 factor=1。
    """
    return factor * numel * element_size * 1e-9 / (ms * 1e-3)


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"):
    """替代 triton.testing.do_bench。返回 mean（或 quantiles 时 (mean, min, max)）ms。

    支持 NPU/CUDA：自动同步对应设备后再计时。
    """
    import torch

    fn()
    if grad_to_none is not None:
        for x in grad_to_none:
            x.grad = None

    dev = None
    for x in grad_to_none or []:
        if hasattr(x, "device") and x.device.type != "cpu":
            dev = x.device
            break

    sync = None
    if dev is None:
        pass
    elif dev.type == "cuda":
        sync = torch.cuda.synchronize
    elif hasattr(torch, "npu") and dev.type == "npu":
        sync = torch.npu.synchronize
    if sync is not None:
        sync()

    start_event = time.perf_counter()
    fn()
    if sync is not None:
        sync()
    estimate_ms = (time.perf_counter() - start_event) * 1000
    n_warmup = max(1, int(warmup / max(estimate_ms, 1e-6)))
    n_repeat = max(1, int(rep / max(estimate_ms, 1e-6)))

    for _ in range(n_warmup):
        fn()
    if sync is not None:
        sync()

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()

    if quantiles is not None:
        n = len(times)
        lo = max(int(n * quantiles[1]) - 1, 0)
        hi = min(int(n * quantiles[2]), n - 1)
        mean = times[n // 2]
        return mean, times[lo], times[hi]
    return times[n // 2] if return_mode == "median" else sum(times) / len(times)


class Benchmark:
    def __init__(
        self,
        x_names,
        x_vals,
        line_arg=None,
        line_vals=None,
        line_names=None,
        plot_name="",
        args=None,
        ylabel="ms",
        x_log=False,
        y_log=False,
        hist=False,
        styles=None,
        y_label=None,
    ):
        self.x_names = [x_names] if isinstance(x_names, str) else list(x_names)
        self.x_vals = list(x_vals)
        self.line_arg = line_arg
        self.line_vals = list(line_vals) if line_vals is not None else None
        self.line_names = list(line_names) if line_names is not None else None
        self.plot_name = plot_name
        self.args = args if args is not None else {}
        self.ylabel = y_label or ylabel
        self.xlabel = " / ".join(self.x_names)
        self.x_log = x_log
        self.y_log = y_log
        self.hist = hist


class _PerfReport:
    def __init__(self, benchmarks, fn):
        self.benchmarks = benchmarks if isinstance(benchmarks, (list, tuple)) else [benchmarks]
        self.fn = fn

    def run(self, print_data=True, show_plots=False, save_path=None, return_df=False):
        for bench in self.benchmarks:
            _run_benchmark(self.fn, bench, print_data, show_plots, save_path)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def perf_report(benchmarks):
    def decorator(fn):
        return _PerfReport(benchmarks, fn)
    return decorator


def _call(fn, bench, x_val, line_val=None):
    kwargs = dict(zip(bench.x_names, x_val)) if len(bench.x_names) > 1 else {bench.x_names[0]: x_val}
    if line_val is not None:
        kwargs[bench.line_arg] = line_val
    if isinstance(bench.args, dict):
        return fn(**kwargs, **bench.args)
    else:
        return fn(*bench.args, **kwargs)


def _parse_measure(value):
    """将 (mean,min,max) / (mean,std) / float 统一为 (mean, std)。"""
    if isinstance(value, (tuple, list)):
        vals = [float(v) for v in value]
        mean = vals[0]
        if len(vals) >= 3:
            std = (vals[2] - vals[1]) / 2
        elif len(vals) >= 2:
            std = vals[1]
        else:
            std = 0.0
        return mean, std
    f = float(value)
    return f, 0.0


def _format_x_val(x_val, bench):
    if len(bench.x_names) == 1:
        return x_val
    return tuple(x_val)


def _make_row(x_val, bench, values):
    x_display = _format_x_val(x_val, bench)
    row = list(x_display) if isinstance(x_display, tuple) else [x_display]
    row += [f"{v:.3f}" for v in values]
    return row


def _run_benchmark(fn, bench, print_data, show_plots, save_path):
    if save_path:
        out_dir = str(save_path)
    else:
        out_dir = os.path.join(_HERE, "output")
    os.makedirs(out_dir, exist_ok=True)
    results_txt = os.path.join(out_dir, f"{bench.plot_name}.txt")

    # 未提供 line 参数（dict 多列模式）时只跑一次，列名来自函数返回值
    lines = [(None, None)] if bench.line_vals is None else zip(bench.line_vals, bench.line_names)

    for lv, ln in lines:
        all_results = [_call(fn, bench, x_val, lv) for x_val in bench.x_vals]

        first = {k: _parse_measure(v) for k, v in all_results[0].items()} if isinstance(all_results[0], dict) else _parse_measure(all_results[0])

        if isinstance(first, dict):
            col_names = list(first.keys())
            headers = list(bench.x_names) + col_names
            rows = []
            col_means = {c: [] for c in col_names}
            col_stds = {c: [] for c in col_names}
            for x_val, res in zip(bench.x_vals, all_results):
                parsed = {k: _parse_measure(v) for k, v in res.items()}
                for c in col_names:
                    col_means[c].append(parsed[c][0])
                    col_stds[c].append(parsed[c][1])
                rows.append(_make_row(x_val, bench, [parsed[c][0] for c in col_names]))
            all_means = [col_means]
            all_stds = [col_stds]
        else:
            headers = list(bench.x_names) + ["mean(ms)", "std(ms)"]
            rows = []
            means, stds = [], []
            for x_val, res in zip(bench.x_vals, all_results):
                m, s = _parse_measure(res)
                means.append(m)
                stds.append(s)
                rows.append(_make_row(x_val, bench, [m, s]))
            all_means = [{"mean": means}]
            all_stds = [{"mean": stds}]

        title = f"[{bench.plot_name}]" + (f" line={ln}" if ln else "")
        if print_data:
            print(f"\n{title}")
            print(tabulate(rows, headers=headers, tablefmt="simple", numalign="right", stralign="right"))

        if print_data:
            with open(results_txt, "a") as f:
                f.write(f"{title}\n")
                f.write(tabulate(rows, headers=headers, tablefmt="simple", numalign="right", stralign="right"))
                f.write("\n\n")

    if show_plots:
        _try_plot(bench.x_vals, all_means, all_stds, bench, out_dir)


def _try_plot(x_vals, all_means, all_stds, bench, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots()
    for i, (means_dict, stds_dict) in enumerate(zip(all_means, all_stds)):
        if bench.line_names:
            ln = bench.line_names[i]
        else:
            ln = None
        for col in means_dict:
            label = f"{ln}-{col}" if ln and len(bench.line_names) > 1 else col
            ax.errorbar(x_vals, means_dict[col], yerr=stds_dict[col], marker="o", label=label)
    ax.set_xlabel(bench.xlabel)
    ax.set_ylabel(bench.ylabel)
    if bench.x_log:
        ax.set_xscale("log")
    if bench.y_log:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    path = os.path.join(out_dir, f"{bench.plot_name or 'bench'}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved → {path}")