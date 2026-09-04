"""BevPool benchmark — 通过 subprocess 调用独立脚本，避免 pytest 环境与 kernel 交互导致偶发崩溃。"""
import os
import subprocess
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))


def test_bev_pool_bench():
    script = os.path.join(_HERE, "run_bench_bev_pool.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        p for p in sys.path if p and os.path.isdir(p)
    ) + ":."
    result = subprocess.run(
        [sys.executable, "-u", script],
        capture_output=True,
        text=True,
        timeout=500,
        env=env,
        cwd=os.path.join(_HERE, ".."),
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-1000:], file=sys.stderr)
    assert result.returncode == 0, f"benchmark failed (rc={result.returncode})"