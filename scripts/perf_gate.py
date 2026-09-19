"""benchmark 性能回归门禁。

用法:
    python scripts/perf_gate.py --snapshot          # 固化当前 benchmark/output/*.txt 最新结果为基线
    python scripts/perf_gate.py [--tolerance 0.2]   # 与基线对比，任一 *_ms 指标回归超阈值则退出码 1

基线文件: benchmark/reports/perf_baseline.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "benchmark" / "output"
BASELINE_PATH = REPO_ROOT / "benchmark" / "reports" / "perf_baseline.json"


def _is_separator(line):
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {"-"}


def _parse_output_files():
    sections = {}
    for txt in sorted(OUTPUT_DIR.glob("*.txt")):
        section = txt.stem
        headers = None
        sections.setdefault(section, {})
        for line in txt.read_text().splitlines():
            line = line.rstrip("\r")
            if line.startswith("["):
                section = f"{txt.stem}:{line.strip()}"
                headers = None
                sections.setdefault(section, {})
                continue
            if not line.strip():
                headers = None
                continue
            if _is_separator(line):
                continue
            cells = line.split()
            if any("ms" in c.lower() for c in cells):
                headers = cells
                sections.setdefault(section, {})
                continue
            if headers is None or len(cells) != len(headers):
                continue
            row = dict(zip(headers, cells))
            sections[section][row[headers[0]]] = row
    return sections


def _metric_columns(headers):
    return [h for h in headers if "ms" in h.lower()]


def _to_sections_with_metrics(sections):
    out = {}
    for title, rows in sections.items():
        for row_key, row in rows.items():
            for col, val in row.items():
                if "ms" not in col.lower():
                    continue
                try:
                    out.setdefault(title, {}).setdefault(row_key, {})[col] = float(val)
                except ValueError:
                    continue
    return out


def snapshot():
    metrics = _to_sections_with_metrics(_parse_output_files())
    if not metrics:
        print("perf_gate: benchmark/output/ 下无可解析的 *_ms 指标，未生成基线")
        return 1
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "sections": metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    n = sum(len(rows) for rows in metrics.values())
    print(f"perf_gate: 基线已写入 {BASELINE_PATH} ({n} 行指标)")
    return 0


def gate(tolerance):
    if not BASELINE_PATH.exists():
        print(f"perf_gate: 基线不存在 ({BASELINE_PATH})，先运行 --snapshot；门禁跳过")
        return 0
    baseline = json.loads(BASELINE_PATH.read_text())["sections"]
    current = _to_sections_with_metrics(_parse_output_files())
    regressions = []
    for title, rows in baseline.items():
        for row_key, cols in rows.items():
            cur_row = current.get(title, {}).get(row_key)
            if cur_row is None:
                continue
            for col, base_val in cols.items():
                cur_val = cur_row.get(col)
                if cur_val is None:
                    continue
                if cur_val > base_val * (1 + tolerance):
                    regressions.append(
                        f"{title} [{row_key}] {col}: {base_val:.3f} -> {cur_val:.3f} ms "
                        f"(+{(cur_val / base_val - 1) * 100:.1f}% > +{tolerance * 100:.0f}%)"
                    )
    if regressions:
        print("perf_gate: 检测到性能回归:")
        for r in regressions:
            print(f"  - {r}")
        return 1
    print("perf_gate: 无性能回归")
    return 0


def main():
    parser = argparse.ArgumentParser(description="benchmark 性能回归门禁")
    parser.add_argument("--snapshot", action="store_true", help="固化当前结果为基线")
    parser.add_argument("--tolerance", type=float, default=0.2, help="回归阈值 (默认 0.2 = +20%%)")
    args = parser.parse_args()
    if args.snapshot:
        sys.exit(snapshot())
    sys.exit(gate(args.tolerance))


if __name__ == "__main__":
    main()
