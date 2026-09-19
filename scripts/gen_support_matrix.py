"""从 unum_ops.INTERFACES_310P 生成 docs/support_matrix.md（310P 接口支持矩阵）。

用法:
    python scripts/gen_support_matrix.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from unum_ops import INTERFACES_310P

OUT_PATH = REPO_ROOT / "docs" / "support_matrix.md"


def main():
    lines = [
        "# 310P 接口支持矩阵",
        "",
        "> 由 `scripts/gen_support_matrix.py` 从 `unum_ops.INTERFACES_310P` 自动生成，请勿手改。",
        "",
        "| 子包 | 接口 | 310P 可用 | 说明 |",
        "|------|------|-----------|------|",
    ]
    n_ok = n_total = 0
    for module in sorted(INTERFACES_310P):
        for symbol, (available, reason) in INTERFACES_310P[module].items():
            n_total += 1
            n_ok += available
            mark = "✅" if available else "❌"
            lines.append(f"| `{module}` | `{symbol}` | {mark} | {reason} |")
    lines += [
        "",
        f"共 {n_total} 个接口，310P 直接可用 {n_ok} 个。",
        "",
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"support matrix written to {OUT_PATH} ({n_ok}/{n_total} available)")


if __name__ == "__main__":
    main()
