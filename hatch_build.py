"""hatchling 构建钩子：安装时自动识别 Ascend 310P，剔除 triton 依赖。

在 310P（无 CUDA，triton 不可用）上执行 `pip install -e .` 时，
triton 会被自动从 install_requires 中移除；其余环境（CUDA/CPU）正常安装。

依赖基线从 pyproject.toml 的 `[tool.unum_ops.dependencies]` 读取，
`dynamic = ["dependencies"]` 使本 hook 在 metadata 阶段被调用。

检测优先级：
  1. 环境变量 UNUM_OPS_SKIP_TRITON=1 → 强制剔除（构建机与运行机分离时用）
  2. npu-smi info 输出包含 310P → 判定为 310P
  3. 其他情况 → 保留 triton（保守默认，避免 CUDA/CPU 误剔除）
"""

import os
import subprocess
import tomllib

from hatchling.metadata.plugin.interface import MetadataHookInterface


def _is_ascend_310p() -> bool:
    """通过 npu-smi 判断当前是否为 Ascend 310P 环境。"""
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = (result.stdout + result.stderr).upper()
        return "310P" in text or "310P3" in text or "310P4" in text
    except Exception:
        return False


def _should_skip_triton() -> bool:
    if os.environ.get("UNUM_OPS_SKIP_TRITON", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return _is_ascend_310p()


def _load_base_deps(root: str) -> list[str]:
    """从 pyproject.toml 的 [tool.unum_ops.dependencies] 读取依赖基线。"""
    with open(os.path.join(root, "pyproject.toml"), "rb") as f:
        cfg = tomllib.load(f)
    table = cfg["tool"]["unum_ops"]["dependencies"]
    return [str(d) for d in table.get("base", [])] + [str(table["triton"])]


class UnumOpsMetadataHook(MetadataHookInterface):
    def update(self, metadata: dict) -> None:
        deps = _load_base_deps(self.root)
        if _should_skip_triton():
            deps = [d for d in deps if not d.strip().lower().startswith("triton")]
            print("[unum_ops] detected Ascend 310P: excluded 'triton' from install dependencies")
        metadata["dependencies"] = deps
