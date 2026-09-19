"""Meta-tests：守护 INTERFACES_310P 真相源与 hatch_build._EXTENSIONS 清单的一致性。

- 每个算子子包必须注册到 INTERFACES_310P，反之亦然
- 每个子包 __init__.py 必须定义 __all__，且 __all__ 与注册表双向一致
- csrc/ascend/ 下每个 OPP vendor 必须加入 hatch_build.py 的 _EXTENSIONS 编译清单
"""

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from unum_ops import INTERFACES_310P

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "src" / "unum_ops"
INFRA_SUBPACKAGES = {"utils"}


def _op_subpackages():
    return sorted(
        p.name
        for p in PKG_DIR.iterdir()
        if p.is_dir()
        and (p / "__init__.py").is_file()
        and not p.name.startswith("_")
        and p.name not in INFRA_SUBPACKAGES
    )


def _extensions_manifest():
    tree = ast.parse((REPO_ROOT / "hatch_build.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_EXTENSIONS" for t in node.targets)
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("_EXTENSIONS 未在 hatch_build.py 中找到")


def _import_or_skip(subpackage):
    try:
        return importlib.import_module(f"unum_ops.{subpackage}")
    except Exception as exc:
        pytest.skip(f"unum_ops.{subpackage} 在当前环境不可导入（依赖缺失）: {exc}")


def test_subpackage_registered():
    unregistered = [s for s in _op_subpackages() if s not in INTERFACES_310P]
    assert not unregistered, f"算子子包未注册到 INTERFACES_310P: {unregistered}"


def test_registry_modules_exist():
    stale = [m for m in INTERFACES_310P if m not in _op_subpackages()]
    assert not stale, f"INTERFACES_310P 含不存在或非算子的子包: {stale}"


def test_public_symbols_declared():
    for subpackage in _op_subpackages():
        mod = _import_or_skip(subpackage)
        exported = getattr(mod, "__all__", None)
        assert isinstance(exported, list) and exported, (
            f"unum_ops.{subpackage}.__init__.py 必须定义非空 __all__"
        )


def test_public_symbols_registered():
    for subpackage in _op_subpackages():
        if subpackage not in INTERFACES_310P:
            continue
        mod = _import_or_skip(subpackage)
        missing = [
            s
            for s in mod.__all__
            if s not in INTERFACES_310P[subpackage]
            and not inspect.ismodule(getattr(mod, s, None))
        ]
        assert not missing, f"unum_ops.{subpackage} 导出但未登记到 INTERFACES_310P: {missing}"


def test_registered_symbols_importable():
    for mod_name, table in INTERFACES_310P.items():
        mod = _import_or_skip(mod_name)
        stale = [s for s in table if not hasattr(mod, s)]
        assert not stale, f"INTERFACES_310P[{mod_name!r}] 含不可导入的过期符号: {stale}"


def test_ascend_vendors_in_build_manifest():
    vendor_dirs = {
        p.name for p in (REPO_ROOT / "csrc" / "ascend").iterdir() if p.is_dir()
    }
    built = {Path(e["opp_source_dir"]).name for e in _extensions_manifest()}
    missing = vendor_dirs - built
    assert not missing, f"csrc/ascend/ 下 vendor 未加入 hatch_build._EXTENSIONS: {missing}"


def test_extensions_manifest_paths_exist():
    for ext in _extensions_manifest():
        for key in ("src_dir", "opp_source_dir"):
            path = REPO_ROOT / ext[key]
            assert path.is_dir(), f"_EXTENSIONS[{ext['name']}].{key} 指向不存在的目录: {path}"
        src_dir = REPO_ROOT / ext["src_dir"]
        assert (src_dir / "CMakeLists.txt").is_file(), (
            f"_EXTENSIONS[{ext['name']}] 的 op_extension 缺少 CMakeLists.txt: {src_dir}"
        )
