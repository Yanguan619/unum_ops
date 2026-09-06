"""hatchling 构建钩子：依赖管理 + AscendC 扩展自动编译。

Metadata hook — 动态解析依赖（310P 跳过 triton）。
Build hook    — 在 Ascend 310P 上自动编译 bev_pool / voxelization 的 C 扩展 .so。
"""

import os
import subprocess
import sys
import tomllib

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface


# ============================================================
# 设备检测
# ============================================================

def _is_ascend_310p() -> bool:
    try:
        result = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=10,
        )
        text = (result.stdout + result.stderr).upper()
        return "310P" in text or "310P3" in text or "310P4" in text
    except Exception:
        return False


def _should_skip_triton() -> bool:
    if os.environ.get("UNUM_OPS_SKIP_TRITON", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return _is_ascend_310p()


# ============================================================
# 依赖基线
# ============================================================

def _load_base_deps(root: str) -> list[str]:
    with open(os.path.join(root, "pyproject.toml"), "rb") as f:
        cfg = tomllib.load(f)
    table = cfg["tool"]["unum_ops"]["dependencies"]
    return [str(d) for d in table.get("base", [])] + [str(table["triton"])]


# ============================================================
# Metadata hook — 动态依赖
# ============================================================

class UnumOpsMetadataHook(MetadataHookInterface):
    def update(self, metadata: dict) -> None:
        deps = _load_base_deps(self.root)
        if _should_skip_triton():
            deps = [d for d in deps if not d.strip().lower().startswith("triton")]
            print("[unum_ops] detected Ascend 310P: excluded 'triton' from install dependencies")
        metadata["dependencies"] = deps


# ============================================================
# Build hook — 编译 AscendC 扩展 .so
# ============================================================

_EXTENSIONS = [
    {
        "name": "bev_pool",
        "src_dir": "csrc/ascend/bev_pool/op_extension",
        "so_name": "libbev_pool_ops.so",
        "require_headers": ["aclnn_bev_pool.h"],
        "opp_source_dir": "csrc/ascend/bev_pool",
    },
    {
        "name": "voxelization",
        "src_dir": "csrc/ascend/voxelization/op_extension",
        "so_name": "libvoxelization_ops.so",
        "require_headers": ["aclnn_voxelization.h"],
        "opp_source_dir": "csrc/ascend/voxelization",
    },
]


def _system_python() -> str:
    """返回系统 Python 可执行路径（非 pip 隔离环境的 venv python）。"""
    return os.path.join(sys.base_exec_prefix, "bin", "python3")


def _get_torch_cmake_prefix(system_py: str) -> str | None:
    try:
        result = subprocess.run(
            [system_py, "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
            capture_output=True, text=True, timeout=30,
        )
        val = result.stdout.strip()
        if val and os.path.isdir(val):
            return val
    except Exception:
        pass
    return None


def _build_opp_and_install(root: str, ext: dict, ascend_home: str, system_py: str) -> bool:
    """构建并安装 OPP 内核包（生成 ACLNN 头文件），成功返回 True。"""
    name = ext["name"]
    opp_dir = os.path.join(root, ext["opp_source_dir"])
    build_sh = os.path.join(opp_dir, "build.sh")
    build_out = os.path.join(opp_dir, "build_out")

    if not os.path.isfile(build_sh):
        print(f"[unum_ops] {name}: build.sh not found at {build_sh}, cannot build OPP")
        return False

    env = os.environ.copy()
    env["PATH"] = os.path.dirname(system_py) + ":" + env.get("PATH", "")
    env["ASCEND_HOME_PATH"] = ascend_home

    print(f"[unum_ops] {name}: building OPP kernel package ...")
    try:
        result = subprocess.run(
            ["bash", build_sh],
            cwd=opp_dir, check=True, capture_output=True, text=True,
            env=env, timeout=600,
        )
    except subprocess.CalledProcessError as e:
        print(f"[unum_ops] {name} OPP build failed (stderr):")
        for line in (e.stderr or "").splitlines()[-15:]:
            print(f"  | {line}")
        return False
    except subprocess.TimeoutExpired:
        print(f"[unum_ops] {name} OPP build timed out (>600s)")
        return False

    # 安装 OPP 到 CANN vendors（使头文件在 op_api/include 下可见）
    pkg_dir = os.path.join(
        build_out, "_CPack_Packages", "Linux", "External",
        "custom_opp_openEuler_aarch64.run", "packages", "vendors", "customize",
    )
    if not os.path.isdir(pkg_dir):
        # 尝试 .run 文件直接安装
        run_file = os.path.join(build_out, "custom_opp_openEuler_aarch64.run")
        if os.path.isfile(run_file):
            import stat
            os.chmod(run_file, os.stat(run_file).st_mode | stat.S_IEXEC)
            try:
                subprocess.run(
                    ["bash", run_file],
                    cwd=build_out, check=True, capture_output=True, text=True,
                    env=env, timeout=120,
                )
                pkg_dir = os.path.join(build_out, "_CPack_Packages", "Linux", "External",
                                       "custom_opp_openEuler_aarch64.run", "packages", "vendors", "customize")
            except Exception:
                pass

    target_opp = os.path.join(ascend_home, "opp", "vendors", "customize")
    if os.path.isdir(pkg_dir):
        import shutil
        print(f"[unum_ops] {name}: installing OPP into {target_opp} ...")
        shutil.copytree(pkg_dir, target_opp, dirs_exist_ok=True)
        print(f"[unum_ops] {name}: OPP installed successfully")
        return True
    else:
        print(f"[unum_ops] {name}: OPP package built but install directory not found at {pkg_dir}")
        print(f"[unum_ops] {name}: manually install: bash {build_sh}  and  bash {os.path.join(opp_dir, 'rebuild_install.sh')}")
        return False


def _build_ascend_extension(root: str, ext: dict, ascend_home: str) -> bool:
    """编译单个 AscendC 扩展 .so，成功返回 True。"""
    name = ext["name"]
    so_name = ext["so_name"]
    src_dir = os.path.join(root, ext["src_dir"])
    build_dir = os.path.join(src_dir, "build")
    so_path = os.path.join(build_dir, so_name)

    if os.path.isfile(so_path):
        print(f"[unum_ops] {so_name} already exists, skipping build")
        return True

    # 检查 / 构建 OPP 内核包（提供 ACLNN 头文件）
    op_api_inc = os.path.join(ascend_home, "opp", "vendors", "customize", "op_api", "include")
    headers_missing = [h for h in ext.get("require_headers", [])
                       if not os.path.isfile(os.path.join(op_api_inc, h))]
    if headers_missing:
        print(f"[unum_ops] {name}: missing OPP headers {headers_missing}, building OPP kernel ...")
        system_py = _system_python()
        if not _build_opp_and_install(root, ext, ascend_home, system_py):
            print(f"[unum_ops] {name}: OPP build+install failed, cannot build {so_name}")
            return False
        # 再次检查头文件
        still_missing = [h for h in headers_missing
                         if not os.path.isfile(os.path.join(op_api_inc, h))]
        if still_missing:
            print(f"[unum_ops] {name}: OPP headers still missing after install: {still_missing}")
            return False

    system_py = _system_python()
    if not os.path.isfile(system_py):
        print(f"[unum_ops] system python not found at {system_py}, skipping {name} build")
        return False

    cmake_prefix = _get_torch_cmake_prefix(system_py)
    if cmake_prefix is None:
        print(f"[unum_ops] torch not found in system python ({system_py}), skipping {name} build")
        print(f"[unum_ops] run: pip install -e . --no-build-isolation  or  build manually with bash build.sh")
        return False

    os.makedirs(build_dir, exist_ok=True)
    try:
        env = os.environ.copy()
        env["PATH"] = os.path.dirname(system_py) + ":" + env.get("PATH", "")

        print(f"[unum_ops] building {name} op_extension ...")
        subprocess.run(
            ["cmake", src_dir,
             "-DCMAKE_PREFIX_PATH=" + cmake_prefix,
             "-DASCEND_HOME_PATH=" + ascend_home],
            cwd=build_dir, check=True, capture_output=True, text=True, env=env,
        )
        subprocess.run(
            ["cmake", "--build", build_dir, "-j", str(os.cpu_count() or 4)],
            check=True, capture_output=True, text=True, env=env,
        )
        if os.path.isfile(so_path):
            print(f"[unum_ops] {so_name} built successfully")
            return True
        print(f"[unum_ops] {so_name} not found after build, check build logs")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[unum_ops] {name} op_extension build failed (stderr below):")
        if e.stderr:
            for line in e.stderr.strip().splitlines()[-15:]:
                print(f"  | {line}")
        return False


class UnumOpsBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if not _is_ascend_310p():
            return

        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
        print("[unum_ops] detected Ascend 310P, building C extensions ...")
        for ext in _EXTENSIONS:
            _build_ascend_extension(self.root, ext, ascend_home)

    def finalize(self, version: str, build_data: dict, artifact_path: str | None) -> None:
        """构建完成后把 .so 拷入包目录，使 pip install 后 `_find_ops_lib` 能找到。"""
        import shutil
        for ext in _EXTENSIONS:
            name = ext["name"]
            so_name = ext["so_name"]
            src_so = os.path.join(self.root, ext["src_dir"], "build", so_name)
            if not os.path.isfile(src_so):
                continue
            dst_dir = os.path.join(self.root, "src", "unum_ops", name, "_libs")
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src_so, os.path.join(dst_dir, so_name))
            print(f"[unum_ops] copied {so_name} to {dst_dir}")