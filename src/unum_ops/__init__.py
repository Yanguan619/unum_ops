"""unum_ops 顶层包。

各子模块采用延迟导入：依赖缺失（如 triton 不可用）时静默跳过，
不影响包整体导入。运行时自动检测设备（CPU/CUDA/NPU 310P/910），
接口可用性清单 ``INTERFACES_310P`` 配合 ``detect_device()`` 联动。
"""

import os
from typing import Dict, List, Tuple

# ============================================================
# 设备自动检测
# ============================================================
_DEVICE_CACHE: Dict[str, str] = {}


def detect_device() -> str:
    """检测当前运行时设备类型。

    返回值:
        "cpu"       — 无加速器，纯 CPU
        "cuda"      — NVIDIA GPU（CUDA）
        "npu_310p"  — Ascend 310P（NPU）
        "npu_910"   — Ascend 910（NPU）
        "npu_other" — 其他 Ascend NPU 型号
    """
    cached = _DEVICE_CACHE.get("type")
    if cached:
        return cached

    try:
        import torch
    except ImportError:
        _DEVICE_CACHE["type"] = "cpu"
        return "cpu"

    if torch.cuda.is_available():
        _DEVICE_CACHE["type"] = "cuda"
        return "cuda"

    if hasattr(torch, "npu") and torch.npu.is_available():
        try:
            name = torch.npu.get_device_name(0)
            if "310P" in name or "Ascend310P" in name:
                _DEVICE_CACHE["type"] = "npu_310p"
            elif "910" in name or "Ascend910" in name:
                _DEVICE_CACHE["type"] = "npu_910"
            else:
                _DEVICE_CACHE["type"] = "npu_other"
            return _DEVICE_CACHE["type"]
        except Exception:
            _DEVICE_CACHE["type"] = "npu_other"
            return "npu_other"

    _DEVICE_CACHE["type"] = "cpu"
    return "cpu"


IS_310P = detect_device() == "npu_310p"
IS_NPU = detect_device().startswith("npu")
IS_CUDA = detect_device() == "cuda"
IS_CPU = detect_device() == "cpu"


# ============================================================
# 延迟导入子模块
# ============================================================
for _mod in ("infllm_v2", "sparse_kernel_extension", "voxelization", "spconv"):
    try:
        __import__(f"unum_ops.{_mod}")
    except Exception:
        pass

# ============================================================
# 静态接口可用性清单（全局真相源）
# ============================================================
# 结构: { 子模块: { 接口名: (status, reason) } }
#   status: True  = 310P 可直接调用（纯 torch/numpy/已编译的 AscendC 算子）
#           False = 310P 不可调用（依赖 triton / 未编译的 .so 等）
#   reason: 简要说明不可用原因或可用条件
#
# 维护约定：新增接口时在此登记一行；修改依赖时同步更新 status/reason。
# ============================================================
INTERFACES_310P: Dict[str, Dict[str, Tuple[bool, str]]] = {
    "spconv": {
        "SparseConvTensor": (True, "纯 torch"),
        "SparseModule": (True, "纯 torch"),
        "SparseSequential": (True, "纯 torch"),
        "SparseReLU": (True, "纯 torch"),
        "SparseBatchNorm1d": (True, "纯 torch"),
        "SparseLinear": (True, "纯 torch"),
        "SparseConv3dAdapter": (True, "纯 torch"),
        "SubMConv3dAdapter": (True, "纯 torch"),
        "SubMConv3d": (True, "纯 torch"),
        "SubMConv2d": (True, "纯 torch"),
        "SparseConv3d": (True, "纯 torch"),
        "SparseConv2d": (True, "纯 torch"),
        "SparseInverseConv3d": (True, "纯 torch"),
        "SparseInverseConv2d": (True, "纯 torch"),
        "VoxelGeneratorV2": (True, "纯 numpy"),
        "VoxelGenerator": (True, "纯 numpy"),
    },
    "infllm_v2": {
        "infllmv2_attn_stage1_ref_torch": (True, "纯 torch 参考实现"),
        "infllmv2_attn_stage1_triton": (False, "triton kernel，需 CUDA"),
        "infllmv2_attn_stage1_triton_v2": (False, "triton kernel，需 CUDA"),
        "max_pooling_1d_varlen_ref_triton": (False, "triton kernel，需 CUDA"),
    },
    "sparse_kernel_extension": {
        "get_block_table_ref_torch": (True, "纯 torch 参考实现"),
        "get_block_table_ref_triton": (False, "triton kernel，需 CUDA"),
        "get_block_table_ref_triton_v2": (False, "triton kernel，需 CUDA"),
        "get_block_table_ref_triton_v3": (False, "triton kernel，需 CUDA"),
    },
    "voxelization": {
        "voxelization": (True, "AscendC 自定义算子，需先编译 .so"),
        "VoxelizationOutput": (True, "纯 python 数据结构"),
    },
    "bev_pool": {
        "bev_pool": (True, "AscendC 自定义算子，需先编译 .so"),
        "bev_pool_torch": (True, "纯 torch scatter_add 实现"),
        "BevPoolOutput": (True, "纯 python 数据结构"),
    },
}


def _runtimes_ok(entry: Tuple[bool, str]) -> bool:
    """根据当前运行时判断某接口是否可用。"""
    ok, reason = entry
    if not ok:
        return False
    # "需编译 .so" 的接口仅当 .so 存在时视为可用
    if "需先编译" in reason:
        return False
    return True


def list_available_interfaces() -> List[str]:
    """列出当前设备上可调用的全部接口。"""
    return [
        f"{mod}.{name}"
        for mod, tbl in INTERFACES_310P.items()
        for name, (ok, reason) in tbl.items()
        if _runtimes_ok((ok, reason))
    ]


def list_unavailable_interfaces() -> List[str]:
    """列出当前设备上不可调用的全部接口。"""
    return [
        f"{mod}.{name}"
        for mod, tbl in INTERFACES_310P.items()
        for name, (ok, reason) in tbl.items()
        if not _runtimes_ok((ok, reason))
    ]


def is_available(module: str, name: str) -> bool:
    """查询某个接口在当前设备上是否可用。"""
    entry = INTERFACES_310P.get(module, {}).get(name)
    if entry is None:
        return False
    return _runtimes_ok(entry)


def list_310p_interfaces() -> List[str]:
    """列出 310P 理论上可调用的全部接口名（忽略 .so 编译状态）。"""
    return [
        f"{mod}.{name}"
        for mod, tbl in INTERFACES_310P.items()
        for name, (ok, _) in tbl.items()
        if ok
    ]


def list_non_310p_interfaces() -> List[str]:
    """列出 310P 不可调用的全部接口名。"""
    return [
        f"{mod}.{name}"
        for mod, tbl in INTERFACES_310P.items()
        for name, (ok, _) in tbl.items()
        if not ok
    ]


def is_310p_compatible(module: str, name: str) -> bool:
    """查询某个接口是否可在 310P 上调用。"""
    return INTERFACES_310P.get(module, {}).get(name, (False, ""))[0]


def print_310p_interfaces() -> None:
    """打印 310P 接口可用性清单（与当前设备联动）。"""
    device = detect_device()
    print(f"当前设备: {device}")
    print(f"{'接口':<42} {'可用':<6} {'说明'}")
    print("-" * 70)
    for mod, tbl in INTERFACES_310P.items():
        for name, (ok, reason) in tbl.items():
            available = _runtimes_ok((ok, reason))
            tag = "可用" if available else "不可用"
            print(f"{mod+'.'+name:<42} {tag:<6} {reason}")


def main() -> None:
    """控制台入口：`unum_ops` 打印设备与接口可用性。"""
    print_310p_interfaces()


__all__ = [
    "infllm_v2",
    "sparse_kernel_extension",
    "voxelization",
    "spconv",
    "INTERFACES_310P",
    "IS_310P",
    "IS_NPU",
    "IS_CUDA",
    "IS_CPU",
    "detect_device",
    "list_available_interfaces",
    "list_unavailable_interfaces",
    "is_available",
    "list_310p_interfaces",
    "list_non_310p_interfaces",
    "is_310p_compatible",
    "print_310p_interfaces",
]