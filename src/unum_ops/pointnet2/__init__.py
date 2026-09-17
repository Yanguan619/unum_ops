# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""PointNet2 / GraspNet 点云算子：CUDA 自定义算子 → 纯 PyTorch 重写。

适配背景（为何要重写）：
  - 原 graspnet 的 pointnet2/knn 全部是 CUDA C++ extension，
    在 Ascend NPU / 纯 CPU 环境无法编译运行。
  - 本包用纯 torch 算子等价重写，可在 CPU/CUDA/NPU 任意设备执行。

模块布局：
  - sampling：最远点采样 FPS（numpy 快路径 + torch 循环 + ONNX 变体）
  - query：半径类邻域查询（ball_query / cylinder_query）
  - nn_search：近邻搜索（knn / three_nn）
  - grouping：按索引取特征（gather）/ 分组特征（grouping）
  - interpolate：三近邻加权插值（three_interpolate）
  - modules：nn.Module 组件（QueryAndGroup / GroupAll / CylinderQueryAndGroup）

关键 NPU 适配手法：
  1. 所有需要 python 循环的算子（FPS / BallQuery / CylinderQuery）在检测到
     ``device.type == 'npu'`` 时主动 fallback 到 CPU 计算再拷回，
     规避 NPU 上逐元素 python 循环的巨大开销。
  2. 提供 ``*_onnx`` 变体（不包裹 autograd.Function），循环在 ONNX trace
     时展开，用于 ATC 导出 OM 静态图。
  3. ATC 要求 GatherElements 索引为 int32：three_nn / cylinder_query_onnx /
     ApproachNet 相关索引输出统一 ``.to(torch.int32)``。

后端选择（模块加载时一次性决定，调用无开销）：
  - 默认使用本包的纯 PyTorch 实现。
  - 若环境中安装了原生 CUDA extension（``pointnet2_utils`` 或
    ``pointnet2._ext``，即原版 graspnet-baseline 编译产物），则自动
    替换为 CUDA 实现。替换时会同时补丁子模块属性与包级导出，
    保证 ``modules.QueryAndGroup`` 等内部消费者也走 CUDA 版本。
"""

from types import ModuleType

import torch

from . import grouping, interpolate, nn_search, query, sampling
from .grouping import (
    GatherOperation,
    GroupingOperation,
    gather_operation,
    grouping_operation,
    grouping_operation_torch,
)
from .interpolate import (
    ThreeInterpolate,
    three_interpolate,
    three_interpolate_torch,
)
from .modules import CylinderQueryAndGroup, GroupAll, QueryAndGroup
from .nn_search import ThreeNN, knn, three_nn
from .query import (
    BallQuery,
    CylinderQuery,
    ball_query,
    cylinder_query,
    cylinder_query_torch,
)
from .sampling import (
    FurthestPointSampling,
    furthest_point_sample,
    furthest_point_sample_torch,
)


# ============================================================
# 可选：替换为原生 CUDA extension（加载时一次性决定）
# ============================================================
def _load_cuda_backend() -> ModuleType | None:
    """尝试加载原生 CUDA extension，返回模块或 None。"""
    if torch.cuda.is_available():
        try:
            import pointnet2_utils  # type: ignore

            return pointnet2_utils
        except ImportError:
            try:
                from pointnet2 import _ext  # type: ignore

                return _ext
            except ImportError:
                return None
    return None


_cuda = _load_cuda_backend()
if _cuda is not None:
    # 同时替换子模块属性与包级导出：
    # - 子模块属性：modules.py 通过模块属性访问算子（延迟绑定），
    #   不替换则 QueryAndGroup 等内部调用仍走纯 torch 回退；
    # - 包级导出：外部 ``from pointnet2 import xxx`` / ``pn2.xxx``。
    sampling.furthest_point_sample = _cuda.furthest_point_sample
    grouping.gather_operation = _cuda.gather_operation
    grouping.grouping_operation = _cuda.grouping_operation
    nn_search.three_nn = _cuda.three_nn
    interpolate.three_interpolate = _cuda.three_interpolate
    query.ball_query = _cuda.ball_query
    query.cylinder_query = _cuda.cylinder_query
    if hasattr(_cuda, "knn"):
        nn_search.knn = _cuda.knn

    furthest_point_sample = _cuda.furthest_point_sample
    gather_operation = _cuda.gather_operation
    three_nn = _cuda.three_nn
    three_interpolate = _cuda.three_interpolate
    grouping_operation = _cuda.grouping_operation
    ball_query = _cuda.ball_query
    cylinder_query = _cuda.cylinder_query
    if hasattr(_cuda, "knn"):
        knn = _cuda.knn
    del _cuda

__version__ = "0.1.0"

__all__ = [
    "CylinderQueryAndGroup",
    "GroupAll",
    # nn.Module 组件
    "QueryAndGroup",
    "ball_query",
    "cylinder_query",
    "cylinder_query_torch",
    # 算子（CUDA extension 可用时自动替换为 CUDA 实现）
    "furthest_point_sample",
    "furthest_point_sample_torch",
    "gather_operation",
    "grouping_operation",
    "grouping_operation_torch",
    "knn",
    "three_interpolate",
    "three_interpolate_torch",
    "three_nn",
]
