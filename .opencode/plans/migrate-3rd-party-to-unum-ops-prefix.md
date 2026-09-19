# 第三方 CUDA 库 → unum_ops 前缀统一

## 目标

所有第三方 CUDA 库（spconv, pointnet2, infllm_v2, sparse_kernel_extension）的 import 前缀换成 `unum_ops.` 后，透明使用本仓库的算子（纯 torch / triton / AscendC）。

## 改动清单

### 1. `spconv/__init__.py` — 补导出 spconv_ascendc

**位置**: `src/unum_ops/spconv/__init__.py`

```python
# 在文件末尾追加
from .spconv_ascendc import available, spconv_gemm  # 或实际导出的符号
```

### 2. `benchmark/bench_spconv.py` — 迁移 4 处 import + 移除 sys.path hack

**当前**（行 17-18, 27-30）:
```python
sys.path.insert(0, os.path.join(_HERE, "..", "src", "unum_ops"))
sys.path.insert(0, _HERE)
...
from spconv import ascendc
import spconv.conv as conv_mod
from spconv.conv import SubMConv3d, SparseConv3d, SparseInverseConv3d
from spconv.sparse_modules import SparseConvTensor, SparseSequential
```

**改为**:
```python
# 删除 sys.path.insert 两行
...
from unum_ops.spconv import spconv_ascendc as ascendc
import unum_ops.spconv.conv as conv_mod
from unum_ops.spconv.conv import SubMConv3d, SparseConv3d, SparseInverseConv3d
from unum_ops.spconv.sparse_modules import SparseConvTensor, SparseSequential
```

### 3. `benchmark/bench_spconv_ascendc.py` — 迁移 3 处 import + 移除 sys.path hack

**当前**（行 19-20, 30-32）:
```python
sys.path.insert(0, os.path.join(_HERE, "..", "src", "unum_ops"))
sys.path.insert(0, _HERE)
...
from spconv import ascendc
from spconv.conv import SubMConv3d, SparseConv3d, SparseInverseConv3d
from spconv.sparse_modules import SparseConvTensor
```

**改为**:
```python
# 删除 sys.path.insert 两行
...
from unum_ops.spconv import spconv_ascendc as ascendc
from unum_ops.spconv.conv import SubMConv3d, SparseConv3d, SparseInverseConv3d
from unum_ops.spconv.sparse_modules import SparseConvTensor
```

### 4. `benchmark/bench_spconv_compare.py` — 迁移 3 处 import + 移除 sys.path hack

**当前**（行 22-27）:
```python
sys.path.insert(0, os.path.join(_HERE, '..', 'src', 'unum_ops'))
import spconv
from spconv.conv import SubMConv3d, SparseConvolution
from spconv.sparse_modules import SparseConvTensor
```

**改为**:
```python
# 删除 sys.path.insert
import unum_ops.spconv as spconv
from unum_ops.spconv.conv import SubMConv3d, SparseConvolution
from unum_ops.spconv.sparse_modules import SparseConvTensor
```

### 5. `test/test_pointnet2.py` — 迁移 5 处 import + 移除 sys.path hack

**位置**:
- `sys.path.insert(0, _PKG_ROOT)` → 删除（行 32-34）
- `from pointnet2 import (...)` → `from unum_ops.pointnet2 import (...)`（行 36-52）
- `import pointnet2 as pn2` → `import unum_ops.pointnet2 as pn2`（行 321, 338, 348, 397）
- 删除 `_PKG_ROOT` 计算（行 32）和 `importlib`（如果不再需要）

```python
# 删掉:
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, "..", "src", "unum_ops"))
sys.path.insert(0, _PKG_ROOT)

# 改为:
from unum_ops.pointnet2 import (
    CylinderQueryAndGroup, GroupAll, QueryAndGroup,
    ball_query, cylinder_query, cylinder_query_onnx,
    furthest_point_sample, furthest_point_sample_onnx,
    gather_operation, grouping_operation, grouping_operation_onnx,
    knn, three_interpolate, three_interpolate_onnx, three_nn,
)
```

### 6. `benchmark/bench_pointnet2.py` — 迁移 1 处 import + 移除 sys.path hack

**当前**（行 21-22, 28-40）:
```python
sys.path.insert(0, os.path.join(_HERE, "..", "src", "unum_ops"))
sys.path.insert(0, _HERE)
...
from pointnet2 import (CylinderQueryAndGroup, ...)
```

**改为**:
```python
# 删除 sys.path.insert 两行
from unum_ops.pointnet2 import (CylinderQueryAndGroup, ...)
```

### 7. `pointnet2/__init__.py` — CUDA 后备后端保持不动

当前 `_load_cuda_backend` 中 `from pointnet2 import _ext` 是可控后备（try/except），
无 sys.path hack —— 走 pip 安装的第三方 CUDA 扩展。保持原样。

## 不迁移的文件

| 文件 | 原因 |
|------|------|
| `test/test_infllmv2_attn_stage1.py` | `from infllm_v2 import infllmv2_attn_stage1` — 第三方 CUDA 基线对比 |
| `test/test_max_pooling_1d.py` | `from infllm_v2.max_pooling_1d import ...` — 第三方 CUDA 基线对比 |
| `test/test_sparse_kernel.py` | `import sparse_kernel_extension` — 第三方 CUDA 基线对比 |
| `test/test_spconv.py` | `import spconv.pytorch as off_pt` — 官版 spconv 基线对比 |
| `benchmark/test_infllmv2_attn_stage1.py` | 同上，CUDA 基线对比 |
| `benchmark/test_max_pooling_1d.py` | 同上 |
| `benchmark/test_sparse_kernel.py` | 同上 |

## API 向下兼容性

| unum_ops 子包 | 对应第三方库 | API 匹配度 |
|---------------|-------------|-----------|
| `unum_ops.spconv` | `spconv` | ✅ 完整匹配（SubMConv3d, SparseConv3d, SparseConvTensor 等） |
| `unum_ops.pointnet2` | `pointnet2` | ✅ 完整匹配（含 onnx 变体） |
| `unum_ops.infllm_v2` | `infllm_v2` | ✅ `infllmv2_attn_stage1` 别名导出 |
| `unum_ops.sparse_kernel_extension` | `sparse_kernel_extension` | ✅ `get_block_table_v2/v3` 别名导出 |

## 验证步骤

1. `python -m pytest test/test_pointnet2.py -x --tb=short`
2. `python -m pytest test/test_spconv.py -x --tb=short`（含官版对比用例）
3. `python -m pytest test/test_sparse_kernel.py -x --tb=short`（含 CUDA 对比用例）
4. `python -m pytest benchmark/bench_spconv.py -x --tb=short`（仅在有 NPU/CUDA 时）