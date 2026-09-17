# unum_ops

此项目提供 Ascend 310P / CPU / GPU 多硬件适配的算子集合，包含：

- **AscendC 自定义算子**：`bev_pool`（LSS Splat）、`voxelization`（体素化），在 310P 上以 NPU kernel 运行
- **纯 torch/numpy 实现**：`spconv`（稀疏卷积）、`infllm_v2`（注意力）、`sparse_kernel_extension`（稀疏查询）、`pointnet2`（点云算子），跨硬件通用

## Modules

### bev_pool

AscendC 实现的 LSS Lift-Splat 的 Splat 阶段：把稀疏的 frustum 点特征 segment-sum scatter 到稠密 BEV 体素网格。

```python
from unum_ops.bev_pool import bev_pool
out = bev_pool(feats, coords, B=1, D=8, H=100, W=100)
```

### voxelization

AscendC 实现的体素化算子：将点云数据转换为稀疏体素表示。

```python
from unum_ops.voxelization import voxelization
out = voxelization(points)
```

### spconv

纯 torch 实现的稀疏卷积（SubMConv3d/2d, SparseConv3d/2d, SparseInverseConv3d/2d），CPU/GPU/NPU 通用。

### infllm_v2

Inference-optimized attention mechanisms for LLM.

#### Functions

- `infllmv2_attn_stage1_ref_torch`: Reference PyTorch implementation of attention stage 1
- `max_pooling_1d_varlen_ref_torch`: Reference PyTorch implementation of variable-length max pooling

#### Usage

```python
from unum_ops.infllm_v2 import (
    infllmv2_attn_stage1_ref_torch,
    max_pooling_1d_varlen_ref_torch
)

# Attention stage 1
output = infllmv2_attn_stage1_ref_torch(...)

# Max pooling
output = max_pooling_1d_varlen_ref_torch(...)
```

### sparse_kernel_extension

Sparse kernel operations for efficient block table lookups.

#### Functions

- `get_block_table_ref_torch`: PyTorch reference implementation
- `get_block_table_ref_triton`: Triton-optimized implementation

#### Usage

```python
from unum_ops.sparse_kernel_extension import (
    get_block_table_ref_torch,
    get_block_table_ref_triton
)

# Get block table with PyTorch
output = get_block_table_ref_torch(topk_idx, block_table, token_to_bs, seqlen_q)

# Get block table with Triton
output = get_block_table_ref_triton(topk_idx, block_table, token_to_bs, seqlen_q)
```

### pointnet2

点云算子：CUDA 自定义算子 → 纯 PyTorch 重写，提取自 graspnet-npu
（[graspnet-baseline 的 Ascend 310P3 适配工程](https://github.com/Iamnotphage/graspnet-npu)），
CPU/CUDA/NPU 跨硬件通用。

重写背景：原 graspnet-baseline 的 pointnet2/knn 全部是 CUDA C++ extension，
在 Ascend NPU / 纯 CPU 环境无法编译运行，本模块用纯 torch 算子等价重写。

关键 NPU 适配手法：

1. 含 python 循环的算子（FPS / BallQuery / CylinderQuery）在 `device.type == 'npu'`
   时主动 fallback 到 CPU 计算再拷回，规避 NPU 上逐元素循环开销
2. `*_onnx` 变体不包裹 autograd.Function，循环在 ONNX trace 时展开，用于 ATC 导出 OM 静态图
3. ATC 要求 GatherElements 索引为 int32：`three_nn` / `cylinder_query_onnx` 索引统一 `int32`
4. **模块加载时一次替换**：若环境中安装了原生 CUDA extension（`pointnet2_utils`），
   自动替换为 CUDA 实现，否则用纯 PyTorch 回退。调用方无感，零运行时开销。

#### Functions

- `furthest_point_sample(xyz, npoint)`: 最远点采样，返回 `(B, npoint)` 索引
- `gather_operation(features, idx)`: 按索引取值，`(B, C, N) + (B, npoint) -> (B, C, npoint)`
- `three_nn(unknown, known)`: 三最近邻，返回 `(dist, idx)`，idx 为 int32
- `three_interpolate(features, idx, weight)` / `three_interpolate_onnx`: 三近邻加权插值
- `grouping_operation(features, idx)` / `grouping_operation_onnx`: 按索引分组
- `ball_query(radius, nsample, xyz, new_xyz)`: 球形邻域查询
- `cylinder_query(radius, hmin, hmax, nsample, xyz, new_xyz, rot)` / `cylinder_query_onnx`: 圆柱邻域查询（含旋转）
- `knn(ref, query, k)`: K 近邻，`(B, C, M) + (B, C, N) -> (B, k, N)`

#### nn.Module

- `QueryAndGroup(radius, nsample, ...)`: PointNet2 SA 层组件（ball_query + grouping）
- `GroupAll(use_xyz, ...)`: 全量分组
- `CylinderQueryAndGroup(radius, hmin, hmax, nsample, ...)`: 圆柱分组（含旋转）

#### Usage

```python
from unum_ops.pointnet2 import (
    furthest_point_sample, ball_query, grouping_operation, three_nn, knn
)

xyz = torch.randn(1, 20000, 3)
idx = furthest_point_sample(xyz, 1024)           # FPS 采样索引
ball_idx = ball_query(0.05, 64, xyz, xyz)        # 球邻域索引
feats = grouping_operation(xyz.transpose(1, 2), ball_idx)  # 分组特征
```

## Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_spconv.py

# pointnet2 算子测试
pytest test/test_pointnet2.py

# Run stability test
pytest test/test_stability.py -v
```
