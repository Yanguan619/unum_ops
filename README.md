# unum_ops

此项目提供 Ascend 310P / CPU / GPU 多硬件适配的算子集合，包含：

- **AscendC 自定义算子**：`bev_pool`（LSS Splat）、`voxelization`（体素化），在 310P 上以 NPU kernel 运行
- **纯 torch/numpy 实现**：`spconv`（稀疏卷积）、`infllm_v2`（注意力）、`sparse_kernel_extension`（稀疏查询），跨硬件通用

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

## Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_spconv.py

# Run stability test
pytest test/test_stability.py -v
```
