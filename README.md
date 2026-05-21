# unum_ops

## Modules

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
pytest test/test_sparse_kernel.py

# Run specific test
pytest test/test_sparse_kernel.py::test_get_table_triton -v
```
