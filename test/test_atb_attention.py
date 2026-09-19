"""ATB (Ascend Transformer Boost) paged attention v2 — NPU 精度冒烟测试。

覆盖 torch_npu.atb._npu_paged_attention_v2 接口:
    query      (num_tokens, num_heads, head_dim)                  fp16
    key_cache  (num_blocks, block_size, num_kv_heads, head_dim)   fp16
    value_cache 同 key_cache
    block_tables (batch_size, max_blocks_per_seq)                 int32
    context_lens (batch_size,)                                    int32
    out        (num_tokens, num_heads, head_dim)                  fp16
"""

import os

os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")

import pytest
import torch


def make_inputs(device="npu:0", dtype=torch.float16):
    num_q_heads = 32
    num_kv_heads = 2
    head_dim = 128
    num_tokens = 11
    batch_size = 4
    block_size = 128
    num_blocks = 1032
    max_blocks_per_seq = 1

    query = torch.randn(num_tokens, num_q_heads, head_dim, dtype=dtype, device=device)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    value_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    block_tables = torch.randint(
        0,
        num_blocks,
        (batch_size, max_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )
    context_lens = torch.randint(1, 8, (batch_size,), dtype=torch.int32, device=device)
    scaling = 0.08838834764831845
    out = torch.empty(num_tokens, num_q_heads, head_dim, dtype=dtype, device=device)
    return (
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        num_q_heads,
        num_kv_heads,
        scaling,
        out,
    )


@pytest.mark.npu
def test_paged_attention_v2():
    import torch_npu

    (
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        num_q_heads,
        num_kv_heads,
        scaling,
        out,
    ) = make_inputs()

    workspace = torch_npu.atb._npu_paged_attention_v2_get_workspace(
        query,
        key_cache,
        block_tables,
        context_lens,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        num_heads=num_q_heads,
        scale_value=scaling,
        out=out,
    )
    torch_npu.atb._npu_paged_attention_v2(
        query,
        key_cache,
        block_tables,
        context_lens,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        num_heads=num_q_heads,
        scale_value=scaling,
        workspace=workspace,
        out=out,
    )

    assert out.shape == (query.shape[0], num_q_heads, query.shape[2])
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert torch.isfinite(out).all(), "Output contains non-finite values"
