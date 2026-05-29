"""
q_reshaped.shape=torch.Size([11, 32, 128])
self.page_size=128, layer.tp_k_head_num=2, layer.head_dim=128
forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).shape=torch.Size([1032, 128, 2, 128])
forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id).shape=torch.Size([1032, 128, 2, 128])
layer.tp_q_head_num=32
layer.scaling=0.08838834764831845
self.forward_metadata.block_tables.shape=torch.Size([1, 1])
forward_batch.seq_lens.shape=torch.Size([1])
attn_output.shape=torch.Size([11, 32, 128])
"""

import os

os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")

import pytest
import torch


def data(device="npu:0", dtype=torch.float16):
    num_q_heads = 32
    num_kv_heads = 2  # 4
    head_dim = 128
    num_tokens = 11  # 4
    batch_size = 4
    block_size = 128
    num_blocks = 1032  # 100
    max_blocks_per_seq = 1  # 8

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
    print(f"query: {query.shape} {query.dtype} {query.device}")
    print(f"key_cache: {key_cache.shape} {key_cache.dtype} {key_cache.device}")
    print(f"value_cache: {value_cache.shape} {value_cache.dtype} {value_cache.device}")
    print(f"block_table: {block_tables.shape} {block_tables.dtype}")
    print(f"context_lens: {context_lens.shape} {context_lens.dtype} val={context_lens}")
    print(f"num_heads={num_q_heads}, num_kv_heads={num_kv_heads}")
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


@pytest.mark.skipif(not hasattr(torch, "npu"), reason="npu not available")
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
    ) = data()

    # v2 requires a workspace tensor
    # workspace = torch.empty(0, dtype=torch.uint8, device=device)
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
    print(f"output: {out.shape} {out.dtype} {out.device}")
    print(f"out[:2]= {out.flatten()[:6]}")
    assert not torch.isnan(out).any()
