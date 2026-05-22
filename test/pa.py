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

import torch
import torch_npu


def gen_data(device="npu:0"):
    dtype = torch.float16
    num_blocks = 1032
    block_size = 128
    num_kv_heads = 2
    num_q_heads = 32
    head_dim = 128
    batch_size = 1
    max_blocks_per_seq = 1

    # Query: [num_tokens, num_q_heads, head_dim]
    query = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=device)
    # KV cache: use same layout as vllm-ascend: [num_blocks, block_size, num_kv_heads, head_dim]
    kv_cache = torch.randn(2, num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    key_cache = kv_cache[0]
    value_cache = kv_cache[1]

    scaling = 0.08838834764831845
    block_tables = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq), dtype=torch.int32, device=device)
    seq_lens = torch.tensor([1], dtype=torch.int32, device=device)
    out = torch.zeros(batch_size, num_q_heads, head_dim, dtype=dtype, device=device)
    return query, key_cache, value_cache, num_q_heads, num_kv_heads, scaling, block_tables, seq_lens, out, kv_cache


def test_paged_attention():
    device = "npu:0"
    query, key_cache, value_cache, num_q_heads, num_kv_heads, scaling, block_tables, seq_lens, out, kv_cache = gen_data(device)

    print(f"query: {query.shape} {query.dtype} {query.device}")
    print(f"key_cache: {key_cache.shape} {key_cache.dtype} {key_cache.device}")
    print(f"value_cache: {value_cache.shape} {value_cache.dtype} {value_cache.device}")
    print(f"block_table: {block_tables.shape} {block_tables.dtype} {block_tables.device} val={block_tables}")
    print(f"context_lens: {seq_lens.shape} {seq_lens.dtype} {seq_lens.device} val={seq_lens}")
    print(f"num_heads={num_q_heads}, num_kv_heads={num_kv_heads}")

    # Try with workspace (from vllm-ascend graph mode)
    try:
        workspace = torch_npu._npu_paged_attention_get_workspace(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=scaling,
            block_table=block_tables,
            context_lens=seq_lens,
            out=out,
        )
        print(f"workspace: {workspace.shape} {workspace.dtype}")
    except Exception as e:
        print(f"get_workspace failed (non-fatal): {e}")
        workspace = None

    if workspace is not None:
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=scaling,
            block_table=block_tables,
            context_lens=seq_lens,
            out=out,
            workspace=workspace,
        )
        print("PASSED with workspace!")
    else:
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale_value=scaling,
            block_table=block_tables,
            context_lens=seq_lens,
            out=out,
        )
        print("PASSED without workspace!")


if __name__ == "__main__":
    test_paged_attention()
