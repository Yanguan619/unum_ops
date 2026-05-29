import os

os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")

from functools import wraps
from inspect import signature
from itertools import chain
from pprint import pprint

import pytest
import torch


def wrap_input_print_args(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        output_param = {"output": func(*args, **kwargs)}
        sig = signature(func)
        param_names = list(sig.parameters.keys()) + list(output_param.keys())

        str_print = {}
        for i, value in enumerate(chain(args, kwargs.values(), output_param.values())):
            name = param_names[i] if i < len(param_names) else f"arg{i}"
            str_print[name] = {}
            if hasattr(value, "shape"):
                str_print[name]["type"] = str(type(value))
                str_print[name]["shape"] = str(value.shape)
                str_print[name]["dtype"] = str(value.dtype)
                if isinstance(value, torch.Tensor) and value.numel() < 32:
                    str_print[name]["value"] = str(value)

            elif isinstance(value, list):
                str_print[name]["type"] = str(type(value))
                str_print[name]["value_preview10"] = str(
                    value[:10] if len(value) > 10 else value
                )
            else:
                str_print[name]["type"] = str(type(value))
                str_print[name]["value"] = str(value)

        print(f"{'-' * 30} 输入参数:")
        pprint(str_print)

        return output_param["output"]

    return wrapper


@wrap_input_print_args
def npu_pa(
    query,
    key_cache,
    value_cache,
    num_heads,
    num_kv_heads,
    scale_value,
    block_table,
    context_lens,
    out,
):
    import torch_npu

    torch_npu._npu_paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale_value=scale_value,
        block_table=block_table,
        context_lens=context_lens,
        out=out,
    )


@pytest.mark.skipif(not hasattr(torch, "npu"), reason="Unsupported npu device")
@pytest.mark.parametrize(
    "num_tokens, num_heads, head_size, num_blocks, block_size, num_kv_heads, head_size_k, head_size_v, max_blocks_per_seq",
    [
        (11, 32, 128, 1032, 128, 2, 128, 128, 1032),
        (4, 32, 128, 1032, 128, 2, 128, 128, 1032),
    ],
)
def test_paged_attention(
    num_tokens,
    num_heads,
    head_size,
    num_blocks,
    block_size,
    num_kv_heads,
    head_size_k,
    head_size_v,
    max_blocks_per_seq,
):
    """
    | 参数名 | 形状 (shape) | dtype | device | 描述 |
    |--------|--------------|------|------|------|
    | query | (num_tokens, num_heads, head_size) | bf16 | npu | 输入的query |
    | key_cache | (num_blocks, block_size, num_kv_heads, head_size_k) | bf16 | npu | kvcache的key |
    | value_cache | (num_blocks, block_size, num_kv_heads, head_size_v) | bf16 | npu | kvcache的value |
    | block_tables | (num_tokens, max_blocks_per_seq) | int32 | npu |  每个query的kvcache对应的block index |
    | context_lens | (batch_size,) | int32 | cpu |  每个query的kvcache对应的key/value的token数量 |
    | attnOut | (num_tokens, num_heads, head_size_v) | bf16 | npu | 输出的attention output |
    """

    device = "npu:0"
    scaling = 0.08838834764831845

    query = torch.randn(
        num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size_k,
        device=device,
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size_v,
        device=device,
        dtype=torch.bfloat16,
    )
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_tokens, max_blocks_per_seq),
        device=device,
        dtype=torch.int32,
    )
    context_lens = torch.full((num_tokens,), 0, dtype=torch.int32)
    out = torch.randn(
        num_tokens, num_heads, head_size_v, device=device, dtype=torch.bfloat16
    )

    npu_pa(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale_value=scaling,
        block_table=block_tables,
        context_lens=context_lens,
        out=out,
    )
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert torch.isfinite(out).all(), "Output contains non-finite values"
