import torch


def assert_close(a, b):
    if a.dtype == torch.bfloat16:
        assert torch.allclose(a, b, rtol=0.00001, atol=1e-8), f"Arrays not close: {a} vs {b}"
    else:
        assert torch.allclose(a, b), f"Arrays not close: {a} vs {b}"
