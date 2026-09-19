"""pytest 共享配置：统一设备检测与按设备跳过。

用法:
    @pytest.mark.npu   — 仅 NPU 可用时执行，否则自动 skip
    @pytest.mark.cuda  — 仅 CUDA 可用时执行，否则自动 skip
    def test_xxx(): ...

已有测试自带的 skipif 不受影响；新测试统一使用上述标记。
"""

import pytest


def _detect_devices():
    try:
        import torch
    except ImportError:
        return False, False
    cuda = torch.cuda.is_available()
    npu = False
    try:
        import torch_npu  # noqa: F401

        npu = hasattr(torch, "npu") and torch.npu.is_available()
    except ImportError:
        pass
    return cuda, npu


CUDA_AVAILABLE, NPU_AVAILABLE = _detect_devices()


def pytest_configure(config):
    config.addinivalue_line("markers", "npu: run only when Ascend NPU is available")
    config.addinivalue_line("markers", "cuda: run only when CUDA is available")


def pytest_collection_modifyitems(config, items):
    skip_npu = pytest.mark.skip(reason="NPU not available")
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "npu" in item.keywords and not NPU_AVAILABLE:
            item.add_marker(skip_npu)
        if "cuda" in item.keywords and not CUDA_AVAILABLE:
            item.add_marker(skip_cuda)


@pytest.fixture(scope="session")
def device():
    from unum_ops import detect_device

    return detect_device()
