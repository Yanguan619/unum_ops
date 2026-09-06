"""Ascend 310P 长稳压测 — bev_pool + voxelization 联合稳定性测试。

验证：
1. 连续多轮调用无设备错误（507015 等）
2. 数据正确性不随迭代次数退化
3. 两个算子交替调用不冲突
4. 随机参数 + 大负载混合压力

运行方式:
    python -m pytest test/test_stability.py -v
"""
import gc
import time

import numpy as np
import pytest
import torch
import torch_npu

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(0)

from unum_ops.bev_pool import bev_pool, bev_pool_torch
from unum_ops.voxelization import voxelization


def _bev_pool_once(B, D, H, W, C, N, seed):
    """单次 bev_pool 调用，返回正确性 max_diff。"""
    torch.manual_seed(seed)
    feats = torch.randn(N, C, dtype=torch.float32)
    coords = torch.zeros(N, 4, dtype=torch.int64)
    coords[:, 0] = torch.randint(0, W, (N,))
    coords[:, 1] = torch.randint(0, H, (N,))
    coords[:, 2] = torch.randint(0, D, (N,))
    coords[:, 3] = torch.randint(0, B, (N,))
    ref = bev_pool_torch(feats, coords, B, D, H, W).out
    pts = feats.npu().contiguous()
    cs = coords.npu().contiguous()
    out = bev_pool(pts, cs, B, D, H, W)
    torch.npu.synchronize()
    return (out.out.cpu() - ref).abs().max().item()


def _voxel_once(N, seed):
    """单次 voxelization 调用，返回 num_voxels。"""
    np.random.seed(seed)
    points = np.random.randn(N, 4).astype(np.float32)
    pts = torch.from_numpy(points.copy()).npu()
    out = voxelization(pts)
    torch.npu.synchronize()
    return out.num_voxels


@pytest.mark.parametrize("n_iter,interval", [(1000, 200)])
def test_small_loop_stability(n_iter, interval):
    """小参数长稳：1000 次随机参数交替调用。"""
    errors = []
    for i in range(1, n_iter + 1):
        try:
            if i % 2 == 0:
                B = np.random.randint(1, 3)
                D = np.random.randint(1, 4)
                H = np.random.randint(4, 16)
                W = np.random.randint(4, 16)
                C = int(np.random.choice([8, 16, 32, 64, 80]))
                N = int(np.random.randint(10, 500))
                diff = _bev_pool_once(B, D, H, W, C, N, seed=i * 100)
                if diff > 1e-5:
                    errors.append((i, "bev_pool", f"diff={diff:.3e}"))
            else:
                N = int(np.random.randint(500, 50000))
                nvox = _voxel_once(N, seed=i * 100)
                if nvox < 0:
                    errors.append((i, "voxelization", f"num_voxels={nvox}"))
        except Exception as e:
            errors.append((i, "unknown", str(e)[:80]))
        if i % interval == 0:
            gc.collect()
            torch.npu.synchronize()
            torch.npu.empty_cache()
    assert not errors, f"长稳压测失败：{len(errors)} 个错误, 前5: {errors[:5]}"


@pytest.mark.parametrize("n_iter,interval", [(500, 100)])
def test_large_loop_stability(n_iter, interval):
    """大负载长稳：500 次，含 512k 点 bev_pool / 150k 点 voxelization。"""
    errors = []
    for i in range(1, n_iter + 1):
        try:
            if i % 2 == 0:
                B, D, H, W, C = 1, 8, 100, 100, 80
                N = 512000 if i % 4 == 0 else 50000
                diff = _bev_pool_once(B, D, H, W, C, N, seed=i)
                tol = 1e-4 if N == 512000 else 1e-5
                if diff > tol:
                    errors.append((i, "bev_pool", f"diff={diff:.3e}"))
            else:
                N = int(np.random.randint(10000, 150000))
                nvox = _voxel_once(N, seed=i)
                if nvox < 0:
                    errors.append((i, "voxelization", f"num_voxels={nvox}"))
        except Exception as e:
            errors.append((i, "unknown", str(e)[:80]))
        if i % interval == 0:
            gc.collect()
            torch.npu.synchronize()
            torch.npu.empty_cache()
    assert not errors, f"大负载长稳压测失败：{len(errors)} 个错误, 前5: {errors[:5]}"


if __name__ == "__main__":
    t0 = time.perf_counter()
    test_small_loop_stability(1000, 200)
    test_large_loop_stability(500, 100)
    print(f"长稳压测全部通过, 总耗时 {time.perf_counter() - t0:.0f}s")