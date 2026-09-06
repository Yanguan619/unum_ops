"""Ascend 310P Voxelization 算子测试（pytest）。

覆盖 unum_ops.voxelization 接口，校验输出与 golden（spconv VoxelGeneratorV2）一致。

运行方式:
    cd /data/workspace/unum_ops
    python -m pytest test/test_voxelization.py -v
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch_npu

_HERE = os.path.dirname(os.path.abspath(__file__))

from unum_ops.voxelization import voxelization

_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "csrc", "ascend", "voxelization", "test", "data"))


@pytest.fixture(scope="module")
def golden():
    m = 3941
    return {
        "voxels": np.fromfile(os.path.join(_DATA_DIR, "ref", "voxels.bin"), dtype=np.float32).reshape(m, 32, 4),
        "coords": np.fromfile(os.path.join(_DATA_DIR, "ref", "coords.bin"), dtype=np.int32).reshape(m, 3),
        "num_points": np.fromfile(os.path.join(_DATA_DIR, "ref", "num_points.bin"), dtype=np.int32),
    }


def _run():
    points = np.fromfile(os.path.join(_DATA_DIR, "input", "points.bin"), dtype=np.float32).reshape(-1, 4)
    return voxelization(torch.from_numpy(points.copy()).npu())


def test_num_voxels():
    assert _run().num_voxels == 3941


def test_coords_match_golden(golden):
    assert np.array_equal(_run().coords.cpu().numpy(), golden["coords"])


def test_num_points_match_golden(golden):
    assert np.array_equal(_run().num_points.cpu().numpy(), golden["num_points"])


def test_voxel_point_sets_match_golden(golden):
    out = _run()
    gv, gn = golden["voxels"], golden["num_points"]
    vox = out.voxels.cpu().numpy()
    npts = out.num_points.cpu().numpy()
    for v in range(out.num_voxels):
        n = int(gn[v])
        assert npts[v] == n
        gs = np.sort(gv[v][:n], axis=0)
        ks = np.sort(vox[v][:n], axis=0)
        assert np.allclose(gs, ks, atol=0.0), f"voxel {v} mismatch"


# ── Edge cases ──────────────────────────────────────────────────────────────

def _make_points_3d(xyz, intensity=1.0):
    pts = np.array([[x, y, z, intensity] for x, y, z in xyz], dtype=np.float32)
    return torch.from_numpy(pts).npu()


def test_empty_input():
    """N=0 应返回空输出。"""
    pts = torch.empty(0, 4, dtype=torch.float32).npu()
    out = voxelization(pts)
    assert out.num_voxels == 0
    assert out.voxels.shape[0] == 0
    assert out.coords.shape[0] == 0
    assert out.num_points.shape[0] == 0


def test_all_out_of_range():
    """全部越界 → 空输出。"""
    pts = _make_points_3d([(100, 100, 100), (-1, -1, -1)])
    out = voxelization(pts, pcr=(0, 0, 0, 10, 10, 10))
    assert out.num_voxels == 0


def test_max_voxels_truncation():
    """超过 max_voxels 时截断。"""
    np.random.seed(42)
    pts = np.random.uniform(0, 5, (500, 4)).astype(np.float32)
    pts = torch.from_numpy(pts).npu()
    # 注：kernel 在 max_voxels < 64 时会 aicore 异常(507015)，此处用 ≥64 的安全值
    out = voxelization(pts, voxel_size=(1, 1, 1), pcr=(0, 0, 0, 5, 5, 5),
                       max_voxels=100)
    assert out.num_voxels <= 100


def test_max_num_points_truncation():
    """同一体素超过 max_num_points 时截断。"""
    pts = np.zeros((100, 4), dtype=np.float32)
    pts[:, :3] = 1.0
    pts = torch.from_numpy(pts).npu()
    out = voxelization(pts, voxel_size=(10, 10, 10), pcr=(0, 0, 0, 100, 100, 100),
                       max_num_points=5)
    assert out.num_points[0].item() <= 5


def test_coordinate_order_zyx():
    """输出坐标顺序为 [z, y, x]。"""
    pts = _make_points_3d([(2.5, 1.5, 0.5)])
    out = voxelization(pts, voxel_size=(1, 1, 1), pcr=(0, 0, 0, 5, 5, 5))
    assert out.coords.shape[0] == 1
    assert out.coords[0, 0].item() == 0  # z
    assert out.coords[0, 1].item() == 1  # y
    assert out.coords[0, 2].item() == 2  # x


def test_single_point():
    """单点应正确体素化。"""
    pts = _make_points_3d([(1.5, 2.5, 3.5)])
    out = voxelization(pts, voxel_size=(1, 1, 1), pcr=(0, 0, 0, 10, 10, 10))
    assert out.num_voxels == 1
    assert out.num_points[0].item() == 1
    assert np.allclose(out.voxels[0, 0].cpu().numpy(), [1.5, 2.5, 3.5, 1.0])


def test_voxel_size_non_uniform():
    """非均匀 voxel_size 应正确映射。"""
    pts = _make_points_3d([(1.5, 2.5, 3.5)])
    out = voxelization(pts, voxel_size=(0.5, 2.0, 1.0), pcr=(0, 0, 0, 10, 10, 10))
    # x: 1.5/0.5=3, y: 2.5/2.0=1, z: 3.5/1.0=3
    assert out.coords[0, 2].item() == 3  # x
    assert out.coords[0, 1].item() == 1  # y
    assert out.coords[0, 0].item() == 3  # z


def test_small_max_voxels_works():
    """max_voxels 过小不应报错（workspace 现在使用框架分配，独立于 voxels buffer）。"""
    np.random.seed(42)
    pts = np.random.uniform(0, 5, (500, 4)).astype(np.float32)
    pts = torch.from_numpy(pts).npu()
    for mv in [60, 40, 10, 1]:
        out = voxelization(pts, voxel_size=(1, 1, 1), pcr=(0, 0, 0, 5, 5, 5),
                          max_voxels=mv)
        torch.npu.synchronize()
        assert out.num_voxels <= mv, f"max_voxels={mv} got {out.num_voxels}"


def test_large_grid_works():
    """小 voxel_size 大网格应正常工作（workspace 使用框架分配，不再受 voxels buffer 限制）。"""
    np.random.seed(42)
    pts = np.random.uniform(0, 5, (500, 4)).astype(np.float32)
    pts = torch.from_numpy(pts).npu()
    out = voxelization(pts, voxel_size=(0.08, 0.08, 4.0), max_voxels=40000)
    torch.npu.synchronize()
    assert out.num_voxels > 0
