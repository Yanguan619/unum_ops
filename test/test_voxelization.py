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
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, "..", "src", "unum_ops"))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

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
