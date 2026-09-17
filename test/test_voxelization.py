"""Ascend 310P Voxelization 算子测试（pytest）。

覆盖 unum_ops.voxelization 接口，校验输出与 golden（spconv VoxelGeneratorV2）一致。

本文件同时支持两种运行环境：
  - NPU 环境：执行 AscendC kernel 正确性测试
  - 纯 CUDA 环境：执行 VoxelGeneratorV2 与 mmcv.ops.Voxelization 的交叉验证

运行方式:
    cd /data/workspace/unum_ops
    python -m pytest test/test_voxelization.py -v
"""
import os

import numpy as np
import pytest
import torch

try:
    import torch_npu
    NPU_AVAIL = torch.npu.is_available()
except Exception:
    NPU_AVAIL = False

_HERE = os.path.dirname(os.path.abspath(__file__))

from unum_ops.voxelization import voxelization
from unum_ops.spconv.utils import VoxelGeneratorV2

_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "csrc", "ascend", "voxelization", "test", "data"))


NPU_SKIP = pytest.mark.skipif(not NPU_AVAIL, reason="NPU not available")


# ── AscendC kernel 测试（NPU 环境） ───────────────────────────────────────────

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


@NPU_SKIP
def test_num_voxels():
    assert _run().num_voxels == 3941


@NPU_SKIP
def test_coords_match_golden(golden):
    assert np.array_equal(_run().coords.cpu().numpy(), golden["coords"])


@NPU_SKIP
def test_num_points_match_golden(golden):
    assert np.array_equal(_run().num_points.cpu().numpy(), golden["num_points"])


@NPU_SKIP
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


@NPU_SKIP
def test_random_vs_numpy_reference():
    """随机输入 vs numpy VoxelGeneratorV2 参考实现（跨设备交叉验证）"""
    np.random.seed(123)
    pts = np.random.uniform(-5, 60, (2000, 4)).astype(np.float32)
    pts[:, 2] = np.random.uniform(-3.0, 1.0, 2000).astype(np.float32)
    pcr = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
    voxel_size = (0.16, 0.16, 4.0)

    out = voxelization(torch.from_numpy(pts.copy()).npu(),
                       voxel_size=voxel_size, pcr=pcr)
    nvox = out.num_voxels
    assert nvox > 0

    gen = VoxelGeneratorV2(voxel_size=voxel_size, point_cloud_range=pcr,
                           max_num_points=32, max_voxels=40000)
    ref = gen.generate(pts)

    assert nvox == ref["coordinates"].shape[0], "voxel count mismatch vs numpy"
    order = np.lexsort([out.coords.cpu().numpy()[:, d] for d in range(2, -1, -1)])
    ref_order = np.lexsort([ref["coordinates"][:, d] for d in range(2, -1, -1)])
    assert np.array_equal(out.coords.cpu().numpy()[order], ref["coordinates"][ref_order]), \
        "coords mismatch vs numpy reference"


# ── Edge cases ──────────────────────────────────────────────────────────────

def _make_points_3d(xyz, intensity=1.0):
    pts = np.array([[x, y, z, intensity] for x, y, z in xyz], dtype=np.float32)
    return torch.from_numpy(pts).npu()


@NPU_SKIP
def test_empty_input():
    """N=0 应返回空输出。"""
    pts = torch.empty(0, 4, dtype=torch.float32).npu()
    out = voxelization(pts)
    assert out.num_voxels == 0
    assert out.voxels.shape[0] == 0
    assert out.coords.shape[0] == 0
    assert out.num_points.shape[0] == 0


@NPU_SKIP
def test_all_out_of_range():
    """全部越界 → 空输出。"""
    pts = _make_points_3d([(100, 100, 100), (-1, -1, -1)])
    out = voxelization(pts, pcr=(0, 0, 0, 10, 10, 10))
    assert out.num_voxels == 0


@NPU_SKIP
def test_max_voxels_truncation():
    """超过 max_voxels 时截断。"""
    np.random.seed(42)
    pts = np.random.uniform(0, 5, (500, 4)).astype(np.float32)
    pts = torch.from_numpy(pts).npu()
    out = voxelization(pts, voxel_size=(1, 1, 1), pcr=(0, 0, 0, 5, 5, 5),
                       max_voxels=100)
    assert out.num_voxels <= 100


@NPU_SKIP
def test_max_num_points_truncation():
    """同一体素超过 max_num_points 时截断。"""
    pts = np.zeros((100, 4), dtype=np.float32)
    pts[:, :3] = 1.0
    pts = torch.from_numpy(pts).npu()
    out = voxelization(pts, voxel_size=(10, 10, 10), pcr=(0, 0, 0, 100, 100, 100),
                       max_num_points=5)
    assert out.num_points[0].item() <= 5


@NPU_SKIP
def test_coordinate_order_zyx():
    """输出坐标顺序为 [z, y, x]。"""
    pts = _make_points_3d([(2.5, 1.5, 0.5)])
    out = voxelization(pts, voxel_size=(1, 1, 1), pcr=(0, 0, 0, 5, 5, 5))
    assert out.coords.shape[0] == 1
    assert out.coords[0, 0].item() == 0  # z
    assert out.coords[0, 1].item() == 1  # y
    assert out.coords[0, 2].item() == 2  # x


@NPU_SKIP
def test_single_point():
    """单点应正确体素化。"""
    pts = _make_points_3d([(1.5, 2.5, 3.5)])
    out = voxelization(pts, voxel_size=(1, 1, 1), pcr=(0, 0, 0, 10, 10, 10))
    assert out.num_voxels == 1
    assert out.num_points[0].item() == 1
    assert np.allclose(out.voxels[0, 0].cpu().numpy(), [1.5, 2.5, 3.5, 1.0])


@NPU_SKIP
def test_voxel_size_non_uniform():
    """非均匀 voxel_size 应正确映射。"""
    pts = _make_points_3d([(1.5, 2.5, 3.5)])
    out = voxelization(pts, voxel_size=(0.5, 2.0, 1.0), pcr=(0, 0, 0, 10, 10, 10))
    # x: 1.5/0.5=3, y: 2.5/2.0=1, z: 3.5/1.0=3
    assert out.coords[0, 2].item() == 3  # x
    assert out.coords[0, 1].item() == 1  # y
    assert out.coords[0, 0].item() == 3  # z


@NPU_SKIP
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


@NPU_SKIP
def test_large_grid_works():
    """小 voxel_size 大网格应正常工作（workspace 使用框架分配，不再受 voxels buffer 限制）。"""
    np.random.seed(42)
    pts = np.random.uniform(0, 5, (500, 4)).astype(np.float32)
    pts = torch.from_numpy(pts).npu()
    out = voxelization(pts, voxel_size=(0.08, 0.08, 4.0), max_voxels=40000)
    torch.npu.synchronize()
    assert out.num_voxels > 0


# ============================================================
# voxelization 参考实现 CUDA/mmcv 交叉验证（纯 CUDA 环境可运行）
# ============================================================

def _sort_voxel_sets(voxels, coords, num_points):
    """把体素按坐标排序，并同步重排 voxels/num_points，便于集合级比较"""
    m = coords.shape[0]
    order = np.lexsort([coords[:, d] for d in range(coords.shape[1] - 1, -1, -1)])
    return voxels[order], coords[order], num_points[order]


def _voxelize_local(points, voxel_size, pcr, max_num_points=32, max_voxels=40000):
    """unum_ops 语义参考（numpy VoxelGeneratorV2，与 AscendC 算子同一参考）"""
    gen = VoxelGeneratorV2(
        voxel_size=voxel_size, point_cloud_range=pcr,
        max_num_points=max_num_points, max_voxels=max_voxels)
    out = gen.generate(points)
    return out["voxels"], out["coordinates"], out["num_points_per_voxel"]


def _mmcv_available():
    """模块级函数：检查 mmcv.ops.Voxelization 是否可导入（供 @skipif 使用）"""
    try:
        from mmcv.ops import Voxelization
        return True
    except Exception:
        return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVoxelizationVsMMCVCUDA:
    """VoxelGeneratorV2（numpy 参考）vs 官方 mmcv.ops.Voxelization CUDA 算子"""

    @pytest.mark.skipif(not _mmcv_available(), reason="mmcv not available")
    def test_random_points_coords_match(self):
        """随机点云：体素坐标与数量 vs mmcv CUDA 一致"""
        from mmcv.ops import Voxelization
        np.random.seed(123)
        N = 2000
        pcr = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
        voxel_size = (0.16, 0.16, 4.0)
        pts = np.random.uniform(-5, 60, (N, 4)).astype(np.float32)
        pts[:, 2] = np.random.uniform(-3.0, 1.0, N).astype(np.float32)

        mmcv_vox = Voxelization(
            voxel_size=voxel_size, point_cloud_range=pcr,
            max_num_points=32, max_voxels=40000, deterministic=True)
        pts_try = torch.zeros(1, 4, dtype=torch.float32, device="cuda")
        try:
            mmcv_vox(pts_try)
        except RuntimeError as e:
            if "not found" in str(e):
                pytest.skip("mmcv Voxelization CUDA kernel not available (CPU-only build)")
            raise
        pts_cuda = torch.as_tensor(pts, dtype=torch.float32).cuda().contiguous()
        with torch.no_grad():
            v_ref, c_ref, n_ref, _ = mmcv_vox(pts_cuda)
        v_ref, c_ref, n_ref = v_ref.cpu().numpy(), c_ref.cpu().numpy(), n_ref.cpu().numpy()

        v_loc, c_loc, n_loc = _voxelize_local(pts, voxel_size, pcr)
        assert c_loc.shape[0] == c_ref.shape[0], \
            f"voxel count mismatch: local={c_loc.shape[0]} vs mmcv={c_ref.shape[0]}"
        _, c_loc_s, n_loc_s = _sort_voxel_sets(v_loc, c_loc, n_loc)
        _, c_ref_s, n_ref_s = _sort_voxel_sets(v_ref, c_ref, n_ref)
        assert np.array_equal(c_loc_s, c_ref_s), "coords mismatch vs mmcv CUDA"
        assert np.array_equal(n_loc_s, n_ref_s), "num_points mismatch vs mmcv CUDA"

    @pytest.mark.skipif(not _mmcv_available(), reason="mmcv not available")
    def test_voxel_point_sets_match(self):
        """同一 voxel 内的点集一致"""
        from mmcv.ops import Voxelization
        np.random.seed(7)
        pcr = (0, 0, 0, 10, 10, 10)
        voxel_size = (1, 1, 1)
        pts = np.random.uniform(0, 10, (1000, 4)).astype(np.float32)

        mmcv_vox = Voxelization(
            voxel_size=voxel_size, point_cloud_range=pcr,
            max_num_points=32, max_voxels=10000, deterministic=True)
        pts_cuda = torch.as_tensor(pts, dtype=torch.float32).cuda().contiguous()
        with torch.no_grad():
            v_ref, c_ref, n_ref, _ = mmcv_vox(pts_cuda)
        v_ref, c_ref, n_ref = v_ref.cpu().numpy(), c_ref.cpu().numpy(), n_ref.cpu().numpy()

        v_loc, c_loc, n_loc = _voxelize_local(pts, voxel_size, pcr, max_voxels=10000)
        assert c_loc.shape[0] == c_ref.shape[0]
        v_loc_s, c_loc_s, n_loc_s = _sort_voxel_sets(v_loc, c_loc, n_loc)
        v_ref_s, c_ref_s, n_ref_s = _sort_voxel_sets(v_ref, c_ref, n_ref)
        assert np.array_equal(c_loc_s, c_ref_s)
        assert np.array_equal(n_loc_s, n_ref_s)
        for i in range(c_loc_s.shape[0]):
            n = int(n_loc_s[i])
            loc_pts = np.sort(v_loc_s[i][:n], axis=0)
            ref_pts = np.sort(v_ref_s[i][:n], axis=0)
            assert np.allclose(loc_pts, ref_pts, atol=1e-4), f"voxel {i} point set mismatch"

    @pytest.mark.skipif(not _mmcv_available(), reason="mmcv not available")
    def test_edge_cases(self):
        """空输入 / 越界 / 单点，与 mmcv CUDA 行为一致"""
        from mmcv.ops import Voxelization
        pcr = (0, 0, 0, 5, 5, 5)
        voxel_size = (1, 1, 1)

        mmcv_vox = Voxelization(
            voxel_size=voxel_size, point_cloud_range=pcr,
            max_num_points=32, max_voxels=40000, deterministic=True)

        # 空输入
        pts = np.zeros((0, 4), dtype=np.float32)
        pts_cuda = torch.as_tensor(pts, dtype=torch.float32).cuda().contiguous()
        with torch.no_grad():
            v_ref, c_ref, n_ref, _ = mmcv_vox(pts_cuda)
        assert c_ref.shape[0] == 0
        v_loc, c_loc, n_loc = _voxelize_local(pts, voxel_size, pcr)
        assert c_loc.shape[0] == 0

        # 全部越界
        pts = np.array([[100, 100, 100, 1], [-1, -1, -1, 1]], dtype=np.float32)
        pts_cuda = torch.as_tensor(pts, dtype=torch.float32).cuda().contiguous()
        with torch.no_grad():
            v_ref, c_ref, n_ref, _ = mmcv_vox(pts_cuda)
        assert c_ref.shape[0] == 0
        v_loc, c_loc, n_loc = _voxelize_local(pts, voxel_size, pcr)
        assert c_loc.shape[0] == 0

        # 单点
        pts = np.array([[2.5, 1.5, 0.5, 1.0]], dtype=np.float32)
        pts_cuda = torch.as_tensor(pts, dtype=torch.float32).cuda().contiguous()
        with torch.no_grad():
            v_ref, c_ref, n_ref, _ = mmcv_vox(pts_cuda)
        assert c_ref.shape[0] == 1
        v_loc, c_loc, n_loc = _voxelize_local(pts, voxel_size, pcr)
        assert c_loc.shape[0] == 1
        assert np.allclose(c_loc[0], c_ref[0])

    @pytest.mark.skipif(not _mmcv_available(), reason="mmcv not available")
    def test_random_seeds(self):
        """多种子下坐标一致"""
        from mmcv.ops import Voxelization
        for seed in range(3):
            np.random.seed(seed)
            pcr = (0, 0, 0, 10, 10, 10)
            voxel_size = (1, 1, 1)
            pts = np.random.uniform(0, 10, (500, 4)).astype(np.float32)

            mmcv_vox = Voxelization(
                voxel_size=voxel_size, point_cloud_range=pcr,
                max_num_points=32, max_voxels=5000, deterministic=True)
            pts_cuda = torch.as_tensor(pts, dtype=torch.float32).cuda().contiguous()
            with torch.no_grad():
                v_ref, c_ref, n_ref, _ = mmcv_vox(pts_cuda)
            v_ref, c_ref, n_ref = v_ref.cpu().numpy(), c_ref.cpu().numpy(), n_ref.cpu().numpy()

            v_loc, c_loc, n_loc = _voxelize_local(pts, voxel_size, pcr, max_voxels=5000)
            assert c_loc.shape[0] == c_ref.shape[0], f"seed={seed} voxel count mismatch"
            _, c_loc_s, _ = _sort_voxel_sets(v_loc, c_loc, n_loc)
            _, c_ref_s, _ = _sort_voxel_sets(v_ref, c_ref, n_ref)
            assert np.array_equal(c_loc_s, c_ref_s), f"seed={seed} coords mismatch"