"""Ascend 310P Voxelization — PyTorch 调用测试"""
import os
import numpy as np
import torch
import torch_npu

torch.ops.load_library(os.path.join(os.path.dirname(__file__), "build", "libvoxelization_ops.so"))

VOXEL_SIZE = [0.16, 0.16, 4.0]
PCR = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
MAX_NUM_POINTS = 32
MAX_VOXELS = 40000


def voxelization(points_np):
    """points_np: (N,4) float32 -> (voxels, coords, num_points, num_voxels)"""
    pts = torch.from_numpy(np.ascontiguousarray(points_np, dtype=np.float32)).npu()
    out = torch.ops.npu.voxelization(pts, VOXEL_SIZE, PCR, MAX_NUM_POINTS, MAX_VOXELS)
    M = out[3].item()
    return out[0][:M].cpu().numpy(), out[1][:M].cpu().numpy(), out[2][:M].cpu().numpy(), M


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "test", "data")
    points = np.fromfile(os.path.join(data_dir, "input", "points.bin"), dtype=np.float32).reshape(-1, 4)
    voxels, coords, npts, M = voxelization(points)
    print(f"num_voxels={M}")

    gv = np.fromfile(os.path.join(data_dir, "ref", "voxels.bin"), dtype=np.float32).reshape(M, 32, 4)
    gc = np.fromfile(os.path.join(data_dir, "ref", "coords.bin"), dtype=np.int32).reshape(M, 3)
    gn = np.fromfile(os.path.join(data_dir, "ref", "num_points.bin"), dtype=np.int32)
    assert M == len(gn) and np.array_equal(coords, gc) and np.array_equal(npts, gn)
    for v in range(M):
        gs = np.sort(gv[v][: int(gn[v])], axis=0)
        ks = np.sort(voxels[v][: int(npts[v])], axis=0)
        assert np.allclose(gs, ks, atol=0.0), f"voxel {v} mismatch"
    print("PASS: PyTorch voxelization output matches golden (M=3941)")
