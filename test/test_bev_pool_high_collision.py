"""高压精度对比：unum_ops AscendC vs torch index_add_ 在高碰撞率场景下。

设计：
  - 网格小 → voxel 数少 → 每 voxel 平均点多 → 强制累积顺序发生作用
  - N 大 → 同样意义
  - C 多 → channel 维度也参与累加

如果 max_diff 仍然 0，说明 unum_ops AscendC 的 UB 顺序累加恰好等价于 atomic add
（不是偶然，而是 N·ε 在 fp32 的舍入路径上对称分布）。
"""
import os
import sys
import torch
import pytest

try:
    import torch_npu  # noqa: F401
    NPU_AVAILABLE = torch.npu.is_available()
except Exception:
    NPU_AVAILABLE = False

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

if NPU_AVAILABLE:
    torch.npu.set_compile_mode(jit_compile=False)
    import unum_ops.bev_pool as um_bev


def bev_pool_torch(feats, coords, B, D, H, W):
    """Atomic scatter-add reference (matches v3 AscendC summation order)."""
    coords_long = coords.long()
    linear_idx = (
        coords_long[:, 3] * (D * H * W)
        + coords_long[:, 2] * (H * W)
        + coords_long[:, 1] * W
        + coords_long[:, 0]
    )
    C = feats.shape[1]
    out_flat = torch.zeros(
        B * D * H * W, C, dtype=feats.dtype, device=feats.device
    )
    out_flat.index_add_(0, linear_idx, feats)
    return out_flat.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU required")
class TestBEVPoolHighCollision:
    """High-collision precision comparison."""

    @staticmethod
    def make_input(N, B, D, H, W, C, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        feats_cpu = torch.randn(N, C, dtype=torch.float32, generator=g)
        coords_cpu = torch.zeros(N, 4, dtype=torch.int32)
        coords_cpu[:, 0] = torch.randint(0, W, (N,), generator=g, dtype=torch.int32)
        coords_cpu[:, 1] = torch.randint(0, H, (N,), generator=g, dtype=torch.int32)
        coords_cpu[:, 2] = torch.randint(0, D, (N,), generator=g, dtype=torch.int32)
        coords_cpu[:, 3] = torch.randint(0, B, (N,), generator=g, dtype=torch.int32)
        return feats_cpu.to("npu"), coords_cpu.to("npu")

    @pytest.mark.parametrize("cfg", [
        # (name, B, D, H, W, C, N, seed)
        # 实际 BEV fusion 场景：avg/voxel ≤ 50，voxel 数 1k~16k
        ("mid_grid_100k",   2,  8, 32,  32,  80, 100000, 2),  # 16384 voxels, ~6 avg/voxel
        ("mid_grid_200k",   2,  8, 32,  32,  80, 200000, 3),  # ~12 avg/voxel
        ("dense_voxel_50k", 1,  4, 16,  16,  80,  50000, 1),  # 1024 voxels, ~49 avg/voxel
        ("dense_large_C",   1,  4, 16,  16, 256,  50000, 4),  # 1024 voxels, ~49 avg/voxel, C=256
    ])
    def test_high_collision(self, cfg):
        name, B, D, H, W, C, N, seed = cfg
        feats, coords = self.make_input(N, B, D, H, W, C, seed)
        ref = bev_pool_torch(feats.cpu(), coords.cpu(), B, D, H, W).to("npu")
        u = um_bev.bev_pool(feats, coords, B, D, H, W).out
        d = (u.float() - ref.float()).abs()
        n_voxel = B * D * H * W
        avg_per_voxel = N / n_voxel
        n_diff = (d > 0).sum().item()
        print(
            f"\n[{name:>20}] N={N:>6} voxels={n_voxel:>6} avg/voxel={avg_per_voxel:>6.1f} C={C:>3}"
            f"  | max_diff={d.max().item():.3e}  mean_diff={d.mean().item():.3e}"
            f"  | nonzero_voxels={n_diff:>6}/{ref.numel()//C}"
        )
        # 硬性断言：即使是 fp32 atomic 差异，也应该 < 1e-4 数量级
        # （unum vs torch 实测 0；real-world 应至少 6 个数量级精度）
        assert d.max().item() < 1e-4, (
            f"[{name}] max_diff {d.max().item():.3e} 超出 fp32 累加误差上限，"
            f"可能是 correctness bug 而非 precision"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))