import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "unum_ops"))

import torch, torch_npu
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(0)

from unum_ops.bev_pool import bev_pool, bev_pool_torch

benchmark_output_dir = "./benchmark/output/"
os.makedirs(benchmark_output_dir, exist_ok=True)

def _time_ms(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / repeat

B, D, H, W, C = 1, 8, 100, 100, 80
rng = torch.Generator()
rng.manual_seed(7)
num_points_list = [2000, 8000, 32000, 128000, 512000]

print(f"\n{'num_points':>10} | {'Torch Native(ms)':>18} | {'AscendC(ms)':>12}")
print("-" * 50)
for num_points in num_points_list:
    feats = torch.randn(num_points, C, generator=rng, dtype=torch.float32)
    coords = torch.stack([
        torch.randint(0, W, (num_points,), generator=rng, dtype=torch.int64),
        torch.randint(0, H, (num_points,), generator=rng, dtype=torch.int64),
        torch.randint(0, D, (num_points,), generator=rng, dtype=torch.int64),
        torch.randint(0, B, (num_points,), generator=rng, dtype=torch.int64),
    ], dim=1)
    pts = feats.npu().contiguous()
    cs = coords.npu().contiguous()

    asc_t = _time_ms(lambda: bev_pool(pts, cs, B, D, H, W))
    torch_t = _time_ms(lambda: bev_pool_torch(feats, coords, B, D, H, W))

    print(f"{num_points:>10} | {torch_t:>18.3f} | {asc_t:>12.3f}")

with open(os.path.join(benchmark_output_dir, "bev_pool_results.txt"), "w") as f:
    f.write(f"B={B} D={D} H={H} W={W} C={C}\n")
    f.write(f"{'num_points':>10} {'torch_ms':>12} {'ascendc_ms':>12}\n")
    for num_points in num_points_list:
        feats = torch.randn(num_points, C, generator=rng, dtype=torch.float32)
        coords = torch.stack([
            torch.randint(0, W, (num_points,), generator=rng, dtype=torch.int64),
            torch.randint(0, H, (num_points,), generator=rng, dtype=torch.int64),
            torch.randint(0, D, (num_points,), generator=rng, dtype=torch.int64),
            torch.randint(0, B, (num_points,), generator=rng, dtype=torch.int64),
        ], dim=1)
        pts = feats.npu().contiguous()
        cs = coords.npu().contiguous()
        asc_t = _time_ms(lambda: bev_pool(pts, cs, B, D, H, W))
        torch_t = _time_ms(lambda: bev_pool_torch(feats, coords, B, D, H, W))
        f.write(f"{num_points:>10} {torch_t:>12.3f} {asc_t:>12.3f}\n")

print(f"\nResults saved to {os.path.join(benchmark_output_dir, 'bev_pool_results.txt')}")