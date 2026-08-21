"""方案 A vs 方案 C 性能对比（_gather 优化版）

方案 A（当前 conv.py）：邻居表构建全程在输入设备 + bmm 优化 _gather
方案 C（候选）：邻居表构建在 CPU，缓存按设备分别存储（命中后无设备往返）

运行：
  cd /data/workspace/unum_ops
  python test/bench_spconv.py            # 默认 CPU + NPU（如可用）
  python test/bench_spconv.py --cpu-only  # 仅 CPU
  python test/bench_spconv.py --n 1000 --c 32 --spatial 30 30 30
"""
import argparse
import sys
import os
import time
import hashlib
import torch
import torch.nn as nn
torch.npu.set_compile_mode(jit_compile=False)

# 关键：把 src/unum_ops 加到 path，让 spconv 成为顶层包
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src', 'unum_ops'))

import spconv  # noqa: E402
from spconv.conv import SubMConv3d, SparseConvolution  # noqa: E402
from spconv.sparse_modules import SparseConvTensor  # noqa: E402


def make_tensor(n_voxels, n_channels, spatial_shape, device='cpu'):
    """生成随机 SparseConvTensor，4 列 indices [batch, x, y, z]

    注意：NPU 上直接调 torch.randn(device='npu') 可能触发
    StatelessRandomNormalV2 算子异常，因此统一在 CPU 生成随机数再 .to(device)。
    """
    D, H, W = spatial_shape
    # 坐标在 CPU 生成
    coords = torch.rand(n_voxels, 3) * torch.tensor([W, H, D], dtype=torch.float)
    coords = coords.int()
    batch = torch.randint(0, 2, (n_voxels, 1))
    indices = torch.cat([batch, coords], dim=1).long()
    indices = torch.unique(indices, dim=0)
    # 特征在 CPU 生成，再搬到目标设备（避免 NPU randn 算子异常）
    feats = torch.randn(indices.shape[0], n_channels)
    device = torch.device(device) if isinstance(device, str) else device
    if device.type != 'cpu':
        indices = indices.to(device)
        feats = feats.to(device)
    return SparseConvTensor(feats, indices, list(spatial_shape), batch_size=2)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'npu':
        try:
            torch.npu.synchronize()
        except Exception:
            pass


def time_fn(fn, device, repeat=5, warmup=2):
    """计时函数，返回平均耗时（毫秒）"""
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return sum(times) / len(times), min(times), max(times)


# ============================================================
# 方案 A：全程在 features 设备上构建（当前 conv.py 实现）
# ============================================================
class SubMConv3d_A(SubMConv3d):
    def forward(self, x):
        features, indices = x.features, x.indices
        spatial = tuple(x.spatial_shape[:3])
        device = features.device

        def build():
            in_c = indices.long().to(device).contiguous()
            return self._build_neighbor_idx(
                in_c, in_c, spatial,
                self.kernel_size, (0, 0, 0), (1, 1, 1),
            )
        fp = self._fingerprint(indices, ('subm', self.kernel_size, self.padding,
                                         self.stride, spatial))
        nb = self._neighbor_cached(fp, build, device)
        out_feat = self._gather(features, nb)
        return out_feat


# ============================================================
# 方案 C：CPU 构建 + 按设备缓存（命中后无设备往返）
# ============================================================
class SubMConv3d_C(SubMConv3d):
    def forward(self, x):
        features, indices = x.features, x.indices
        spatial = tuple(x.spatial_shape[:3])
        device = features.device

        def build():
            in_c = indices.long().cpu().contiguous()
            nb_cpu = self._build_neighbor_idx_cpu(
                in_c, in_c, spatial,
                self.kernel_size, (0, 0, 0), (1, 1, 1),
            )
            return nb_cpu.to(device) if device.type != 'cpu' else nb_cpu

        h = hashlib.md5()
        for p in (indices, ('subm', self.kernel_size, self.padding, self.stride, spatial)):
            if torch.is_tensor(p):
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(repr(p).encode())
        fp = h.hexdigest()

        key = (fp, str(device))
        nb = self._nb_cache_c.get(key)
        if nb is None:
            nb = build()
            self._nb_cache_c[key] = nb
        out_feat = self._gather(features, nb)
        return out_feat


def _build_neighbor_idx_cpu(self, in_indices, out_coords, spatial_shape,
                            kernel, padding, stride):
    """CPU 版本：强制所有操作在 CPU 上"""
    ndim = self.ndim
    offs = self._offsets_t
    K = offs.shape[0]
    N_out = out_coords.shape[0]
    sp_col = torch.tensor(
        [int(spatial_shape[i]) for i in range(ndim - 1, -1, -1)],
        dtype=torch.int64,
    )
    pad = torch.tensor([int(p) for p in padding[:ndim]], dtype=torch.int64)
    st = torch.tensor([int(s) for s in stride[:ndim]], dtype=torch.int64)
    if N_out == 0:
        return torch.empty(0, K, dtype=torch.int64)

    oc = out_coords
    cand_sp = oc[:, None, 1:] * st + offs[None] - pad
    in_range = (cand_sp >= 0) & (cand_sp < sp_col)
    valid = in_range.all(-1)
    batch = oc[:, None, 0:1].expand(N_out, K, 1)
    cand = torch.cat([batch, cand_sp], dim=-1).reshape(-1, ndim + 1)

    rev = [int(spatial_shape[i]) for i in range(ndim - 1, -1, -1)]
    strides = [1] * (ndim + 1)
    acc = 1
    for j in range(ndim, 0, -1):
        strides[j] = acc
        acc *= rev[j - 1]
    strides[0] = acc
    strides_t = torch.tensor(strides, dtype=torch.int64)

    inds = cand.to(torch.int64)
    cand_keys = inds[:, 0] * strides_t[0]
    for d in range(ndim):
        cand_keys = cand_keys + inds[:, d + 1] * strides_t[d + 1]
    cand_keys = cand_keys.reshape(N_out, K)
    cand_keys = torch.where(valid, cand_keys, torch.zeros_like(cand_keys))

    in_inds = in_indices.to(torch.int64)
    in_keys = in_inds[:, 0] * strides_t[0]
    for d in range(ndim):
        in_keys = in_keys + in_inds[:, d + 1] * strides_t[d + 1]

    N_in = in_keys.numel()
    if N_in == 0:
        return torch.full_like(cand_keys, -1)
    in_keys_sorted, order = torch.sort(in_keys)
    pos = torch.searchsorted(in_keys_sorted, cand_keys)
    pos = pos.clamp(0, N_in - 1)
    match = (in_keys_sorted[pos] == cand_keys) & valid
    return torch.where(match, order[pos], torch.full_like(pos, -1))


SubMConv3d_C._build_neighbor_idx_cpu = _build_neighbor_idx_cpu
SubMConv3d_C._nb_cache_c = {}


def run_bench(device_str, n_voxels=500, n_channels=16, spatial=(20, 20, 20)):
    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"设备: {device_str}  N={n_voxels}  C={n_channels}  spatial={spatial}")
    print(f"{'='*60}")

    x = make_tensor(n_voxels, n_channels, spatial, device=device)
    print(f"实际体素数: {x.indices.shape[0]}")

    # 模型权重也在 CPU 初始化再搬到设备（避免 NPU randn 问题）
    conv_a = SubMConv3d_A(n_channels, n_channels, 3, padding=1, indice_key='test')
    conv_c = SubMConv3d_C(n_channels, n_channels, 3, padding=1, indice_key='test')
    if device.type != 'cpu':
        conv_a = conv_a.to(device)
        conv_c = conv_c.to(device)
    conv_c.weight.data = conv_a.weight.data.clone()
    conv_c.bias.data = conv_a.bias.data.clone()

    # 数值一致性
    out_a = conv_a(x)
    out_c = conv_c(x)
    if out_a.shape != out_c.shape:
        print(f"[WARN] 输出形状不一致: A={out_a.shape} C={out_c.shape}")
    else:
        maxdiff = (out_a - out_c).abs().max().item()
        print(f"数值一致性 maxdiff: {maxdiff:.6e}  {'PASS' if maxdiff < 1e-4 else 'FAIL'}")

    # 清缓存
    conv_a._nb_cache = {}
    SubMConv3d_C._nb_cache_c = {}

    # 首次 forward（含 build）
    avg_a_first, _, _ = time_fn(lambda: conv_a(x), device, repeat=3, warmup=0)
    conv_a._nb_cache = {}
    SubMConv3d_C._nb_cache_c = {}
    avg_c_first, _, _ = time_fn(lambda: conv_c(x), device, repeat=3, warmup=0)

    # 缓存命中 forward（无 build）
    conv_a._nb_cache = {}
    SubMConv3d_C._nb_cache_c = {}
    _ = conv_a(x)
    _ = conv_c(x)
    avg_a_hit, _, _ = time_fn(lambda: conv_a(x), device, repeat=10, warmup=2)
    avg_c_hit, _, _ = time_fn(lambda: conv_c(x), device, repeat=10, warmup=2)

    print(f"\n首次 forward（含 build + bmm gather）:")
    print(f"  方案 A (设备上构建): {avg_a_first:8.2f} ms")
    print(f"  方案 C (CPU 构建+传输): {avg_c_first:8.2f} ms")
    print(f"  加速比 C/A: {avg_a_first/avg_c_first:.2f}x")

    print(f"\n缓存命中 forward（无 build，纯 bmm gather）:")
    print(f"  方案 A: {avg_a_hit:8.2f} ms")
    print(f"  方案 C: {avg_c_hit:8.2f} ms")
    print(f"  加速比 C/A: {avg_a_hit/avg_c_hit:.2f}x")

    return {
        'device': device_str,
        'first_a': avg_a_first, 'first_c': avg_c_first,
        'hit_a': avg_a_hit, 'hit_c': avg_c_hit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--n', type=int, default=500, help='体素数')
    parser.add_argument('--c', type=int, default=16, help='通道数')
    parser.add_argument('--spatial', type=int, nargs=3, default=[20, 20, 20])
    args = parser.parse_args()

    results = []
    results.append(run_bench('cpu', args.n, args.c, tuple(args.spatial)))

    if not args.cpu_only:
        try:
            import torch_npu  # noqa: F401
            torch.npu.set_device(0)
            results.append(run_bench('npu', args.n, args.c, tuple(args.spatial)))
        except Exception as e:
            print(f"\nNPU 不可用: {e}")

    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    print(f"{'设备':<8} {'首次A(ms)':<12} {'首次C(ms)':<12} {'加速比':<8} {'命中A(ms)':<12} {'命中C(ms)':<12} {'加速比':<8}")
    for r in results:
        print(f"{r['device']:<8} {r['first_a']:<12.2f} {r['first_c']:<12.2f} {r['first_a']/r['first_c']:<8.2f} "
              f"{r['hit_a']:<12.2f} {r['hit_c']:<12.2f} {r['hit_a']/r['hit_c']:<8.2f}")


if __name__ == '__main__':
    main()
