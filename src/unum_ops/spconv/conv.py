import hashlib
import itertools

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union

from .sparse_modules import SparseConvTensor, SparseModule


class SparseConvolution(SparseModule):
    """spconv SparseConvolution 的 torch-native 实现（纯 PyTorch，CPU/GPU/NPU 通用）

    提供通用的稀疏 N 维卷积（2D/3D），包括：
      - SubMConv（子流形卷积）：输出坐标 == 输入坐标，空间形状不变
      - SparseConv（步长下采样卷积）：输出坐标 = floor(输入坐标 / stride)
      - SparseInverseConv（转置卷积 / 上采样）：与对应的 SparseConv 配对使用

    优化说明（相对早期逐体素 Python 循环版本）：
      - 邻居查找：dict 逐点查找 → 坐标编码为 int64 key + torch.searchsorted 批量查找
      - 特征聚合：逐体素小矩阵乘 → 按核偏移循环的批量 addmm（torch.addmm_）
      - 邻居表缓存：按 (坐标指纹, 核/步长/padding/空间形状, 设备) 缓存，
        网络内坐标不变且设备不变时只构建一次；NPU 命中后无设备往返。

    设备策略（方案 A）：
      - 邻居表构建全程留在 indices 所在设备（CPU/NPU），避免 host-device 同步
      - _fingerprint 仅用 shape + 标量统计构造缓存 key（NPU 标量 .item() 同步开销小）
      - _neighbor_cached 按 (fp, device) 分别缓存，避免跨设备复用错误
      - _gather 中 neighbor_idx 已与 features 同设备，无需再 .to(device)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 bias: bool = True,
                 indice_key: Optional[str] = None,
                 ndim: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ndim = ndim
        self.stride = self._triple(stride)
        self.padding = self._triple(padding)
        self.dilation = self._triple(dilation)
        self.kernel_size = self._triple(kernel_size)
        self.indice_key = indice_key

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # 惰性缓存
        self._offsets_cache = None
        self._offsets_ndim = None
        self._nb_cache = {}

    def _triple(self, x) -> Tuple[int, ...]:
        if isinstance(x, (tuple, list)):
            return tuple(int(v) for v in x)
        return (int(x),) * self.ndim

    def _kernel_offsets(self):
        """返回所有卷积核偏移量，形状 (K^ndim, ndim)，顺序为 itertools.product 字典序"""
        offsets = []
        for o in itertools.product(*[range(k) for k in self.kernel_size]):
            offsets.append(o)
        return offsets

    @property
    def _flat_slice_idx(self):
        """与 _kernel_offsets 顺序一致的平坦权重索引（按 np.ravel_multi_index 规则）"""
        offsets = self._kernel_offsets()
        idx = []
        for o in offsets:
            idx.append(int(np.ravel_multi_index(o, tuple(self.kernel_size))))
        return idx

    @property
    def _offsets_t(self) -> torch.Tensor:
        """核偏移量张量 (K, ndim)，K = prod(kernel_size)。

        注意：返回 CPU 张量，调用方需自行 .to(device)。
        仅取每个偏移的前 ndim 维，与逐体素实现的邻居坐标计算保持一致
        （2D 卷积的 kernel_size 仍按 ndim=3 初始化，此处截断保留原语义）。
        """
        if self._offsets_cache is None or self._offsets_ndim != self.ndim:
            prod = self._kernel_offsets()
            offs = [p[:self.ndim] for p in prod]
            self._offsets_cache = torch.tensor(offs, dtype=torch.int64)
            self._offsets_ndim = self.ndim
        return self._offsets_cache

    def _offsets_on(self, device) -> torch.Tensor:
        """返回指定设备上的核偏移量 (K, ndim)"""
        return self._offsets_t.to(device)

    @property
    def _wT(self) -> torch.Tensor:
        """权重重排 (K, C_in, C_out)，k 与 _offsets_t 第 k 行对应"""
        K = self._offsets_t.shape[0]
        w = self.weight.reshape(self.out_channels, self.in_channels, K)
        return w.permute(2, 1, 0).contiguous()

    # ------------------------------------------------------------------
    # 邻居表构建（向量化，无 Python 逐点循环，全程留在输入设备）
    # ------------------------------------------------------------------

    def _encode(self, indices: torch.Tensor, spatial_shape) -> torch.Tensor:
        """将坐标 (N, 1+ndim) [batch, spatial...] 编码为单一 int64 key（空间内唯一）。

        spatial_shape 按 [D, H, W]（z,y,x）顺序给出，而坐标列按 [x, y, z] 顺序，
        因此列乘数需从最后一列（z）向第一列（x）累乘空间尺寸。
        所有计算在 indices 所在设备完成。
        """
        ndim = self.ndim
        rev = [int(spatial_shape[i]) for i in range(ndim - 1, -1, -1)]
        strides = [1] * (ndim + 1)
        acc = 1
        for j in range(ndim, 0, -1):
            strides[j] = acc
            acc *= rev[j - 1]
        strides[0] = acc
        # 构造 strides 张量，放在 indices 同设备
        strides_t = torch.tensor(strides, dtype=torch.int64, device=indices.device)
        inds = indices.to(torch.int64)
        keys = inds[:, 0] * strides_t[0]
        for d in range(ndim):
            keys = keys + inds[:, d + 1] * strides_t[d + 1]
        return keys

    def _lookup(self, in_indices: torch.Tensor, spatial_shape,
                cand_keys: torch.Tensor, cand_valid: torch.Tensor) -> torch.Tensor:
        """批量查找候选坐标对应的输入行索引；未命中置 -1。
        全程在 cand_keys 所在设备完成。"""
        in_keys = self._encode(in_indices, spatial_shape)
        N_in = in_keys.numel()
        if N_in == 0:
            return torch.full_like(cand_keys, -1)
        in_keys_sorted, order = torch.sort(in_keys)
        pos = torch.searchsorted(in_keys_sorted, cand_keys)
        pos = pos.clamp(0, N_in - 1)
        match = (in_keys_sorted[pos] == cand_keys) & cand_valid
        return torch.where(match, order[pos], torch.full_like(pos, -1))

    def _build_neighbor_idx(self, in_indices: torch.Tensor, out_coords: torch.Tensor,
                            spatial_shape, kernel, padding, stride) -> torch.Tensor:
        """普通（SubM / 步长）卷积邻居表。

        对每个输出坐标 q 与每个核偏移 k：候选输入坐标 = q*stride + k - padding，
        SubM 语义下（stride=1, padding=0）退化为 q + k。
        返回 (N_out, K) 邻居行索引，-1 表示邻居不存在。
        全程在 out_coords 所在设备完成（不再强制 .cpu()）。
        """
        ndim = self.ndim
        device = out_coords.device
        offs = self._offsets_on(device)
        K = offs.shape[0]
        N_out = out_coords.shape[0]
        sp_col = torch.tensor(
            [int(spatial_shape[i]) for i in range(ndim - 1, -1, -1)],
            dtype=torch.int64, device=device,
        )
        pad = torch.tensor([int(p) for p in padding[:ndim]], dtype=torch.int64, device=device)
        st = torch.tensor([int(s) for s in stride[:ndim]], dtype=torch.int64, device=device)
        if N_out == 0:
            return torch.empty(0, K, dtype=torch.int64, device=device)

        oc = out_coords
        cand_sp = oc[:, None, 1:] * st + offs[None] - pad   # (N_out, K, ndim) 列序 [x,y,z]
        in_range = (cand_sp >= 0) & (cand_sp < sp_col)
        valid = in_range.all(-1)
        batch = oc[:, None, 0:1].expand(N_out, K, 1)
        cand = torch.cat([batch, cand_sp], dim=-1).reshape(-1, ndim + 1)
        cand_keys = self._encode(cand, spatial_shape).reshape(N_out, K)
        cand_keys = torch.where(valid, cand_keys, torch.zeros_like(cand_keys))
        return self._lookup(in_indices, spatial_shape, cand_keys, valid)

    def _build_inverse_neighbor_idx(self, out_coords: torch.Tensor, in_indices: torch.Tensor,
                                    spatial_shape, kernel, padding, stride) -> torch.Tensor:
        """转置卷积邻居表。

        对每个细坐标 c 与核偏移 k：粗坐标 q = (c + padding - k) / stride（仅当整除且落在范围内）。
        返回 (N_out, K) 邻居行索引，-1 表示无贡献。
        全程在 out_coords 所在设备完成（不再强制 .cpu()）。
        """
        ndim = self.ndim
        device = out_coords.device
        offs = self._offsets_on(device)
        K = offs.shape[0]
        N_out = out_coords.shape[0]
        sp_col = torch.tensor(
            [int(spatial_shape[i]) for i in range(ndim - 1, -1, -1)],
            dtype=torch.int64, device=device,
        )
        pad = torch.tensor([int(p) for p in padding[:ndim]], dtype=torch.int64, device=device)
        st = torch.tensor([int(s) for s in stride[:ndim]], dtype=torch.int64, device=device)
        if N_out == 0:
            return torch.empty(0, K, dtype=torch.int64, device=device)

        oc = out_coords
        num = oc[:, None, 1:] + pad - offs[None]            # c + padding - k
        q = num // st
        div = (num % st) == 0
        in_range = (q >= 0) & (q < sp_col)
        valid = (div & in_range).all(-1)
        batch = oc[:, None, 0:1].expand(N_out, K, 1)
        cand = torch.cat([batch, q], dim=-1).reshape(-1, ndim + 1)
        cand_keys = self._encode(cand, spatial_shape).reshape(N_out, K)
        cand_keys = torch.where(valid, cand_keys, torch.zeros_like(cand_keys))
        return self._lookup(in_indices, spatial_shape, cand_keys, valid)

    # ------------------------------------------------------------------
    # 特征聚合（批量 GEMM）
    # ------------------------------------------------------------------

    def _gather(self, features: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
        """out[n] = bias + sum_k wT[k] @ features[neighbor_idx[n,k]]（邻居缺失贡献为 0）

        neighbor_idx 必须与 features 同设备（由上层 _neighbor_cached 保证）。
        """
        N, K = neighbor_idx.shape
        C_in = features.shape[1]
        C_out = self.out_channels
        if N == 0:
            return features.new_zeros(0, C_out)
        wT = self._wT  # (K, C_in, C_out)

        # 单次大 GEMM 代替 K 次小 addmm 循环
        # 1. 展平邻居索引，一次性 gather 所有邻居特征
        nb_flat = neighbor_idx.reshape(-1)               # (N*K,)
        valid = nb_flat >= 0                              # (N*K,)
        nb_safe = nb_flat.clamp(min=0)                   # 无效位置暂时指向第 0 行
        all_feats = features[nb_safe]                    # (N*K, C_in)
        # 无效邻居置零（不影响累加）
        all_feats = all_feats * valid.unsqueeze(-1).to(dtype=features.dtype)
        all_feats = all_feats.reshape(N, K, C_in)       # (N, K, C_in)

        # 2. 批量矩阵乘: (N, K, C_in) @ (K, C_in, C_out) → (N, K, C_out) → (N, C_out)
        # 用 einsum 代替 bmm（bmm 要求 batch 维一致，不支持广播）
        out = torch.einsum('nki,kio->no', all_feats, wT)  # (N, C_out)

        if self.bias is not None:
            out.add_(self.bias)
        return out

    # ------------------------------------------------------------------
    # 邻居表缓存
    # ------------------------------------------------------------------

    def _fingerprint(self, *parts) -> str:
        """构造缓存 key。仅用 shape + 标量统计（sum/min），避免 .cpu().numpy() 同步开销。

        注意：相比原 md5(numpy bytes) 版本，冲突概率略升，但用于缓存 key 可接受
        （误命中只会导致用错邻居表，forward 数值会立刻错误，测试能发现）。
        """
        h = hashlib.md5()
        for p in parts:
            if torch.is_tensor(p):
                # 仅取 shape + 数值摘要，避免 .cpu().numpy().tobytes() 的 host 同步
                h.update(str(tuple(p.shape)).encode())
                if p.numel() > 0:
                    # 标量统计的 .item() 只同步一个数，远快于整张量 .cpu()
                    h.update(repr(p.detach().to(torch.float32).sum().item()).encode())
                    h.update(repr(p.detach().to(torch.float32).min().item()).encode())
                    h.update(repr(p.detach().to(torch.float32).max().item()).encode())
            else:
                h.update(repr(p).encode())
        return h.hexdigest()

    def _neighbor_cached(self, fp: str, build, device=None):
        """按 (fp, device) 缓存邻居表。

        - device=None 表示与 build 返回值同设备（向后兼容）
        - 命中后无设备往返：缓存的就是目标设备上的 nb
        """
        key = (fp, str(device) if device is not None else 'cpu')
        nb = self._nb_cache.get(key)
        if nb is None:
            nb = build()
            self._nb_cache[key] = nb
            if len(self._nb_cache) > 8:
                self._nb_cache.pop(next(iter(self._nb_cache)))
        return nb

    def _downsample_coords(self, indices: torch.Tensor):
        """计算 SparseConv（步长卷积）的输出坐标：floor(in / stride)"""
        inds = indices.clone()
        for d in range(self.ndim):
            inds[:, d + 1] = inds[:, d + 1] // self.stride[d]
        return inds

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        raise NotImplementedError


class SubMConv3d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, indice_key=None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         bias=bias, indice_key=indice_key, ndim=3)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        features, indices = x.features, x.indices
        ndim = self.ndim
        spatial = tuple(x.spatial_shape[:ndim])
        device = features.device

        def build():
            # 不再 .cpu()，全程留在 features 设备
            in_c = indices.long().to(device).contiguous()
            return self._build_neighbor_idx(
                in_c, in_c, spatial,
                self.kernel_size, (0,) * ndim, (1,) * ndim,
            )

        fp = self._fingerprint(indices, ('subm', self.kernel_size, self.padding,
                                         self.stride, spatial))
        nb = self._neighbor_cached(fp, build, device)
        out_feat = self._gather(features, nb)
        out = SparseConvTensor(out_feat, indices, x.spatial_shape, x.batch_size,
                               grid=x.grid)
        out._indice_dict = dict(x._indice_dict)
        if self.indice_key is not None:
            out._indice_dict[self.indice_key] = {
                'in_coords': indices.clone(),
                'stride': tuple(self.stride),
                'padding': tuple(self.padding),
                'kernel': tuple(self.kernel_size),
            }
        return out


class SubMConv2d(SubMConv3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, indice_key=None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         bias=bias, indice_key=indice_key)
        self.ndim = 2


class SparseConv3d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, indice_key=None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         bias=bias, indice_key=indice_key, ndim=3)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        features, indices = x.features, x.indices
        in_shape = tuple(x.spatial_shape[:self.ndim])
        out_shape = tuple(
            (s + 2 * self.padding[d] - (self.kernel_size[d] - 1) - 1) // self.stride[d] + 1
            for d, s in enumerate(in_shape)
        )
        device = features.device

        def build():
            # 不再 .cpu()，全程留在 features 设备
            in_c = indices.long().to(device).contiguous()
            out_i = self._downsample_coords(in_c)
            keep = torch.ones(out_i.shape[0], dtype=torch.bool, device=device)
            for d in range(self.ndim):
                keep &= (out_i[:, d + 1] >= 0) & (out_i[:, d + 1] < out_shape[d])
            out_i = out_i[keep]
            out_i = torch.unique(out_i, dim=0)
            nb = self._build_neighbor_idx(
                in_c, out_i, in_shape,
                self.kernel_size, self.padding, self.stride,
            )
            return nb, out_i

        fp = self._fingerprint(indices, ('sparse', self.kernel_size, self.padding,
                                         self.stride, in_shape, out_shape))
        nb, out_indices = self._neighbor_cached(fp, build, device)
        out_feat = self._gather(features, nb)

        spatial_shape = list(out_shape) + list(x.spatial_shape[self.ndim:])
        out = SparseConvTensor(out_feat, out_indices, spatial_shape, x.batch_size,
                               grid=x.grid)
        out._indice_dict = dict(x._indice_dict)
        if self.indice_key is not None:
            out._indice_dict[self.indice_key] = {
                'in_coords': indices.clone(),
                'in_shape': list(in_shape),
                'stride': tuple(self.stride),
                'padding': tuple(self.padding),
                'kernel': tuple(self.kernel_size),
            }
        return out


class SparseConv2d(SparseConv3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, indice_key=None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         bias=bias, indice_key=indice_key)
        self.ndim = 2


class SparseInverseConv3d(SparseConvolution):
    """转置稀疏卷积（上采样）

    与配对的下采样 SparseConv3d 使用相同的 indice_key。
    输出坐标 = 下采样前的输入坐标（由输入 tensor 携带的 indice_dict 提供）。
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 indice_key=None, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=1, padding=0, bias=bias,
                         indice_key=indice_key, ndim=3)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        key = self.indice_key
        # 读取配对下采样卷积记录的 fine coords 与参数
        if key is not None and key in x._indice_dict:
            info = x._indice_dict[key]
            out_indices = info['in_coords']
            stride = tuple(info['stride'])
            padding = tuple(info['padding'])
            kernel = tuple(info['kernel'])
        else:
            raise ValueError(
                f"SparseInverseConv3d requires matching SparseConv3d with indice_key='{key}' "
                f"recorded in the input tensor's indice_dict."
            )

        features = x.features
        in_shape = tuple(x.spatial_shape[:self.ndim])
        device = features.device

        def build():
            # 不再 .cpu()，全程留在 features 设备
            fine_c = out_indices.long().to(device).contiguous()
            coarse_c = x.indices.long().to(device).contiguous()
            return self._build_inverse_neighbor_idx(
                fine_c, coarse_c, in_shape, kernel, padding, stride,
            )

        fp = self._fingerprint(out_indices, x.indices,
                               ('inverse', kernel, padding, stride, in_shape))
        nb = self._neighbor_cached(fp, build, device)
        out = self._gather(features, nb)

        spatial_shape = list(info.get('in_shape',
                                      tuple(s * st for s, st in zip(in_shape, stride))))
        return SparseConvTensor(out, out_indices, spatial_shape, x.batch_size,
                                grid=x.grid)


class SparseInverseConv2d(SparseInverseConv3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 indice_key=None, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         indice_key=indice_key, bias=bias)
        self.ndim = 2
