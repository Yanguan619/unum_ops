import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Any


class SparseConvTensor:
    """spconv SparseConvTensor 的 torch-native 实现（纯 PyTorch，CPU/GPU/NPU 通用）

    封装稀疏体素数据：特征 + 坐标 + 空间形状
    """

    def __init__(self,
                 features: torch.Tensor,
                 indices: torch.Tensor,
                 spatial_shape,
                 batch_size: int,
                 grid: Optional[torch.Tensor] = None):
        """
        Args:
            features: (N, C) 体素特征
            indices:  (N, 4) 坐标 (batch_idx, x, y, z)
            spatial_shape: (D, H, W) 空间大小（3D），(H, W) 空间大小（2D），
                           OpenPCDet 传 [D, H, W, 1, 0, 0] 这类 6 维时取前 ndim 维
            batch_size: batch 大小
            grid: 可选的密集网格索引（用于加速查找）
        """
        self.features = features
        self.indices = indices
        self.spatial_shape = tuple(spatial_shape)
        self.batch_size = batch_size
        self.grid = grid
        # 记录是否有索引映射（用于 replace_feature / SparseInverseConv）
        self._indice_dict: Dict[str, Any] = {}

    @property
    def ndim(self) -> int:
        """空间维度数（2D -> 2, 3D -> 3）"""
        return self.indices.shape[1] - 1

    @property
    def spatial_size(self):
        return self.spatial_shape[:self.ndim]

    @property
    def shape(self):
        """返回 (N, C)"""
        return self.features.shape

    def replace_feature(self, features: torch.Tensor):
        """用新特征替换当前特征，保持其他属性不变"""
        out = SparseConvTensor(
            features, self.indices, self.spatial_shape,
            self.batch_size, self.grid
        )
        out._indice_dict = self._indice_dict
        return out

    def dense(self, channels_first: bool = True) -> torch.Tensor:
        """将稀疏张量转为密集张量

        Args:
            channels_first: True -> (B, C, *spatial), False -> (B, *spatial, C)
        Returns:
            密集张量
        """
        spatial = self.spatial_shape[:self.ndim]
        N, C = self.features.shape
        B = self.batch_size

        if channels_first:
            dense = torch.zeros((B, C, *spatial), dtype=self.features.dtype)
        else:
            dense = torch.zeros((B, *spatial, C), dtype=self.features.dtype)

        indices = self.indices.long()
        for i in range(N):
            b = indices[i, 0].item()
            sp = [indices[i, d + 1].item() for d in range(self.ndim)]
            if any(sp[d] >= spatial[d] or sp[d] < 0 for d in range(self.ndim)):
                continue
            if b < B:
                if channels_first:
                    dense[(b, slice(None), *sp)] = self.features[i]
                else:
                    dense[(b, *sp, slice(None))] = self.features[i]

        return dense

    @staticmethod
    def from_dense(dense: torch.Tensor, channels_first: bool = True) -> 'SparseConvTensor':
        """从密集张量构建稀疏张量"""
        if channels_first:
            # (B, C, D, H, W) -> 提取非零位置
            B, C, D, H, W = dense.shape
            mask = dense.abs().sum(dim=1) > 0  # (B, D, H, W)
        else:
            # (B, D, H, W, C)
            B, D, H, W, C = dense.shape
            mask = dense.abs().sum(dim=-1) > 0  # (B, D, H, W)

        indices_list = []
        features_list = []
        for b in range(B):
            nonzero = mask[b].nonzero()  # (N, 3)
            for idx in nonzero:
                x, y, z = idx[0].item(), idx[1].item(), idx[2].item()
                indices_list.append([b, x, y, z])
                if channels_first:
                    features_list.append(dense[b, :, x, y, z])
                else:
                    features_list.append(dense[b, x, y, z, :])

        if features_list:
            indices = torch.tensor(indices_list, dtype=torch.int32)
            features = torch.stack(features_list).float()
        else:
            indices = torch.empty(0, 4, dtype=torch.int32)
            features = torch.empty(0, C if channels_first else dense.shape[-1])

        return SparseConvTensor(features, indices, (D, H, W), B)

    def to(self, device) -> 'SparseConvTensor':
        """设备迁移（CPU 版仅做 dtype/设备一致性）"""
        self.features = self.features.to(device)
        self.indices = self.indices.to(device)
        return self

    def cpu(self) -> 'SparseConvTensor':
        return self.to('cpu')

    def cuda(self) -> 'SparseConvTensor':
        return self.to('cuda')

    def __repr__(self):
        return (f"SparseConvTensor(features={self.features.shape}, "
                f"indices={self.indices.shape}, "
                f"spatial_shape={self.spatial_shape}, "
                f"batch_size={self.batch_size})")


class SparseModule(nn.Module):
    """spconv SparseModule 的 torch-native 实现（纯 PyTorch，CPU/GPU/NPU 通用）

    所有稀疏模块的基类，接受 SparseConvTensor 作为输入，输出 SparseConvTensor
    """

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        raise NotImplementedError


class SparseSequential(SparseModule):
    """spconv SparseSequential 的 torch-native 实现（纯 PyTorch，CPU/GPU/NPU 通用）

    顺序容器，依次执行各模块，数据流为 SparseConvTensor
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            # 字典方式构造: SparseSequential({'conv1': conv, 'relu1': relu})
            for name, module in args[0].items():
                self.add_module(name, module)
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            # 列表方式构造
            for idx, module in enumerate(args[0]):
                self.add_module(str(idx), module)
        else:
            # 位置参数构造: SparseSequential(conv, relu, conv2)
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, x):
        """依次通过所有模块

        - 稀疏模块（SparseModule）直接以 SparseConvTensor 为输入输出；
        - 普通 nn.Module（BatchNorm/ReLU 等）只作用在 .features 上，再 replace_feature。
        """
        for module in self:
            if isinstance(x, SparseConvTensor) and not isinstance(module, SparseModule):
                x = x.replace_feature(module(x.features))
            else:
                x = module(x)
        return x

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, idx) -> Union[SparseModule, 'SparseSequential']:
        """支持索引访问"""
        if isinstance(idx, slice):
            modules = list(self._modules.values())[idx]
            return SparseSequential(*modules)
        else:
            return list(self._modules.values())[idx]

    def __iter__(self):
        return iter(self._modules.values())


# ============================================================
# 适配层：让 SparseConv3dCPU / SubMConv3dCPU 接受 SparseConvTensor
# ============================================================

class SparseConv3dAdapter(SparseModule):
    """将 SparseConv3d 适配为 SparseModule 接口"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        from spconv.conv import SparseConv3d as SparseConv3dCPU
        self.conv = SparseConv3dCPU(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        # SparseConv3d.forward 接受 SparseConvTensor，返回 SparseConvTensor
        return self.conv(x)


class SubMConv3dAdapter(SparseModule):
    """将 SubMConv3d 适配为 SparseModule 接口"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        from spconv.conv import SubMConv3d as SubMConv3dCPU
        self.conv = SubMConv3dCPU(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, bias=bias
        )

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        # SubMConv3d.forward 接受 SparseConvTensor，返回 SparseConvTensor
        return self.conv(x)


# ============================================================
# 常用激活/归一化层的稀疏适配
# ============================================================

class SparseReLU(SparseModule):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        return x.replace_feature(self.relu(x.features))


class SparseBatchNorm1d(SparseModule):
    """对稀疏特征做 BatchNorm（等价于对 N 个体素做 BN）"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        return x.replace_feature(self.bn(x.features))


class SparseLinear(SparseModule):
    """对每个体素独立做线性变换"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        return x.replace_feature(self.linear(x.features))