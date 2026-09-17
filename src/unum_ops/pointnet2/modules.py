# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""nn.Module 组件：QueryAndGroup / GroupAll / CylinderQueryAndGroup。

内部算子通过子模块属性访问（如 ``query.ball_query(...)``）而非
``from .query import ball_query``：包 ``__init__`` 检测到 CUDA extension
时会替换子模块属性，属性访问保证替换对内部调用同样生效——
与原单文件实现中方法体对模块全局名的动态查找行为一致。
"""

import torch
from torch import nn

from . import grouping, query


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(
        self,
        radius: float,
        nsample: int,
        use_xyz=True,
        ret_grouped_xyz=False,
        normalize_xyz=False,
        sample_uniformly=False,
        ret_unique_cnt=False,
    ):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = query.ball_query(self.radius, self.nsample, xyz, new_xyz)
        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(
                        0, num_unique, (self.nsample - num_unique,), dtype=torch.long
                    )
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        if features is not None:
            grouped_features = grouping.grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz
        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False) -> None:
        super().__init__()
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features


class CylinderQueryAndGroup(nn.Module):
    r"""
    Groups with a cylinder query of radius and height

    Parameters
    ---------
    radius : float32
        Radius of cylinder
    hmin, hmax: float32
        endpoints of cylinder height in x-rotation axis
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(
        self,
        radius: float,
        hmin: float,
        hmax: float,
        nsample: int,
        use_xyz=True,
        ret_grouped_xyz=False,
        normalize_xyz=False,
        rotate_xyz=True,
        sample_uniformly=False,
        ret_unique_cnt=False,
    ):
        super().__init__()
        self.radius, self.nsample, self.hmin, self.hmax = radius, nsample, hmin, hmax
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.rotate_xyz = rotate_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, rot: torch.Tensor, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        rot : torch.Tensor
            rotation matrices (B, npoint, 3, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        B, npoint, _ = new_xyz.size()
        idx = query.cylinder_query(
            self.radius, self.hmin, self.hmax, self.nsample, xyz, new_xyz, rot.view(B, npoint, 9)
        )
        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(
                        0, num_unique, (self.nsample - num_unique,), dtype=torch.long
                    )
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        if self.rotate_xyz:
            grouped_xyz_ = grouped_xyz.permute(0, 2, 3, 1).contiguous()  # (B, npoint, nsample, 3)
            grouped_xyz_ = torch.matmul(grouped_xyz_, rot)
            grouped_xyz = grouped_xyz_.permute(0, 3, 1, 2).contiguous()
        if features is not None:
            grouped_features = grouping.grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz
        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)
