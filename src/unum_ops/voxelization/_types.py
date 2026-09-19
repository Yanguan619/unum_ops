from dataclasses import dataclass

import torch

DEFAULT_MAX_NUM_POINTS = 32
DEFAULT_MAX_VOXELS = 40000
DEFAULT_VOXEL_SIZE = (0.16, 0.16, 4.0)
DEFAULT_PCR = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)


@dataclass
class VoxelizationOutput:
    voxels: torch.Tensor
    coords: torch.Tensor
    num_points: torch.Tensor
    num_voxels: int

    def __iter__(self):
        return iter((self.voxels, self.coords, self.num_points, self.num_voxels))