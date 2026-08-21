#ifndef VOXELIZATION_OPS_H
#define VOXELIZATION_OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

// voxelization 输出 4 个 tensor：
//   voxels (M, max_num_points, 4) fp32
//   coords (M, 3) int32
//   num_points (M,) int32
//   num_voxels (1,) int32
struct VoxelizationOutputs {
    at::Tensor voxels;
    at::Tensor coords;
    at::Tensor num_points;
    at::Tensor num_voxels;
};

VoxelizationOutputs voxelization(const at::Tensor& points, c10::ArrayRef<double> voxel_size,
                                 c10::ArrayRef<double> pcr, int64_t max_num_points,
                                 int64_t max_voxels);

}  // namespace ascend_kernel

#endif  // VOXELIZATION_OPS_H