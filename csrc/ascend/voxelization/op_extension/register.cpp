#include <torch/extension.h>
#include "voxelization_ops.h"

// Meta backend：torch.compile / fx 推导输出 shape（full capacity）
namespace ascend_kernel {

std::vector<at::Tensor> voxelization_meta(const at::Tensor& points,
                                          c10::ArrayRef<double> voxel_size,
                                          c10::ArrayRef<double> pcr,
                                          int64_t max_num_points,
                                          int64_t max_voxels) {
    auto opt = points.options();
    return {
        at::empty({max_voxels, max_num_points, 4}, opt.dtype(at::kFloat)),
        at::empty({max_voxels, 3}, opt.dtype(at::kInt)),
        at::empty({max_voxels}, opt.dtype(at::kInt)),
        at::empty({1}, opt.dtype(at::kInt)),
    };
}

}  // namespace ascend_kernel

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("voxelization(Tensor points, float[3] voxel_size, float[6] pcr, "
          "int max_num_points, int max_voxels) -> Tensor[4]");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("voxelization", [](const at::Tensor& points, c10::ArrayRef<double> voxel_size,
                              c10::ArrayRef<double> pcr, int64_t max_num_points,
                              int64_t max_voxels) {
        auto out = ascend_kernel::voxelization(points, voxel_size, pcr,
                                               max_num_points, max_voxels);
        return std::vector<at::Tensor>{out.voxels, out.coords, out.num_points, out.num_voxels};
    });
}

TORCH_LIBRARY_IMPL(npu, Meta, m) {
    m.impl("voxelization", &ascend_kernel::voxelization_meta);
}