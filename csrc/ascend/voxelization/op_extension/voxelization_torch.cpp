#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "aclnn_voxelization.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "voxelization_ops.h"

namespace ascend_kernel {

namespace {

aclTensor* MakeTensor(aclDataType dataType, const at::Tensor& t) {
    auto ndim = t.dim();
    if (ndim == 0) {
        int64_t shape[1] = {1};
        int64_t strides[1] = {1};
        return aclCreateTensor(shape, 1, dataType, strides, ACL_FORMAT_ND, ACL_FORMAT_ND,
                               shape, 1, t.mutable_data_ptr());
    }
    std::vector<int64_t> shape(ndim);
    std::vector<int64_t> strides(ndim);
    for (int i = 0; i < ndim; i++) {
        shape[i] = t.size(i);
        strides[i] = t.stride(i);
    }
    return aclCreateTensor(shape.data(), ndim, dataType, strides.data(), ACL_FORMAT_ND,
                           ACL_FORMAT_ND, shape.data(), ndim, t.mutable_data_ptr());
}

}  // namespace

VoxelizationOutputs voxelization(const at::Tensor& points, c10::ArrayRef<double> voxel_size,
                                 c10::ArrayRef<double> pcr, int64_t max_num_points,
                                 int64_t max_voxels) {
    TORCH_CHECK(points.is_privateuseone(), "points must be on NPU");
    TORCH_CHECK(points.scalar_type() == at::kFloat, "points must be float32");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 4, "points must be (N, 4)");
    TORCH_CHECK(voxel_size.size() == 3, "voxel_size must have 3 elements");
    TORCH_CHECK(pcr.size() == 6, "pcr must have 6 elements");

    auto N = points.size(0);

    // 分配输出（full capacity, 用 empty 避免 zeros_like 的乱序问题）
    at::Tensor voxels = at::empty({max_voxels, max_num_points, 4},
                                  points.options().dtype(at::kFloat));
    at::Tensor coords = at::empty({max_voxels, 3},
                                  points.options().dtype(at::kInt));
    at::Tensor num_points = at::empty({max_voxels},
                                      points.options().dtype(at::kInt));
    at::Tensor num_voxels = at::empty({1},
                                      points.options().dtype(at::kInt));

    // 获取 NPU stream（stream(true) 清 queue，防乱序）
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 创建 aclTensor 包装 torch tensor
    aclTensor* ptsTensor = MakeTensor(ACL_FLOAT, points);
    aclTensor* voxTensor = MakeTensor(ACL_FLOAT, voxels);
    aclTensor* coordTensor = MakeTensor(ACL_INT32, coords);
    aclTensor* nptsTensor = MakeTensor(ACL_INT32, num_points);
    aclTensor* nvoxTensor = MakeTensor(ACL_INT32, num_voxels);

    // 参数数组
    float vs[3] = {(float)voxel_size[0], (float)voxel_size[1], (float)voxel_size[2]};
    float pcrArr[6] = {(float)pcr[0], (float)pcr[1], (float)pcr[2],
                       (float)pcr[3], (float)pcr[4], (float)pcr[5]};
    aclFloatArray* vsArr = aclCreateFloatArray(vs, 3);
    aclFloatArray* pcrArrAc = aclCreateFloatArray(pcrArr, 6);

    // 获取 workspace + executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus st = aclnnVoxelizationGetWorkspaceSize(
        ptsTensor, vsArr, pcrArrAc, max_num_points, max_voxels,
        voxTensor, coordTensor, nptsTensor, nvoxTensor,
        &workspaceSize, &executor);
    TORCH_CHECK(st == ACL_SUCCESS, "aclnnVoxelizationGetWorkspaceSize failed: ", st);

    // 分配 workspace
    void* wsDev = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&wsDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 执行
    st = aclnnVoxelization(wsDev, workspaceSize, executor, aclStream);
    TORCH_CHECK(st == ACL_SUCCESS, "aclnnVoxelization failed: ", st);

    // 同步等待完成
    aclrtSynchronizeStream(aclStream);

    // 清理
    aclDestroyTensor(ptsTensor);
    aclDestroyTensor(voxTensor);
    aclDestroyTensor(coordTensor);
    aclDestroyTensor(nptsTensor);
    aclDestroyTensor(nvoxTensor);
    aclDestroyFloatArray(vsArr);
    aclDestroyFloatArray(pcrArrAc);
    if (wsDev) aclrtFree(wsDev);

    return {voxels, coords, num_points, num_voxels};
}

}  // namespace ascend_kernel