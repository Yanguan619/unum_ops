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

aclTensor* MakeTensorFromPtr(aclDataType dataType, const std::vector<int64_t>& shape,
                             void* ptr) {
    int64_t ndim = shape.size();
    std::vector<int64_t> strides(ndim);
    int64_t s = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = s;
        s *= shape[i];
    }
    return aclCreateTensor(shape.data(), ndim, dataType, strides.data(), ACL_FORMAT_ND,
                           ACL_FORMAT_ND, shape.data(), ndim, ptr);
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

    // torch_npu 当前 stream（stream(true) 返回前清 queue，确保与之前 NPU 操作有序）
    aclrtStream aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 用 aclrtMalloc 分配输出
    size_t voxBytes = max_voxels * max_num_points * 4 * sizeof(float);
    size_t coordBytes = max_voxels * 3 * sizeof(int32_t);
    size_t nptsBytes = max_voxels * sizeof(int32_t);
    size_t nvoxBytes = 64;

    void* voxDev = nullptr; void* coordDev = nullptr;
    void* nptsDev = nullptr; void* nvoxDev = nullptr;
    aclrtMalloc(&voxDev, voxBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&coordDev, coordBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&nptsDev, nptsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&nvoxDev, nvoxBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    std::vector<int64_t> inShape = {N, 4};
    std::vector<int64_t> inStrides = {4, 1};
    aclTensor* ptsTensor = aclCreateTensor(inShape.data(), 2, ACL_FLOAT, inStrides.data(),
                                           ACL_FORMAT_ND, ACL_FORMAT_ND, inShape.data(), 2,
                                           points.data_ptr());

    aclTensor* voxTensor = MakeTensorFromPtr(ACL_FLOAT, {max_voxels, max_num_points, 4}, voxDev);
    aclTensor* coordTensor = MakeTensorFromPtr(ACL_INT32, {max_voxels, 3}, coordDev);
    aclTensor* nptsTensor = MakeTensorFromPtr(ACL_INT32, {max_voxels}, nptsDev);
    aclTensor* nvoxTensor = MakeTensorFromPtr(ACL_INT32, {1}, nvoxDev);

    float vs[3] = {(float)voxel_size[0], (float)voxel_size[1], (float)voxel_size[2]};
    float pcrArr[6] = {(float)pcr[0], (float)pcr[1], (float)pcr[2],
                       (float)pcr[3], (float)pcr[4], (float)pcr[5]};
    aclFloatArray* vsArr = aclCreateFloatArray(vs, 3);
    aclFloatArray* pcrArrAc = aclCreateFloatArray(pcrArr, 6);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus st = aclnnVoxelizationGetWorkspaceSize(
        ptsTensor, vsArr, pcrArrAc, max_num_points, max_voxels,
        voxTensor, coordTensor, nptsTensor, nvoxTensor,
        &workspaceSize, &executor);
    TORCH_CHECK(st == ACL_SUCCESS, "aclnnVoxelizationGetWorkspaceSize failed: ", st);

    void* wsDev = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&wsDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    st = aclnnVoxelization(wsDev, workspaceSize, executor, aclStream);
    TORCH_CHECK(st == ACL_SUCCESS, "aclnnVoxelization failed: ", st);
    aclrtSynchronizeStream(aclStream);

    int32_t nvox[16];
    aclrtMemcpy(nvox, nvoxBytes, nvoxDev, nvoxBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor voxels = at::empty({max_voxels, max_num_points, 4}, points.options().dtype(at::kFloat));
    at::Tensor coords = at::empty({max_voxels, 3}, points.options().dtype(at::kInt));
    at::Tensor num_points = at::empty({max_voxels}, points.options().dtype(at::kInt));
    at::Tensor num_voxels = at::empty({1}, points.options().dtype(at::kInt));

    aclrtMemcpy(voxels.mutable_data_ptr(), voxBytes, voxDev, voxBytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
    aclrtMemcpy(coords.mutable_data_ptr(), coordBytes, coordDev, coordBytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
    aclrtMemcpy(num_points.mutable_data_ptr(), nptsBytes, nptsDev, nptsBytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
    aclrtMemcpy(num_voxels.mutable_data_ptr(), 4, nvoxDev, 4, ACL_MEMCPY_DEVICE_TO_DEVICE);

    aclDestroyTensor(ptsTensor);
    aclDestroyTensor(voxTensor); aclDestroyTensor(coordTensor);
    aclDestroyTensor(nptsTensor); aclDestroyTensor(nvoxTensor);
    aclDestroyFloatArray(vsArr); aclDestroyFloatArray(pcrArrAc);
    aclrtFree(voxDev); aclrtFree(coordDev); aclrtFree(nptsDev); aclrtFree(nvoxDev);
    if (wsDev) aclrtFree(wsDev);

    return {voxels, coords, num_points, num_voxels};
}

}  // namespace ascend_kernel