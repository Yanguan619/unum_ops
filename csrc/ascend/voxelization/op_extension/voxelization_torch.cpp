#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "voxelization_ops.h"

#ifndef UNUM_VENDOR_DIR
#define UNUM_VENDOR_DIR "voxelization"
#endif

namespace {

struct aclTensor;
struct aclOpExecutor;
struct aclFloatArray;
typedef void* aclrtStream;
typedef int aclError;

constexpr int kAclMemMallocHugeFirst = 0;
constexpr int kAclMemcpyDeviceToHost = 1;
constexpr int kAclMemcpyDeviceToDevice = 3;

typedef aclError (*GetWorkspaceSizeFunc)(const aclTensor*, const aclFloatArray*,
                                         const aclFloatArray*, int64_t, int64_t,
                                         const aclTensor*, const aclTensor*,
                                         const aclTensor*, const aclTensor*,
                                         uint64_t*, aclOpExecutor**);
typedef aclError (*RunFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);
typedef aclTensor* (*CreateTensorFunc)(const int64_t*, uint64_t, int,
                                       const int64_t*, int64_t, int,
                                       const int64_t*, uint64_t, void*);
typedef aclError (*DestroyTensorFunc)(const aclTensor*);
typedef aclFloatArray* (*CreateFloatArrayFunc)(const float*, uint64_t);
typedef aclError (*DestroyFloatArrayFunc)(const aclFloatArray*);
typedef aclError (*RtMallocFunc)(void**, uint64_t, uint32_t);
typedef aclError (*RtFreeFunc)(void*);
typedef aclError (*RtMemcpyFunc)(void*, size_t, const void*, size_t, int);
typedef aclError (*RtSyncFunc)(aclrtStream);

struct AclnnFuncs {
    GetWorkspaceSizeFunc getWorkspaceSize = nullptr;
    RunFunc run = nullptr;
    CreateTensorFunc createTensor = nullptr;
    DestroyTensorFunc destroyTensor = nullptr;
    CreateFloatArrayFunc createFloatArray = nullptr;
    DestroyFloatArrayFunc destroyFloatArray = nullptr;
    RtMallocFunc rtMalloc = nullptr;
    RtFreeFunc rtFree = nullptr;
    RtMemcpyFunc rtMemcpy = nullptr;
    RtSyncFunc rtSync = nullptr;
    bool loaded = false;
};

template <typename T>
T Dlsym(void* handle, const char* name) {
    return reinterpret_cast<T>(dlsym(handle, name));
}

std::string VendorLibPath() {
    const char* opp = std::getenv("ASCEND_OPP_PATH");
    if (!opp) opp = "/usr/local/Ascend/cann-9.0.0/opp";
    return std::string(opp) + "/vendors/" UNUM_VENDOR_DIR "/op_api/lib/libcust_opapi.so";
}

AclnnFuncs& GetAclnnFuncs() {
    static AclnnFuncs funcs;
    if (!funcs.loaded) {
        // 先 RTLD_GLOBAL 加载依赖库，使 libcust_opapi.so 内部符号可解析
        dlopen("libnnopbase.so", RTLD_LAZY | RTLD_GLOBAL);
        dlopen("libascendcl.so", RTLD_LAZY | RTLD_GLOBAL);

        std::string path = VendorLibPath();
        void* cust = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (cust) {
            funcs.getWorkspaceSize = Dlsym<GetWorkspaceSizeFunc>(cust, "aclnnVoxelizationGetWorkspaceSize");
            funcs.run = Dlsym<RunFunc>(cust, "aclnnVoxelization");
        }
        void* nnop = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_GLOBAL);
        if (nnop) {
            funcs.createTensor = Dlsym<CreateTensorFunc>(nnop, "aclCreateTensor");
            funcs.destroyTensor = Dlsym<DestroyTensorFunc>(nnop, "aclDestroyTensor");
            funcs.createFloatArray = Dlsym<CreateFloatArrayFunc>(nnop, "aclCreateFloatArray");
            funcs.destroyFloatArray = Dlsym<DestroyFloatArrayFunc>(nnop, "aclDestroyFloatArray");
        }
        void* rt = dlopen("libascendcl.so", RTLD_LAZY | RTLD_GLOBAL);
        if (rt) {
            funcs.rtMalloc = Dlsym<RtMallocFunc>(rt, "aclrtMalloc");
            funcs.rtFree = Dlsym<RtFreeFunc>(rt, "aclrtFree");
            funcs.rtMemcpy = Dlsym<RtMemcpyFunc>(rt, "aclrtMemcpy");
            funcs.rtSync = Dlsym<RtSyncFunc>(rt, "aclrtSynchronizeStream");
        }
        funcs.loaded = true;
    }
    return funcs;
}

aclTensor* MakeTensor(int dataType, const std::vector<int64_t>& shape, void* ptr) {
    auto& funcs = GetAclnnFuncs();
    int64_t ndim = shape.size();
    std::vector<int64_t> strides(ndim);
    int64_t s = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = s;
        s *= shape[i];
    }
    // 与原始实现一致：offset 位置传 ACL_FORMAT_ND，format 位置传 ACL_FORMAT_ND
    return funcs.createTensor(shape.data(), ndim, dataType, strides.data(),
                              /*ACL_FORMAT_ND*/ 2, /*ACL_FORMAT_ND*/ 2,
                              shape.data(), ndim, ptr);
}

}  // namespace

namespace ascend_kernel {

VoxelizationOutputs voxelization(const at::Tensor& points, c10::ArrayRef<double> voxel_size,
                                 c10::ArrayRef<double> pcr, int64_t max_num_points,
                                 int64_t max_voxels) {
    auto& funcs = GetAclnnFuncs();
    TORCH_CHECK(funcs.getWorkspaceSize && funcs.run && funcs.createTensor &&
                    funcs.destroyTensor && funcs.createFloatArray &&
                    funcs.destroyFloatArray && funcs.rtMalloc && funcs.rtFree &&
                    funcs.rtMemcpy && funcs.rtSync,
                "aclnnVoxelization symbols not found (install OPP package / CANN env)");

    TORCH_CHECK(points.is_privateuseone(), "points must be on NPU");
    TORCH_CHECK(points.scalar_type() == at::kFloat, "points must be float32");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 4, "points must be (N, 4)");
    TORCH_CHECK(voxel_size.size() == 3, "voxel_size must have 3 elements");
    TORCH_CHECK(pcr.size() == 6, "pcr must have 6 elements");

    auto N = points.size(0);
    aclrtStream aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // 直接用 torch tensor 作为输出（Python 层按 num_voxels 截断，
    // workspace 写在 voxels 尾部无影响），省去 aclrtMalloc + D2D 拷贝。
    at::Tensor voxels = at::empty({max_voxels, max_num_points, 4},
                                  points.options().dtype(at::kFloat));
    at::Tensor coords = at::empty({max_voxels, 3},
                                  points.options().dtype(at::kInt));
    at::Tensor num_points = at::empty({max_voxels},
                                      points.options().dtype(at::kInt));
    at::Tensor num_voxels = at::empty({1}, points.options().dtype(at::kInt));

    aclTensor* ptsTensor = MakeTensor(/*ACL_FLOAT*/ 0, {N, 4}, points.data_ptr());
    aclTensor* voxTensor = MakeTensor(/*ACL_FLOAT*/ 0, {max_voxels, max_num_points, 4},
                                      voxels.data_ptr());
    aclTensor* coordTensor = MakeTensor(/*ACL_INT32*/ 3, {max_voxels, 3}, coords.data_ptr());
    aclTensor* nptsTensor = MakeTensor(/*ACL_INT32*/ 3, {max_voxels}, num_points.data_ptr());
    aclTensor* nvoxTensor = MakeTensor(/*ACL_INT32*/ 3, {1}, num_voxels.data_ptr());

    float vs[3] = {(float)voxel_size[0], (float)voxel_size[1], (float)voxel_size[2]};
    float pcrArr[6] = {(float)pcr[0], (float)pcr[1], (float)pcr[2],
                       (float)pcr[3], (float)pcr[4], (float)pcr[5]};
    aclFloatArray* vsArr = funcs.createFloatArray(vs, 3);
    aclFloatArray* pcrArrAc = funcs.createFloatArray(pcrArr, 6);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclError st = funcs.getWorkspaceSize(
        ptsTensor, vsArr, pcrArrAc, max_num_points, max_voxels,
        voxTensor, coordTensor, nptsTensor, nvoxTensor,
        &workspaceSize, &executor);
    TORCH_CHECK(st == 0, "aclnnVoxelizationGetWorkspaceSize failed: ", st);

    void* wsDev = nullptr;
    if (workspaceSize > 0) {
        funcs.rtMalloc(&wsDev, workspaceSize, kAclMemMallocHugeFirst);
    }

    st = funcs.run(wsDev, workspaceSize, executor, aclStream);
    TORCH_CHECK(st == 0, "aclnnVoxelization failed: ", st);
    funcs.rtSync(aclStream);

    funcs.destroyTensor(ptsTensor);
    funcs.destroyTensor(voxTensor); funcs.destroyTensor(coordTensor);
    funcs.destroyTensor(nptsTensor); funcs.destroyTensor(nvoxTensor);
    funcs.destroyFloatArray(vsArr); funcs.destroyFloatArray(pcrArrAc);
    if (wsDev) funcs.rtFree(wsDev);

    return {voxels, coords, num_points, num_voxels};
}

}  // namespace ascend_kernel