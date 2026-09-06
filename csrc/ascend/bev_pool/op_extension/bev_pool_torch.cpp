#include <cstdint>
#include <vector>
#include <string>
#include <cstdlib>
#include <dlfcn.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "bev_pool_ops.h"

#ifndef UNUM_VENDOR_DIR
#define UNUM_VENDOR_DIR "bev_pool"
#endif

namespace {

struct aclTensor;
struct aclOpExecutor;
typedef void* aclrtStream;

constexpr int kAclFloat = 0;
constexpr int kAclInt32 = 3;
constexpr int kAclFormatNd = 2;

typedef int (*GetWorkspaceSizeFunc)(const aclTensor*, const aclTensor*, const aclTensor*,
                                    const aclTensor*, int64_t, int64_t, int64_t, int64_t,
                                    const aclTensor*, uint64_t*, aclOpExecutor**);
typedef int (*RunFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);
typedef aclTensor* (*CreateTensorFunc)(const int64_t*, uint64_t, int,
                                       const int64_t*, int64_t, int,
                                       const int64_t*, uint64_t, void*);

template <typename T>
T Dlsym(void* handle, const char* name) {
    return reinterpret_cast<T>(dlsym(handle, name));
}

std::string VendorLibPath() {
    const char* opp = std::getenv("ASCEND_OPP_PATH");
    if (!opp) opp = "/usr/local/Ascend/cann-9.0.0/opp";
    return std::string(opp) + "/vendors/" UNUM_VENDOR_DIR "/op_api/lib/libcust_opapi.so";
}

struct AclnnFuncs {
    GetWorkspaceSizeFunc getWorkspaceSize = nullptr;
    RunFunc run = nullptr;
    CreateTensorFunc createTensor = nullptr;
    int (*destroyTensor)(const aclTensor*) = nullptr;
    bool loaded = false;
};

AclnnFuncs& GetAclnnFuncs() {
    static AclnnFuncs funcs;
    if (!funcs.loaded) {
        std::string path = VendorLibPath();
        void* cust = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (cust) {
            funcs.getWorkspaceSize = Dlsym<GetWorkspaceSizeFunc>(cust, "aclnnBevPoolGetWorkspaceSize");
            funcs.run = Dlsym<RunFunc>(cust, "aclnnBevPool");
        }
        void* nnop = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_LOCAL);
        if (nnop) {
            funcs.createTensor = Dlsym<CreateTensorFunc>(nnop, "aclCreateTensor");
            funcs.destroyTensor = Dlsym<int (*)(const aclTensor*)>(nnop, "aclDestroyTensor");
        }
        funcs.loaded = true;
    }
    return funcs;
}

aclTensor* WrapTensor(const at::Tensor& t, int acl_dtype) {
    auto& funcs = GetAclnnFuncs();
    const auto& sizes = t.sizes();
    const auto& strides = t.strides();
    int64_t storageDim = t.storage().nbytes() / t.itemsize();
    return funcs.createTensor(
        sizes.data(), sizes.size(), acl_dtype,
        strides.data(), t.storage_offset(), kAclFormatNd,
        &storageDim, 1,
        const_cast<void*>(t.storage().data()));
}

}  // namespace

namespace ascend_kernel {

BevPoolOutputs bev_pool(const at::Tensor& feats, const at::Tensor& coords,
                        const at::Tensor& interval_starts,
                        const at::Tensor& interval_lengths,
                        int64_t batch, int64_t depth, int64_t height, int64_t width) {
    auto& funcs = GetAclnnFuncs();
    TORCH_CHECK(funcs.getWorkspaceSize && funcs.run && funcs.createTensor &&
                    funcs.destroyTensor,
                "aclnnBevPool symbols not found (install OPP package / CANN env)");

    TORCH_CHECK(feats.dim() == 2, "feats must be 2-D (N, C), got ", feats.dim(), "-D");
    TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 4,
                "coords must be 2-D (N, 4), got ", coords.dim(), "-D, size(1)=", coords.size(1));
    TORCH_CHECK(interval_starts.dim() == 1, "interval_starts must be 1-D, got ", interval_starts.dim(), "-D");
    TORCH_CHECK(interval_lengths.dim() == 1, "interval_lengths must be 1-D, got ", interval_lengths.dim(), "-D");
    TORCH_CHECK(interval_starts.size(0) == interval_lengths.size(0),
                "interval_starts and interval_lengths must have same length, got ",
                interval_starts.size(0), " vs ", interval_lengths.size(0));
    TORCH_CHECK(feats.size(0) == coords.size(0), "feats and coords must have same N, got ",
                feats.size(0), " vs ", coords.size(0));
    TORCH_CHECK(feats.scalar_type() == at::kFloat, "feats must be float32, got ", feats.scalar_type());
    TORCH_CHECK(coords.scalar_type() == at::kInt, "coords must be int32, got ", coords.scalar_type());
    TORCH_CHECK(interval_starts.scalar_type() == at::kInt, "interval_starts must be int32");
    TORCH_CHECK(interval_lengths.scalar_type() == at::kInt, "interval_lengths must be int32");
    TORCH_CHECK(feats.is_privateuseone(), "feats must be on NPU device, got ", feats.device());
    TORCH_CHECK(coords.is_privateuseone(), "coords must be on NPU device, got ", coords.device());
    TORCH_CHECK(interval_starts.is_privateuseone(), "interval_starts must be on NPU device");
    TORCH_CHECK(interval_lengths.is_privateuseone(), "interval_lengths must be on NPU device");
    TORCH_CHECK(feats.is_contiguous(), "feats must be contiguous");
    TORCH_CHECK(coords.is_contiguous(), "coords must be contiguous");
    TORCH_CHECK(interval_starts.is_contiguous(), "interval_starts must be contiguous");
    TORCH_CHECK(interval_lengths.is_contiguous(), "interval_lengths must be contiguous");
    TORCH_CHECK(batch > 0 && depth > 0 && height > 0 && width > 0,
                "batch/depth/height/width must be > 0, got ", batch, "/", depth, "/", height, "/", width);
    TORCH_CHECK(feats.size(0) > 0, "feats must have at least 1 point, got N=", feats.size(0));
    TORCH_CHECK(feats.size(1) > 0, "feats must have at least 1 channel, got C=", feats.size(1));

    auto N = feats.size(0);
    auto C = feats.size(1);
    int64_t gridTotal = batch * depth * height * width;
    TORCH_CHECK(gridTotal > 0, "gridTotal must be > 0");
    TORCH_CHECK(gridTotal <= (int64_t)UINT32_MAX, "gridTotal must fit in uint32, got ", gridTotal);

    at::Tensor out = at::zeros({gridTotal, C}, feats.options().dtype(at::kFloat));

    aclTensor* featsTensor = WrapTensor(feats, kAclFloat);
    aclTensor* coordsTensor = WrapTensor(coords, kAclInt32);
    aclTensor* startsTensor = WrapTensor(interval_starts, kAclInt32);
    aclTensor* lengthsTensor = WrapTensor(interval_lengths, kAclInt32);
    aclTensor* outTensor = WrapTensor(out, kAclFloat);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int st = funcs.getWorkspaceSize(featsTensor, coordsTensor, startsTensor,
                                    lengthsTensor, batch, depth, height, width,
                                    outTensor, &workspaceSize, &executor);
    TORCH_CHECK(st == 0, "aclnnBevPoolGetWorkspaceSize failed: ", st);

    void* wsAddr = nullptr;
    at::Tensor wsTensor;
    if (workspaceSize > 0) {
        wsTensor = at::empty({(int64_t)workspaceSize},
                             feats.options().dtype(at::kByte));
        wsAddr = const_cast<void*>(wsTensor.storage().data());
    }

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnBevPool");
    cmd.SetCustomHandler(
        [funcs, wsAddr, workspaceSize, executor, aclStream,
         featsTensor, coordsTensor, startsTensor, lengthsTensor, outTensor]() -> int {
            int ret = funcs.run(wsAddr, workspaceSize, executor, aclStream);
            TORCH_CHECK(ret == 0, "aclnnBevPool failed: ", ret);
            funcs.destroyTensor(featsTensor);
            funcs.destroyTensor(coordsTensor);
            funcs.destroyTensor(startsTensor);
            funcs.destroyTensor(lengthsTensor);
            funcs.destroyTensor(outTensor);
            return ret;
        });
    cmd.Run();

    return {out};
}

}  // namespace ascend_kernel