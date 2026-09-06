#include <cstdint>
#include <vector>
#include <dlfcn.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "bev_pool_ops.h"

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
typedef int (*DestroyTensorFunc)(const aclTensor*);

struct AclnnFuncs {
    GetWorkspaceSizeFunc getWorkspaceSize = nullptr;
    RunFunc run = nullptr;
    CreateTensorFunc createTensor = nullptr;
    DestroyTensorFunc destroyTensor = nullptr;
    bool loaded = false;
};

template <typename T>
T Dlsym(void* handle, const char* name) {
    return reinterpret_cast<T>(dlsym(handle, name));
}

AclnnFuncs& GetAclnnFuncs() {
    static AclnnFuncs funcs;
    if (!funcs.loaded) {
        void* cust = dlopen("libcust_opapi.so", RTLD_LAZY | RTLD_LOCAL);
        if (cust) {
            funcs.getWorkspaceSize = Dlsym<GetWorkspaceSizeFunc>(cust, "aclnnBevPoolGetWorkspaceSize");
            funcs.run = Dlsym<RunFunc>(cust, "aclnnBevPool");
        }
        void* nnop = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_LOCAL);
        if (nnop) {
            funcs.createTensor = Dlsym<CreateTensorFunc>(nnop, "aclCreateTensor");
            funcs.destroyTensor = Dlsym<DestroyTensorFunc>(nnop, "aclDestroyTensor");
        }
        funcs.loaded = true;
    }
    return funcs;
}

// 仿照 vllm-ascend ConvertType：把 at::Tensor 的实际 storage 直接包装成
// aclTensor（无 D2D 拷贝）。sizes()/strides() 返回的 IntArrayRef 由
// tensor 内部存储支撑，tensor 存活期间指针有效。
// storageDims 用栈上局部变量，aclCreateTensor 会拷贝。
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

    // 输入验证
    TORCH_CHECK(feats.dim() == 2, "feats must be 2-D (N, C), got ", feats.dim(), "-D");
    TORCH_CHECK(coords.dim() == 2 && coords.size(1) == 4,
                "coords must be 2-D (N, 4), got ", coords.dim(), "-D, size(1)=",
                coords.size(1));
    TORCH_CHECK(interval_starts.dim() == 1,
                "interval_starts must be 1-D, got ", interval_starts.dim(), "-D");
    TORCH_CHECK(interval_lengths.dim() == 1,
                "interval_lengths must be 1-D, got ", interval_lengths.dim(), "-D");
    TORCH_CHECK(interval_starts.size(0) == interval_lengths.size(0),
                "interval_starts and interval_lengths must have same length, got ",
                interval_starts.size(0), " vs ", interval_lengths.size(0));
    TORCH_CHECK(feats.size(0) == coords.size(0),
                "feats and coords must have same N, got ", feats.size(0),
                " vs ", coords.size(0));
    TORCH_CHECK(feats.scalar_type() == at::kFloat,
                "feats must be float32, got ", feats.scalar_type());
    TORCH_CHECK(coords.scalar_type() == at::kInt,
                "coords must be int32, got ", coords.scalar_type());
    TORCH_CHECK(interval_starts.scalar_type() == at::kInt,
                "interval_starts must be int32, got ", interval_starts.scalar_type());
    TORCH_CHECK(interval_lengths.scalar_type() == at::kInt,
                "interval_lengths must be int32, got ", interval_lengths.scalar_type());
    TORCH_CHECK(feats.is_privateuseone(),
                "feats must be on NPU device, got ", feats.device());
    TORCH_CHECK(coords.is_privateuseone(),
                "coords must be on NPU device, got ", coords.device());
    TORCH_CHECK(interval_starts.is_privateuseone(),
                "interval_starts must be on NPU device, got ", interval_starts.device());
    TORCH_CHECK(interval_lengths.is_privateuseone(),
                "interval_lengths must be on NPU device, got ", interval_lengths.device());
    TORCH_CHECK(feats.is_contiguous(),
                "feats must be contiguous, got strides=", feats.strides());
    TORCH_CHECK(coords.is_contiguous(),
                "coords must be contiguous, got strides=", coords.strides());
    TORCH_CHECK(interval_starts.is_contiguous(),
                "interval_starts must be contiguous");
    TORCH_CHECK(interval_lengths.is_contiguous(),
                "interval_lengths must be contiguous");
    TORCH_CHECK(batch > 0 && depth > 0 && height > 0 && width > 0,
                "batch/depth/height/width must be > 0, got ",
                batch, "/", depth, "/", height, "/", width);
    TORCH_CHECK(feats.size(0) > 0,
                "feats must have at least 1 point, got N=", feats.size(0));
    TORCH_CHECK(feats.size(1) > 0,
                "feats must have at least 1 channel, got C=", feats.size(1));

    auto N = feats.size(0);
    auto C = feats.size(1);

    // 输出用 2D [gridTotal, C] 避免 W 非 8 对齐时 DataCopy 写出未对齐。
    // kernel 按平坦 offset 写入，Python 层再 view+permute 回 [B,D,H,W,C]。
    int64_t gridTotal = batch * depth * height * width;
    TORCH_CHECK(gridTotal > 0,
                "gridTotal (B*D*H*W) must be > 0, got B=", batch, " D=", depth,
                " H=", height, " W=", width);
    TORCH_CHECK(gridTotal <= (int64_t)UINT32_MAX,
                "gridTotal (B*D*H*W) must fit in uint32, got ", gridTotal);
    at::Tensor out = at::zeros({gridTotal, C},
                               feats.options().dtype(at::kFloat));

    // 把用户 tensor 的实际 storage 传给 aclnn（不再拷贝进持久 buffer）。
    // 用户 tensor 地址稳定 → aclnn 按地址缓存的 kernel 始终有效。
    aclTensor* featsTensor = WrapTensor(feats, kAclFloat);
    aclTensor* coordsTensor = WrapTensor(coords, kAclInt32);
    aclTensor* startsTensor = WrapTensor(interval_starts, kAclInt32);
    aclTensor* lengthsTensor = WrapTensor(interval_lengths, kAclInt32);
    aclTensor* outTensor = WrapTensor(out, kAclFloat);

    // 每次调用重建 fresh executor（不跨调用缓存），消除 stale-descriptor 全零问题。
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int st = funcs.getWorkspaceSize(featsTensor, coordsTensor, startsTensor,
                                    lengthsTensor, batch, depth, height, width,
                                    outTensor, &workspaceSize, &executor);
    TORCH_CHECK(st == 0, "aclnnBevPoolGetWorkspaceSize failed: ", st);

    // workspace 用 NPU tensor（RAII，替代 aclrtMalloc/rtFree）。
    void* wsAddr = nullptr;
    at::Tensor wsTensor;
    if (workspaceSize > 0) {
        wsTensor = at::empty({(int64_t)workspaceSize},
                             feats.options().dtype(at::kByte));
        wsAddr = const_cast<void*>(wsTensor.storage().data());
    }

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    // 通过 OpCommand 提交，融入 torch_npu stream 异步执行顺序。
    // lambda 按值捕获（仿照 vllm-ascend EXEC_NPU_CMD），避免引用悬空。
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
