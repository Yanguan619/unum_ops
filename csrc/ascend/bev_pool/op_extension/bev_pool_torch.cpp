#include <cstdint>
#include <vector>
#include <dlfcn.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "bev_pool_ops.h"

namespace {

struct aclTensor;
struct aclOpExecutor;
typedef void* aclrtStream;

constexpr int kAclFloat = 0;
constexpr int kAclInt32 = 3;
constexpr int kAclFormatNd = 2;
constexpr uint32_t kAclMemMallocHugeFirst = 0;

typedef int (*GetWorkspaceSizeFunc)(const aclTensor*, const aclTensor*, const aclTensor*,
                                    const aclTensor*, int64_t, int64_t, int64_t, int64_t,
                                    const aclTensor*, uint64_t*, aclOpExecutor**);
typedef int (*RunFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);
typedef aclTensor* (*CreateTensorFunc)(const int64_t*, uint64_t, int,
                                       const int64_t*, int64_t, int,
                                       const int64_t*, uint64_t, void*);
typedef int (*DestroyTensorFunc)(const aclTensor*);
typedef int (*RtMallocFunc)(void**, uint64_t, uint32_t);
typedef int (*RtFreeFunc)(void*);
typedef int (*RtSyncFunc)(aclrtStream);

struct AclnnFuncs {
    GetWorkspaceSizeFunc getWorkspaceSize = nullptr;
    RunFunc run = nullptr;
    CreateTensorFunc createTensor = nullptr;
    DestroyTensorFunc destroyTensor = nullptr;
    RtMallocFunc rtMalloc = nullptr;
    RtFreeFunc rtFree = nullptr;
    RtSyncFunc rtSync = nullptr;
    void (*executorClear)(void*) = nullptr;
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
            funcs.executorClear = Dlsym<void (*)(void*)>(nnop, "NnopbaseExecutorClear");
        }
        void* rt = dlopen("libascendcl.so", RTLD_LAZY | RTLD_LOCAL);
        if (rt) {
            funcs.rtMalloc = Dlsym<RtMallocFunc>(rt, "aclrtMalloc");
            funcs.rtFree = Dlsym<RtFreeFunc>(rt, "aclrtFree");
            funcs.rtSync = Dlsym<RtSyncFunc>(rt, "aclrtSynchronizeStream");
        }
        funcs.loaded = true;
    }
    return funcs;
}

aclTensor* MakeTensor(int dataType, const std::vector<int64_t>& shape,
                      void* storageData, int64_t storageOffset = 0) {
    auto& funcs = GetAclnnFuncs();
    int64_t ndim = shape.size();
    std::vector<int64_t> strides(ndim);
    int64_t s = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = s;
        s *= shape[i];
    }
    return funcs.createTensor(shape.data(), ndim, dataType, strides.data(),
                              storageOffset, kAclFormatNd,
                              shape.data(), ndim, storageData);
}

}  // namespace

namespace ascend_kernel {

BevPoolOutputs bev_pool(const at::Tensor& feats, const at::Tensor& coords,
                        const at::Tensor& interval_starts,
                        const at::Tensor& interval_lengths,
                        int64_t batch, int64_t depth, int64_t height, int64_t width) {
    auto& funcs = GetAclnnFuncs();
    TORCH_CHECK(funcs.getWorkspaceSize && funcs.run && funcs.createTensor &&
                    funcs.destroyTensor && funcs.rtMalloc && funcs.rtFree && funcs.rtSync,
                "aclnnBevPool symbols not found (install OPP package / CANN env)");

    auto N = feats.size(0);
    auto C = feats.size(1);
    aclrtStream aclStream = c10_npu::getCurrentNPUStream().stream(false);

    // 持久缓冲区：aclnn executor 是单例，缓存第一次调用的 tensor 地址。
    // 地址固定则缓存命中时 kerne 读写正确地址；地址变化则需清缓存。
    // 用 at::zeros 每次新建会导致地址变化 → 间歇性全零。
    static at::Tensor bufFeats, bufCoords, bufStarts, bufLengths, bufOut;
    bool shapeChanged = false;
    auto ensure = [&](at::Tensor& t, const std::vector<int64_t>& s,
                      at::ScalarType dt, const at::TensorOptions& opts) {
        bool same = t.defined() && t.sizes().vec() == s && t.dtype() == dt;
        if (!same) { t = torch::empty(s, opts.dtype(dt)); shapeChanged = true; }
    };
    auto opts = feats.options();
    ensure(bufFeats, {N, C}, feats.scalar_type(), opts);
    ensure(bufCoords, {N, 4}, coords.scalar_type(), opts);
    ensure(bufStarts, {interval_starts.size(0)}, interval_starts.scalar_type(), opts);
    ensure(bufLengths, {interval_lengths.size(0)}, interval_lengths.scalar_type(), opts);
    ensure(bufOut, {batch, depth, height, width, C}, at::kFloat, opts);
    bufFeats.copy_(feats); bufCoords.copy_(coords);
    bufStarts.copy_(interval_starts); bufLengths.copy_(interval_lengths);
    bufOut.zero_();

    aclTensor* featsTensor = MakeTensor(kAclFloat, {N, C},
        const_cast<void*>(bufFeats.storage().data()), bufFeats.storage_offset());
    aclTensor* coordsTensor = MakeTensor(kAclInt32, {N, 4},
        const_cast<void*>(bufCoords.storage().data()), bufCoords.storage_offset());
    aclTensor* startsTensor = MakeTensor(kAclInt32, {interval_starts.size(0)},
        const_cast<void*>(bufStarts.storage().data()), bufStarts.storage_offset());
    aclTensor* lengthsTensor = MakeTensor(kAclInt32, {interval_lengths.size(0)},
        const_cast<void*>(bufLengths.storage().data()), bufLengths.storage_offset());
    aclTensor* outTensor = MakeTensor(kAclFloat, {batch, depth, height, width, C},
        const_cast<void*>(bufOut.storage().data()), bufOut.storage_offset());

    // 每次调用都清除 executor 缓存：persistent 缓冲区在 shape 变化时
    // 会重分配（地址变化），不清除会导致 kernel 写到旧地址 → 全零。
    // 每次清确保 getWorkspaceSize 用当前地址重建。

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int st = funcs.getWorkspaceSize(featsTensor, coordsTensor, startsTensor, lengthsTensor,
                                    batch, depth, height, width, outTensor,
                                    &workspaceSize, &executor);
    TORCH_CHECK(st == 0, "aclnnBevPoolGetWorkspaceSize failed: ", st);

    void* wsDev = nullptr;
    if (workspaceSize > 0) {
        funcs.rtMalloc(&wsDev, workspaceSize, kAclMemMallocHugeFirst);
    }

    st = funcs.run(wsDev, workspaceSize, executor, aclStream);
    TORCH_CHECK(st == 0, "aclnnBevPool failed: ", st);
    funcs.rtSync(aclStream);

    // shape 变化 → 缓冲区重分配 → 地址变化 → 清除 executor 缓存
    // 使下次 getWorkspaceSize 用当前地址重建
    if (funcs.executorClear) {
        funcs.executorClear(executor);
    }

    funcs.destroyTensor(featsTensor); funcs.destroyTensor(coordsTensor);
    funcs.destroyTensor(startsTensor); funcs.destroyTensor(lengthsTensor);
    funcs.destroyTensor(outTensor);
    if (wsDev) funcs.rtFree(wsDev);

    return {bufOut.clone()};
}

}  // namespace ascend_kernel