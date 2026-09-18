/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * BevPoolV3 op_extension — dlopen the v3 OPP custom op_api, dispatch via
 * aclnnBEVPoolV3GetWorkspaceSize / aclnnBEVPoolV3. Mirrors unum_ops bev_pool
 * binding style but with v3's tensor contract:
 *   depth = None, ranks_depth = None, ranks_feat = None
 *   feat = [N, C] float32, ranks_bev = [N] int32 flat voxel indices
 *   attrs: with_depth=false, b,d,h,w,c
 */

#include <cstdint>
#include <vector>
#include <string>
#include <cstdlib>
#include <dlfcn.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "bev_pool_v3_ops.h"

#ifndef UNUM_VENDOR_DIR
#define UNUM_VENDOR_DIR "bevpoolV3"
#endif

namespace {

struct aclTensor;
struct aclOpExecutor;
typedef void* aclrtStream;

constexpr int kAclFloat = 0;
constexpr int kAclInt32 = 3;
constexpr int kAclFormatNd = 2;
// ACL_BOOL = 12 实际值，但 with_depth 是 native bool 直接传 aclnn 接口，
// 不需要 wrap 成 aclTensor，所以这里不定义 kAclBool。

// aclnnBEVPoolV3GetWorkspaceSize 真实参数顺序（见 aclnn_bev_pool_v3.h）：
//   depth, feat, ranksDepth, ranksFeat, ranksBev, withDepth(bool),
//   b, d, h, w, c (int64), out, workspaceSize, executor
typedef int (*GetWorkspaceSizeFunc)(const aclTensor*, const aclTensor*, const aclTensor*,
                                    const aclTensor*, const aclTensor*, bool,
                                    int64_t, int64_t, int64_t, int64_t, int64_t,
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
        if (!cust) {
            TORCH_CHECK(false, "dlopen failed for ", path, ": ", dlerror());
        }
        // Class name is BEVPoolV3 → CANN auto-generates aclnnBEVPoolV3*
        funcs.getWorkspaceSize = Dlsym<GetWorkspaceSizeFunc>(
            cust, "aclnnBEVPoolV3GetWorkspaceSize");
        funcs.run = Dlsym<RunFunc>(cust, "aclnnBEVPoolV3");
        // libnnopbase.so 在 aarch64-linux/devlib/ 下；优先 RTLD 搜索名让 ld.so 解析。
        void* nnop = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_LOCAL);
        if (!nnop) {
            // 兜底：直接读 devlib 路径
            const char* opp = std::getenv("ASCEND_HOME_PATH");
            if (!opp) opp = "/usr/local/Ascend/cann-9.0.0";
            std::string nnop_path = std::string(opp) + "/aarch64-linux/devlib/libnnopbase.so";
            nnop = dlopen(nnop_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        }
        TORCH_CHECK(nnop != nullptr, "dlopen libnnopbase.so failed: ", dlerror());
        funcs.createTensor = Dlsym<CreateTensorFunc>(nnop, "aclCreateTensor");
        funcs.destroyTensor = Dlsym<int (*)(const aclTensor*)>(nnop, "aclDestroyTensor");
        TORCH_CHECK(funcs.getWorkspaceSize != nullptr,
                    "dlsym aclnnBEVPoolV3GetWorkspaceSize failed: ", dlerror());
        TORCH_CHECK(funcs.run != nullptr,
                    "dlsym aclnnBEVPoolV3 failed: ", dlerror());
        TORCH_CHECK(funcs.createTensor != nullptr,
                    "dlsym aclCreateTensor failed: ", dlerror());
        TORCH_CHECK(funcs.destroyTensor != nullptr,
                    "dlsym aclDestroyTensor failed: ", dlerror());
        funcs.loaded = true;
    }
    return funcs;
}

aclTensor* WrapTensor(const at::Tensor& t, int acl_dtype) {
    auto& funcs = GetAclnnFuncs();
    const auto& sizes = t.sizes();
    const auto& strides = t.strides();
    int64_t storageDim = t.storage().nbytes() / t.itemsize();
    aclTensor* out = funcs.createTensor(
        sizes.data(), sizes.size(), acl_dtype,
        strides.data(), t.storage_offset(), kAclFormatNd,
        &storageDim, 1,
        const_cast<void*>(t.storage().data()));
    TORCH_CHECK(out != nullptr, "aclCreateTensor returned null for dtype=", acl_dtype,
                " dim=", sizes.size(), " total=", storageDim);
    return out;
}

}  // namespace

namespace ascend_kernel {

BevPoolV3Outputs bev_pool_v3(const c10::optional<at::Tensor>& depth,
                             const at::Tensor& feat,
                             const c10::optional<at::Tensor>& ranks_depth,
                             const c10::optional<at::Tensor>& ranks_feat,
                             const at::Tensor& ranks_bev,
                             int64_t b, int64_t d, int64_t h, int64_t w, int64_t c) {
    auto& funcs = GetAclnnFuncs();
    TORCH_CHECK(funcs.getWorkspaceSize && funcs.run && funcs.createTensor &&
                    funcs.destroyTensor,
                "aclnnBEVPoolV3 symbols not found (install OPP package / CANN env)");

    TORCH_CHECK(feat.dim() == 2, "feat must be 2-D (N, C), got ", feat.dim(), "-D");
    TORCH_CHECK(ranks_bev.dim() == 1, "ranks_bev must be 1-D (flat voxel indices), got ",
                ranks_bev.dim(), "-D");
    TORCH_CHECK(feat.size(0) == ranks_bev.size(0),
                "feat and ranks_bev must have same N, got ",
                feat.size(0), " vs ", ranks_bev.size(0));
    TORCH_CHECK(feat.scalar_type() == at::kFloat, "feat must be float32");
    TORCH_CHECK(ranks_bev.scalar_type() == at::kInt, "ranks_bev must be int32");
    TORCH_CHECK(feat.is_privateuseone(), "feat must be on NPU device");
    TORCH_CHECK(ranks_bev.is_privateuseone(), "ranks_bev must be on NPU device");
    TORCH_CHECK(feat.is_contiguous(), "feat must be contiguous");
    TORCH_CHECK(ranks_bev.is_contiguous(), "ranks_bev must be contiguous");
    TORCH_CHECK(b > 0 && d > 0 && h > 0 && w > 0 && c > 0,
                "b/d/h/w/c must be > 0, got ", b, "/", d, "/", h, "/", w, "/", c);
    TORCH_CHECK(feat.size(0) > 0, "feat must have at least 1 point");
    TORCH_CHECK(feat.size(1) > 0, "feat must have at least 1 channel");
    // with_depth path is out of scope for this comparison harness — only
    // without_depth is exercised, matching BEVFusion's use case.

    auto N = feat.size(0);
    auto C = feat.size(1);
    int64_t gridTotal = b * d * h * w;
    TORCH_CHECK(gridTotal > 0 && gridTotal <= (int64_t)UINT32_MAX,
                "gridTotal must be (0, UINT32_MAX], got ", gridTotal);

    // 输出 [B, D, H, W, C] 由 Python 层 permute 到 [B, C, D, H, W]
    at::Tensor out = at::zeros({b, d, h, w, C}, feat.options().dtype(at::kFloat));

    aclTensor* depthTensor = nullptr;
    aclTensor* ranksDepthTensor = nullptr;
    aclTensor* ranksFeatTensor = nullptr;
    aclTensor* featTensor = WrapTensor(feat, kAclFloat);
    aclTensor* ranksBevTensor = WrapTensor(ranks_bev, kAclInt32);
    aclTensor* outTensor = WrapTensor(out, kAclFloat);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int st = funcs.getWorkspaceSize(depthTensor, featTensor, ranksDepthTensor,
                                    ranksFeatTensor, ranksBevTensor,
                                    /*with_depth=*/false,
                                    b, d, h, w, c,
                                    outTensor, &workspaceSize, &executor);
    TORCH_CHECK(st == 0, "aclnnBEVPoolV3GetWorkspaceSize failed: ", st);

    void* wsAddr = nullptr;
    at::Tensor wsTensor;
    if (workspaceSize > 0) {
        wsTensor = at::empty({(int64_t)workspaceSize},
                             feat.options().dtype(at::kByte));
        wsAddr = const_cast<void*>(wsTensor.storage().data());
    }

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnBEVPoolV3");
    cmd.SetCustomHandler(
        [funcs, wsAddr, workspaceSize, executor, aclStream,
         featTensor, ranksBevTensor, outTensor]() -> int {
            int ret = funcs.run(wsAddr, workspaceSize, executor, aclStream);
            TORCH_CHECK(ret == 0, "aclnnBEVPoolV3 failed: ", ret);
            funcs.destroyTensor(featTensor);
            funcs.destroyTensor(ranksBevTensor);
            funcs.destroyTensor(outTensor);
            return ret;
        });
    cmd.Run();

    return {out};
}

}  // namespace ascend_kernel
