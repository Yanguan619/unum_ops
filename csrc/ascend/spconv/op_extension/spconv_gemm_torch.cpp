/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstdint>
#include <vector>
#include <string>
#include <cstdlib>
#include <dlfcn.h>
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "spconv_gemm_ops.h"

#ifndef UNUM_VENDOR_DIR
#define UNUM_VENDOR_DIR "spconv"
#endif

namespace {

struct aclTensor;
struct aclOpExecutor;
typedef void* aclrtStream;

constexpr int kAclFloat16 = 1;
constexpr int kAclFloat = 0;
constexpr int kAclInt32 = 3;
constexpr int kAclFormatNd = 2;

typedef int (*GetWorkspaceSizeFunc)(const aclTensor*, const aclTensor*, const aclTensor*,
                                    const aclTensor*, const aclTensor*, uint64_t*, aclOpExecutor**);
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
            funcs.getWorkspaceSize = Dlsym<GetWorkspaceSizeFunc>(cust, "aclnnSpconvGemmGetWorkspaceSize");
            funcs.run = Dlsym<RunFunc>(cust, "aclnnSpconvGemm");
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

SpconvGemmOutputs spconv_gemm(const at::Tensor& feats, const at::Tensor& weight,
                              const at::Tensor& bias, const at::Tensor& params) {
    auto& funcs = GetAclnnFuncs();
    TORCH_CHECK(funcs.getWorkspaceSize && funcs.run && funcs.createTensor &&
                    funcs.destroyTensor,
                "aclnnSpconvGemm symbols not found (install OPP package / CANN env)");

    TORCH_CHECK(feats.dim() == 2, "feats must be 2-D (N, K_flat), got ", feats.dim(), "-D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2-D (K_flat, C_out), got ", weight.dim(), "-D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1-D (C_out), got ", bias.dim(), "-D");
    TORCH_CHECK(params.dim() == 1 && params.size(0) == 5,
                "params must be 1-D of size 5, got ", params.dim(), "-D, size=", params.size(0));
    TORCH_CHECK(feats.size(1) == weight.size(0),
                "feats K_flat and weight K_flat mismatch: ", feats.size(1), " vs ", weight.size(0));
    TORCH_CHECK(bias.size(0) == weight.size(1),
                "bias size and weight C_out mismatch: ", bias.size(0), " vs ", weight.size(1));

    TORCH_CHECK(feats.scalar_type() == at::kHalf, "feats must be float16");
    TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be float16");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(params.scalar_type() == at::kInt, "params must be int32");

    TORCH_CHECK(feats.is_privateuseone(), "feats must be on NPU device");
    TORCH_CHECK(weight.is_privateuseone(), "weight must be on NPU device");
    TORCH_CHECK(bias.is_privateuseone(), "bias must be on NPU device");
    TORCH_CHECK(params.is_privateuseone(), "params must be on NPU device");

    TORCH_CHECK(feats.is_contiguous(), "feats must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");

    TORCH_CHECK(feats.size(0) > 0, "N must be > 0, got ", feats.size(0));
    TORCH_CHECK(feats.size(1) > 0, "K_flat must be > 0, got ", feats.size(1));
    TORCH_CHECK(weight.size(1) > 0, "C_out must be > 0, got ", weight.size(1));

    auto N = feats.size(0);
    auto C_out = weight.size(1);

    at::Tensor out = at::empty({N, C_out}, feats.options().dtype(at::kFloat));

    aclTensor* featsTensor = WrapTensor(feats, kAclFloat16);
    aclTensor* weightTensor = WrapTensor(weight, kAclFloat16);
    aclTensor* biasTensor = WrapTensor(bias, kAclFloat);
    aclTensor* paramsTensor = WrapTensor(params, kAclInt32);
    aclTensor* outTensor = WrapTensor(out, kAclFloat);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int st = funcs.getWorkspaceSize(featsTensor, weightTensor, biasTensor, paramsTensor,
                                    outTensor, &workspaceSize, &executor);
    TORCH_CHECK(st == 0, "aclnnSpconvGemmGetWorkspaceSize failed: ", st);

    void* wsAddr = nullptr;
    at::Tensor wsTensor;
    if (workspaceSize > 0) {
        wsTensor = at::empty({(int64_t)workspaceSize},
                             feats.options().dtype(at::kByte));
        wsAddr = const_cast<void*>(wsTensor.storage().data());
    }

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnSpconvGemm");
    cmd.SetCustomHandler(
        [funcs, wsAddr, workspaceSize, executor, aclStream,
         featsTensor, weightTensor, biasTensor, paramsTensor, outTensor]() -> int {
            int ret = funcs.run(wsAddr, workspaceSize, executor, aclStream);
            TORCH_CHECK(ret == 0, "aclnnSpconvGemm failed: ", ret);
            funcs.destroyTensor(featsTensor);
            funcs.destroyTensor(weightTensor);
            funcs.destroyTensor(biasTensor);
            funcs.destroyTensor(paramsTensor);
            funcs.destroyTensor(outTensor);
            return ret;
        });
    cmd.Run();

    return {out};
}

}  // namespace ascend_kernel