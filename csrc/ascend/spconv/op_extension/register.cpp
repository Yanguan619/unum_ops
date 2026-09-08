/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <torch/extension.h>
#include "spconv_gemm_ops.h"

TORCH_LIBRARY_FRAGMENT(unum, m) {
    m.def("spconv_gemm(Tensor feats, Tensor weight, Tensor bias, Tensor params) -> Tensor");
}

TORCH_LIBRARY_IMPL(unum, PrivateUse1, m) {
    m.impl("spconv_gemm", [](const at::Tensor& feats,
                              const at::Tensor& weight,
                              const at::Tensor& bias,
                              const at::Tensor& params) {
        auto out = ascend_kernel::spconv_gemm(feats, weight, bias, params);
        return out.out;
    });
}