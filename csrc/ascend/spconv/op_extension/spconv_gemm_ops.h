#ifndef SPCONV_GEMM_OPS_H
#define SPCONV_GEMM_OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

struct SpconvGemmOutputs {
    at::Tensor out;
};

SpconvGemmOutputs spconv_gemm(const at::Tensor& feats, const at::Tensor& weight,
                              const at::Tensor& bias, const at::Tensor& params);

}  // namespace ascend_kernel

#endif  // SPCONV_GEMM_OPS_H