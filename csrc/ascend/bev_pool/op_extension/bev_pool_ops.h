/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef BEV_POOL_OPS_H
#define BEV_POOL_OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

// bev_pool 输出：out (gridTotal, C) float32（Python 层 view+permute 回 [B,C,D,H,W]）
// 输入：feats (N, C) float32（已排序）
//       coords (N, 4) int32（已排序）
//       interval_starts (K,) int32
//       interval_lengths (K,) int32
struct BevPoolOutputs {
    at::Tensor out;
};

BevPoolOutputs bev_pool(const at::Tensor& feats, const at::Tensor& coords,
                        const at::Tensor& interval_starts,
                        const at::Tensor& interval_lengths,
                        int64_t batch, int64_t depth, int64_t height, int64_t width);

}  // namespace ascend_kernel

#endif  // BEV_POOL_OPS_H