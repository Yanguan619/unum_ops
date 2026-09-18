/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef BEV_POOL_V3_OPS_H
#define BEV_POOL_V3_OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

struct BevPoolV3Outputs {
    at::Tensor out;
};

// v3 (without_depth path) — lifted from Ascend DrivingSDK bev_pool_v3.
// Inputs:
//   depth        : None (or [B,D,H,W] for with_depth path, unused here)
//   feat         : [N, C] float32 (already gathered by ranks_feat upstream)
//   ranks_depth  : None
//   ranks_feat   : None
//   ranks_bev    : [N] int32, FLAT 1D voxel indices in [0, B*D*H*W)
//   b,d,h,w,c    : BEV grid dims and channel count
// Output:
//   out          : [B, D, H, W, C] float32 (internal layout; Python layer permutes to [B,C,D,H,W])
BevPoolV3Outputs bev_pool_v3(const c10::optional<at::Tensor>& depth,
                             const at::Tensor& feat,
                             const c10::optional<at::Tensor>& ranks_depth,
                             const c10::optional<at::Tensor>& ranks_feat,
                             const at::Tensor& ranks_bev,
                             int64_t b, int64_t d, int64_t h, int64_t w, int64_t c);

}  // namespace ascend_kernel

#endif  // BEV_POOL_V3_OPS_H
