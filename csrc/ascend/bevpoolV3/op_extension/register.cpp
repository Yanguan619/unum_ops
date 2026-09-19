/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <torch/extension.h>
#include "bev_pool_v3_ops.h"

TORCH_LIBRARY_FRAGMENT(unum, m) {
    m.def("bev_pool_v3(Tensor? depth, Tensor feat, Tensor? ranks_depth, "
          "Tensor? ranks_feat, Tensor ranks_bev, int b, int d, int h, int w, int c) -> Tensor");
}

TORCH_LIBRARY_IMPL(unum, PrivateUse1, m) {
    m.impl("bev_pool_v3", [](const c10::optional<at::Tensor>& depth,
                             const at::Tensor& feat,
                             const c10::optional<at::Tensor>& ranks_depth,
                             const c10::optional<at::Tensor>& ranks_feat,
                             const at::Tensor& ranks_bev,
                             int64_t b, int64_t d, int64_t h, int64_t w, int64_t c) {
        auto out = ascend_kernel::bev_pool_v3(
            depth, feat, ranks_depth, ranks_feat, ranks_bev,
            b, d, h, w, c);
        return out.out;
    });
}
