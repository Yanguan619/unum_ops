#include <torch/extension.h>
#include "bev_pool_ops.h"

TORCH_LIBRARY_FRAGMENT(unum, m) {
    m.def("bev_pool(Tensor feats, Tensor coords, Tensor interval_starts, "
          "Tensor interval_lengths, int batch, int depth, int height, "
          "int width) -> Tensor");
}

TORCH_LIBRARY_IMPL(unum, PrivateUse1, m) {
    m.impl("bev_pool", [](const at::Tensor& feats,
                          const at::Tensor& coords,
                          const at::Tensor& interval_starts,
                          const at::Tensor& interval_lengths,
                          int64_t batch, int64_t depth,
                          int64_t height, int64_t width) {
        auto out = ascend_kernel::bev_pool(feats, coords, interval_starts,
                                           interval_lengths, batch, depth,
                                           height, width);
        return out.out;
    });
}