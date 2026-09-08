/* -------------------------------------------------------------------------
 * SpconvGemm tiling data: host/kernel 共享。
 *
 * 算子：稀疏卷积的特征聚合（gather+GEMM）阶段 — Cube 加速版。
 *   out[n, o] = sum_k feats[n, k] * weight[k, o] + bias[o]
 *
 * 输入（torch 侧已按邻居表 gather 并展平为 2D）：
 *   feats   [N, K]  float16  (n, k) 布局，k = prod(kernel_size) * C_in
 *   weight  [K, C]  float16  (k, o) 布局
 *   bias    [C]     float32
 * 输出：
 *   out     [N, C]  float32
 *
 * 多核按输出行连续分区，各核只写自己分区的输出行。
 * Cube 计算通过 AscendC::Matmul 高层 API 完成。
 * ------------------------------------------------------------------------- */

#ifndef SPCONV_GEMM_TILING_H
#define SPCONV_GEMM_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

constexpr uint32_t SPCONV_MAX_CORES = 8;

struct SpconvGemmTilingData {
    AscendC::tiling::TCubeTiling mm;  // Matmul API tiling（~400B）
    uint32_t M;        // N（输出体素数）
    uint32_t N;        // C_out（输出通道数）
    uint32_t K;        // K_flat = prod(kernel_size) * C_in
    uint32_t hasBias;  // 1 = 加 bias
    uint32_t blockNum; // 实际核数
};

#endif // SPCONV_GEMM_TILING_H