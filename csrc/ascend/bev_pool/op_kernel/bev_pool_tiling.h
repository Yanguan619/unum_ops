/* -------------------------------------------------------------------------
 * BevPool tiling data: host/kernel 共享。
 * 算子：稀疏点到稠密 BEV 网格的 segment-sum scatter（LSS Lift-Splat 的 Splat 阶段）。
 *
 * 输入：feats [N, C] float32（未排序，通过 sort_indices 间接读取）
 *       coords [N, 4] int32（未排序，通过 sort_indices 间接读取）
 *       sort_indices [N] int64（argsort 结果）
 *       interval_starts [K] int32（每个 voxel interval 的起始点下标）
 *       interval_lengths [K] int32（每个 interval 的点数）
 * 输出：out [B, D, H, W, C] float32（稠密 BEV 体素特征，未 permute）
 * ------------------------------------------------------------------------- */

#ifndef BEV_POOL_TILING_H
#define BEV_POOL_TILING_H
#include <cstdint>

constexpr uint32_t BEV_TILE_POINTS = 32;
constexpr uint32_t BEV_MAX_CHANNELS = 512;
constexpr uint32_t BEV_MAX_CORES = 8;

struct BevPoolTilingData {
    // --- 12×uint32 (48 bytes) ---
    uint32_t numPoints;        // N
    uint32_t numChannels;      // C
    uint32_t numIntervals;     // K
    uint32_t gridB;            // B
    uint32_t gridD;            // D
    uint32_t gridH;            // H
    uint32_t gridW;            // W
    uint32_t gridTotal;        // B*D*H*W
    uint32_t intervalsPerCore; // ceil(K / blockNum)
    uint32_t blockNum;         // 实际核数
    uint32_t channelAligned;   // 1 表示 C % 8 == 0（向量化路径），0 表示标量路径
    uint32_t pad;              // 填充对齐
};

static_assert(sizeof(BevPoolTilingData) == 48, "BevPoolTilingData must be 48 bytes");

#endif // BEV_POOL_TILING_H