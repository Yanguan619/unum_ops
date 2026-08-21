/* -------------------------------------------------------------------------
 * Voxelization tiling data: host/kernel 共享。
 * P0 优化版：新增 invVoxelSizeX/Y/Z（倒数预乘）+ offScratch（coords/npts 暂存区）。
 * 布局：10×uint32 + 9×float (4B 对齐) + 7×uint64 (8B 对齐从 76B 起，无 padding 空洞)。
 * ------------------------------------------------------------------------- */

#ifndef VOXELIZATION_TILING_H
#define VOXELIZATION_TILING_H
#include <cstdint>

constexpr uint32_t VOXEL_TILE_POINTS = 1024;
constexpr uint32_t VOXEL_MAX_CORES = 8;

struct VoxelizationTilingData {
    // --- 10×uint32 (40 bytes) ---
    uint32_t numPoints;
    uint32_t padNumPoints;
    uint32_t blockNum;
    uint32_t gridX;
    uint32_t gridY;
    uint32_t gridZ;
    uint32_t gridTotal;
    uint32_t maxNumPoints;
    uint32_t maxVoxels;
    uint32_t binsPerCore;
    // --- 9×float (36 bytes, total 76) ---
    float voxelSizeX;
    float voxelSizeY;
    float voxelSizeZ;
    float pcrX0;
    float pcrY0;
    float pcrZ0;
    float invVoxelSizeX;   // 1.0f / voxelSizeX — 预计算倒数，kernel 用 Mul 替代 Div
    float invVoxelSizeY;
    float invVoxelSizeZ;
    // --- 7×uint64 (56 bytes, total 132) ---
    uint64_t offSync;
    uint64_t offLocalCnt;
    uint64_t offVid;
    uint64_t offPtLocalPos;
    uint64_t offBlockSum;
    uint64_t offScratch;   // coords/npts 暂存区（8×padN×int32，每 vid 一个 32B 槽位），ScatterPoints 写入、WriteCoordsNpts 读出
    uint64_t workspaceSize;
};

#endif // VOXELIZATION_TILING_H