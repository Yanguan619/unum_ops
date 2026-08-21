#include "kernel_operator.h"
#include "voxelization_tiling.h"

/* Voxelization kernel — workspace 数据放在 voxels 输出 tensor 末尾。
 * 310P aclnn 框架的 workspace GM 不被 MTE2 引擎映射（DDR 地址越界），
 * 但输出 tensor 的 GM 可被 MTE2/MTE3 正常访问。
 *
 * voxels 输出布局: (maxVoxels=40000, maxNumPoints=32, 4) float32 = 20MB
 * 实际 voxel 数据从 offset 0 开始，最多使用 M*32*4 floats（M~3941）。
 * workspace 数据放在末尾：wsBase = voxels[maxVoxels*maxNumPoints*4 - wsFloats]
 *
 * workspace 布局（从 wsBase 起，int32 为单位）：
 *   sync[blockNum * 2 * 8]（保留空间，硬件 SyncAll 不使用）
 *   localCnt[core_ * gridTotal .. core_ * gridTotal + gridTotal]
 *   vid[0 .. gridTotal]
 *   ptLocalPos[0 .. padN]
 *   blockSum[0 .. blockNum]
 *   scratch: coords/npts 暂存，每 vid 一个 8 int32 槽位（容量 padN），
 *            ScatterPoints 用 MTE3 写，WriteCoordsNpts 由 core 0 单写者读出到输出
 */

/* 核间同步：使用硬件 SyncAll<true>（ffts 跨核 flag + wait_flag_dev），
 * 不依赖软件轮询 GM 标志（MTE2/MTE3 跨核可见性在 310P7 上不可靠，实测会挂起）。 */

__aicore__ inline uint32_t VoxMinU(uint32_t a, uint32_t b) { return a < b ? a : b; }

class KernelVoxelization {
public:
    __aicore__ inline KernelVoxelization(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR points, GM_ADDR voxels, GM_ADDR coords,
                                GM_ADDR numPointsOut, GM_ADDR numVoxelsOut,
                                const VoxelizationTilingData* tiling)
    {
        t_ = tiling;
        core_ = AscendC::GetBlockIdx();
        startBin_ = core_ * tiling->binsPerCore;
        endBin_ = VoxMinU(startBin_ + tiling->binsPerCore, tiling->gridTotal);

        // aclnn 框架对 tensor GM_ADDR 加了 8 字节 header
        points = (__gm__ uint8_t*)points - 8;
        voxels = (__gm__ uint8_t*)voxels - 8;
        coords = (__gm__ uint8_t*)coords - 8;
        numPointsOut = (__gm__ uint8_t*)numPointsOut - 8;
        numVoxelsOut = (__gm__ uint8_t*)numVoxelsOut - 8;

        // 在 voxels 输出 tensor 末尾划分 workspace 区域
        uint32_t maxVox = tiling->maxVoxels;
        uint32_t maxPts = tiling->maxNumPoints;
        // wsFloats = offScratch/4 + 8*padN（scratch 每 vid 8 int32）
        uint32_t wsFloats = (uint32_t)(tiling->offScratch / sizeof(int32_t))
                          + 8u * tiling->padNumPoints;
        // wsBase（float 为单位）= voxels 末尾减去 wsFloats
        uint32_t voxTotalFloats = maxVox * maxPts * 4;
        __gm__ int32_t* wsBase = (__gm__ int32_t*)voxels + (voxTotalFloats - wsFloats);

        // workspace 指针（全部指向 voxels 输出 tensor 内）
        localCntPtr_ = wsBase + (tiling->offLocalCnt / sizeof(int32_t))
                     + core_ * tiling->gridTotal;
        vidPtr_ = wsBase + (tiling->offVid / sizeof(int32_t));
        ptLocalPosPtr_ = wsBase + (tiling->offPtLocalPos / sizeof(int32_t))
                       + core_ * tiling->padNumPoints;  // 每核独立 ptLocalPos 区域
        blockSumPtr_ = wsBase + (tiling->offBlockSum / sizeof(int32_t));
        scrPtr_ = wsBase + (tiling->offScratch / sizeof(int32_t));
        // sync 区域（offSync 起）为保留空间，硬件 SyncAll 不使用
        blockSumGm_.SetGlobalBuffer(blockSumPtr_, tiling->blockNum * 8);  // 每 8 int32 对齐

        localCntGm_.SetGlobalBuffer(localCntPtr_, tiling->gridTotal);
        ptLocalPosGm_.SetGlobalBuffer(ptLocalPosPtr_, tiling->padNumPoints);
        vidGm_.SetGlobalBuffer(vidPtr_, tiling->gridTotal);
        // coords/npts 暂存区：每 vid 8 int32（32B 槽位），MTE3 写 / MTE2 读
        scrGm_.SetGlobalBuffer(scrPtr_, (int64_t)tiling->padNumPoints * 8);

        pointsGm_.SetGlobalBuffer((__gm__ float*)points, (int64_t)tiling->numPoints * 4);
        pointsPtr_ = (__gm__ float*)points;
        voxPtr_ = (__gm__ float*)voxels;
        coordPtr_ = (__gm__ int32_t*)coords;
        numPtPtr_ = (__gm__ int32_t*)numPointsOut;
        numVoxelsPtr_ = (__gm__ int32_t*)numVoxelsOut;

        // UB buffer
        pipe_->InitBuffer(bufRaw_, VOXEL_TILE_POINTS * 4 * sizeof(float));
        pipe_->InitBuffer(bufX_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufY_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufZ_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufInt_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufT_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufW_, (VOXEL_TILE_POINTS + 8) * sizeof(int32_t));
        pipe_->InitBuffer(bufXF_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufYF_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufZF_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufXI_, VOXEL_TILE_POINTS * sizeof(int32_t));
        pipe_->InitBuffer(bufYI_, VOXEL_TILE_POINTS * sizeof(int32_t));
        pipe_->InitBuffer(bufZI_, VOXEL_TILE_POINTS * sizeof(int32_t));
        pipe_->InitBuffer(bufBin_, VOXEL_TILE_POINTS * sizeof(int32_t));
        pipe_->InitBuffer(bufDivX_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufDivY_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufDivZ_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufZero_, VOXEL_TILE_POINTS * sizeof(float));
        pipe_->InitBuffer(bufOffset_, VOXEL_TILE_POINTS * sizeof(uint32_t));

        InitOffsetTable();
        LoadDivisors();
    }

    __aicore__ inline void Process()
    {
        // 各阶段核间数据依赖分析：
        //   InitWorkspace/CountPoints/BlockCount   —— 只读写本核 bin 区间，无跨核依赖
        //   SyncAll                                —— blockSum 归约同步点（硬件跨核同步）
        //   AssignVoxelIds/ScatterPoints           —— 只读写本核区间 + 读全部 blockSum
        //   SyncAll                                —— scatter 写入 scratch 的可见性同步点
        //   WriteCoordsNpts                        —— 仅 core 0：单写者写 coords/npts 输出（避免多核并发写同一缓存行丢写）
        InitWorkspace();
        CountPoints();
        BlockCount();
        AscendC::SyncAll<true>();
        AssignVoxelIds();
        ScatterPoints();
        AscendC::SyncAll<true>();
        WriteCoordsNpts();
    }

    private:
    __aicore__ inline void InitOffsetTable()
    {
        AscendC::LocalTensor<uint32_t> off = bufOffset_.Get<uint32_t>();
        for (uint32_t i = 0; i < VOXEL_TILE_POINTS; i++) {
            off.SetValue(i, i * 16u);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadDivisors()
    {
        AscendC::LocalTensor<float> dx = bufDivX_.Get<float>();
        AscendC::LocalTensor<float> dy = bufDivY_.Get<float>();
        AscendC::LocalTensor<float> dz = bufDivZ_.Get<float>();
        AscendC::Duplicate<float>(dx, t_->voxelSizeX, (int32_t)VOXEL_TILE_POINTS);
        AscendC::Duplicate<float>(dy, t_->voxelSizeY, (int32_t)VOXEL_TILE_POINTS);
        AscendC::Duplicate<float>(dz, t_->voxelSizeZ, (int32_t)VOXEL_TILE_POINTS);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitWorkspace()
    {
        // DataCopy 批量清零 localCnt（MTE3 写 output tensor GM 可见）
        AscendC::LocalTensor<int32_t> zeroI = bufZero_.Get<int32_t>();
        AscendC::Duplicate<int32_t>(zeroI, 0, (int32_t)VOXEL_TILE_POINTS);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t b = startBin_; b < endBin_;) {
            uint32_t cnt = VoxMinU(VOXEL_TILE_POINTS, endBin_ - b);
            // DataCopy 要求 32B（8 int32）对齐：尾块向上取整，越界部分属 workspace，无害
            uint32_t cntA = (cnt + 7u) & ~7u;
            AscendC::DataCopy(localCntGm_[b], zeroI, (int32_t)cntA);
            b += cnt;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadTile(uint32_t off, uint32_t cnt)
    {
        AscendC::LocalTensor<float> raw = bufRaw_.Get<float>();
        AscendC::LocalTensor<float> xb = bufX_.Get<float>();
        AscendC::LocalTensor<float> yb = bufY_.Get<float>();
        AscendC::LocalTensor<float> zb = bufZ_.Get<float>();
        AscendC::LocalTensor<float> ib = bufInt_.Get<float>();
        AscendC::LocalTensor<uint32_t> offTbl = bufOffset_.Get<uint32_t>();
        if (cnt == VOXEL_TILE_POINTS) {
            AscendC::DataCopy(raw, pointsGm_[off * 4], (int32_t)(4 * VOXEL_TILE_POINTS));
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            for (uint32_t i = 0; i < cnt; i++) {
                uint32_t idx = (off + i) * 4;
                raw.SetValue(i * 4,     pointsPtr_[idx]);
                raw.SetValue(i * 4 + 1, pointsPtr_[idx + 1]);
                raw.SetValue(i * 4 + 2, pointsPtr_[idx + 2]);
                raw.SetValue(i * 4 + 3, pointsPtr_[idx + 3]);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::Gather<float>(xb, raw, offTbl, 0u, cnt);
        AscendC::Gather<float>(yb, raw, offTbl, 4u, cnt);
        AscendC::Gather<float>(zb, raw, offTbl, 8u, cnt);
        AscendC::Gather<float>(ib, raw, offTbl, 12u, cnt);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeTile(uint32_t cnt)
    {
        AscendC::LocalTensor<float> xt = bufX_.Get<float>();
        AscendC::LocalTensor<float> yt = bufY_.Get<float>();
        AscendC::LocalTensor<float> zt = bufZ_.Get<float>();
        AscendC::LocalTensor<float> xf = bufXF_.Get<float>();
        AscendC::LocalTensor<float> yf = bufYF_.Get<float>();
        AscendC::LocalTensor<float> zf = bufZF_.Get<float>();
        AscendC::LocalTensor<float> tmp = bufT_.Get<float>();
        AscendC::LocalTensor<int32_t> xi = bufXI_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> yi = bufYI_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> zi = bufZI_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> bin = bufBin_.Get<int32_t>();
        AscendC::LocalTensor<float> dx = bufDivX_.Get<float>();
        AscendC::LocalTensor<float> dy = bufDivY_.Get<float>();
        AscendC::LocalTensor<float> dz = bufDivZ_.Get<float>();
        AscendC::Adds(xf, xt, -t_->pcrX0, cnt);
        AscendC::Div(tmp, xf, dx, cnt);
        AscendC::Cast(xi, tmp, AscendC::RoundMode::CAST_FLOOR, cnt);

        AscendC::Adds(yf, yt, -t_->pcrY0, cnt);
        AscendC::Div(tmp, yf, dy, cnt);
        AscendC::Cast(yi, tmp, AscendC::RoundMode::CAST_FLOOR, cnt);

        AscendC::Adds(zf, zt, -t_->pcrZ0, cnt);
        AscendC::Div(tmp, zf, dz, cnt);
        AscendC::Cast(zi, tmp, AscendC::RoundMode::CAST_FLOOR, cnt);
        uint32_t gy = t_->gridY;
        uint32_t gz = t_->gridZ;
        AscendC::LocalTensor<int32_t> w = bufW_.Get<int32_t>();
        AscendC::Muls(bin, xi, (int32_t)(gy * gz), cnt);
        AscendC::Muls(w, yi, (int32_t)gz, cnt);
        AscendC::Add(bin, bin, w, cnt);
        AscendC::Add(bin, bin, zi, cnt);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool IsPointValid(uint32_t off, uint32_t i, uint32_t n,
                                        int32_t xv, int32_t yv, int32_t zv)
    {
        if (off + i >= n) return false;
        if (xv < 0 || xv >= (int32_t)t_->gridX) return false;
        if (yv < 0 || yv >= (int32_t)t_->gridY) return false;
        if (zv < 0 || zv >= (int32_t)t_->gridZ) return false;
        return true;
    }

    __aicore__ inline void CountPoints()
    {
        // Point-centric: per-point DataCopy 读-改-写 localCnt（8-int32 块）
        uint32_t n = t_->numPoints;
        AscendC::LocalTensor<int32_t> bin = bufBin_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> xi = bufXI_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> yi = bufYI_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> zi = bufZI_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> cntBlk = bufT_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> posBlk = bufW_.Get<int32_t>();

        for (uint32_t off = 0; off < n; off += VOXEL_TILE_POINTS) {
            uint32_t cnt = VoxMinU(VOXEL_TILE_POINTS, n - off);
            LoadTile(off, cnt);
            ComputeTile(cnt);

            AscendC::Duplicate<int32_t>(posBlk, 0, (int32_t)VOXEL_TILE_POINTS);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t i = 0; i < cnt; i++) {
                int32_t b = bin.GetValue(i);
                int32_t xv = xi.GetValue(i);
                int32_t yv = yi.GetValue(i);
                int32_t zv = zi.GetValue(i);
                if (!IsPointValid(off, i, n, xv, yv, zv)) continue;
                if (b >= (int32_t)startBin_ && b < (int32_t)endBin_) {
                    uint32_t blkBase = (uint32_t)b & ~7u;
                    AscendC::DataCopy(cntBlk, localCntGm_[blkBase], 8);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    uint32_t idx = (uint32_t)b - blkBase;
                    int32_t v = cntBlk.GetValue(idx);
                    cntBlk.SetValue(idx, v + 1);
                    posBlk.SetValue(i, v);
                    AscendC::DataCopy(localCntGm_[blkBase], cntBlk, 8);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
            uint32_t alignedCnt = (cnt + 7u) & ~7u;
            AscendC::DataCopy(ptLocalPosGm_[off], posBlk, alignedCnt);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void BlockCount()
    {
        uint32_t blockCnt = 0;
        AscendC::LocalTensor<int32_t> cntBlk = bufT_.Get<int32_t>();
        for (uint32_t b = startBin_; b < endBin_; b += 8) {
            AscendC::DataCopy(cntBlk, localCntGm_[b], 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t j = 0; j < 8 && b + j < endBin_; j++) {
                if (cntBlk.GetValue(j) > 0) blockCnt++;
            }
        }
        // DataCopy 写 blockSum（跨核需要 MTE3 可见）
        AscendC::LocalTensor<int32_t> bsUb = bufW_.Get<int32_t>();
        bsUb.SetValue(0, (int32_t)blockCnt);
        AscendC::DataCopy(blockSumGm_[core_ * 8], bsUb, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void AssignVoxelIds()
    {
        // DataCopy 读全部 blockSum（跨核 MTE2 可见）
        AscendC::LocalTensor<int32_t> bsBlk = bufT_.Get<int32_t>();
        uint32_t bsFloats = ((t_->blockNum * 8) + 7u) & ~7u;
        AscendC::DataCopy(bsBlk, blockSumGm_, bsFloats);
        AscendC::PipeBarrier<PIPE_ALL>();
        uint32_t blockOff = 0;
        uint32_t total = 0;
        for (uint32_t c = 0; c < t_->blockNum; c++) {
            uint32_t s = (uint32_t)bsBlk.GetValue(c * 8);
            if (c < core_) blockOff += s;
            total += s;
        }
        if (core_ == 0) {
            numVoxelsPtr_[0] = (int32_t)(VoxMinU(total, t_->maxVoxels));
        }
        AscendC::LocalTensor<int32_t> cntBlk = bufXF_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> vidBlk = bufW_.Get<int32_t>();
        uint32_t localRank = 0;
        for (uint32_t b = startBin_; b < endBin_; b += 8) {
            AscendC::DataCopy(cntBlk, localCntGm_[b], 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t j = 0; j < 8 && b + j < endBin_; j++) {
                if (cntBlk.GetValue(j) > 0) {
                    uint32_t vid = blockOff + localRank;
                    localRank++;
                    vidBlk.SetValue(j, (vid < t_->maxVoxels) ? (int32_t)vid : -1);
                } else {
                    vidBlk.SetValue(j, -1);
                }
            }
            AscendC::DataCopy(vidGm_[b], vidBlk, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ScatterPoints()
    {
        uint32_t n = t_->numPoints;
        uint32_t maxPts = t_->maxNumPoints;
        AscendC::LocalTensor<float> xt = bufX_.Get<float>();
        AscendC::LocalTensor<float> yt = bufY_.Get<float>();
        AscendC::LocalTensor<float> zt = bufZ_.Get<float>();
        AscendC::LocalTensor<float> it = bufInt_.Get<float>();
        AscendC::LocalTensor<int32_t> xi = bufXI_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> yi = bufYI_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> zi = bufZI_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> bin = bufBin_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> vidBlk = bufT_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> posBlk = bufW_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> cntBlk = bufXF_.Get<int32_t>();

        for (uint32_t off = 0; off < n; off += VOXEL_TILE_POINTS) {
            uint32_t cnt = VoxMinU(VOXEL_TILE_POINTS, n - off);
            uint32_t alignedCnt = (cnt + 7u) & ~7u;
            LoadTile(off, cnt);
            ComputeTile(cnt);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(posBlk, ptLocalPosGm_[off], alignedCnt);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t i = 0; i < cnt; i++) {
                int32_t b = bin.GetValue(i);
                int32_t xv = xi.GetValue(i);
                int32_t yv = yi.GetValue(i);
                int32_t zv = zi.GetValue(i);
                if (!IsPointValid(off, i, n, xv, yv, zv)) continue;
                if (b < (int32_t)startBin_ || b >= (int32_t)endBin_) continue;
                uint32_t blkBase = (uint32_t)b & ~7u;
                AscendC::DataCopy(vidBlk, vidGm_[blkBase], 8);
                AscendC::PipeBarrier<PIPE_ALL>();
                int32_t vid = vidBlk.GetValue((uint32_t)b - blkBase);
                if (vid < 0) continue;
                int32_t pos = posBlk.GetValue(i);
                if (pos < (int32_t)maxPts) {
                    float xvF = xt.GetValue(i);
                    float yvF = yt.GetValue(i);
                    float zvF = zt.GetValue(i);
                    float iv = it.GetValue(i);
                    __gm__ float* vp = voxPtr_ + ((uint64_t)vid * maxPts + pos) * 4;
                    vp[0] = xvF;
                    vp[1] = yvF;
                    vp[2] = zvF;
                    vp[3] = iv;
                    if (pos == 0) {
                        // 读 localCnt 得到 cntV，打包到 scratch 槽位（MTE3 写 32B 单槽）。
                        // 不再直接标量写 coords/npts 输出：多核并发写同一 32B 缓存行会丢写。
                        AscendC::DataCopy(cntBlk, localCntGm_[blkBase], 8);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        int32_t cntV = cntBlk.GetValue((uint32_t)b - blkBase);
                        cntV = (cntV < (int32_t)maxPts) ? cntV : (int32_t)maxPts;
                        // 槽位 [vid, zv, yv, xv, cntV, 0, 0, 0]，vid 唯一 → 无跨核写竞争
                        cntBlk.SetValue(0, vid);
                        cntBlk.SetValue(1, zv);
                        cntBlk.SetValue(2, yv);
                        cntBlk.SetValue(3, xv);
                        cntBlk.SetValue(4, cntV);
                        cntBlk.SetValue(5, 0);
                        cntBlk.SetValue(6, 0);
                        cntBlk.SetValue(7, 0);
                        AscendC::DataCopy(scrGm_[(int64_t)vid * 8], cntBlk, 8);
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // 仅 core 0：单写者最终写 coords/npts 输出。
    // 读取各核在 ScatterPoints 写入 scratch 的槽位（MTE2），校验槽内 vid 后写输出。
    // 单核标量写输出无跨核竞争，不会丢写（与原始单核版本一致）。
    __aicore__ inline void WriteCoordsNpts()
    {
        if (core_ != 0) return;
        AscendC::LocalTensor<int32_t> blk = bufT_.Get<int32_t>();
        // MTE2 读 blockSum 重建 M（WriteCoordsNpts 是独立方法，标量读/标量写不可靠）
        uint32_t bsFloats = ((t_->blockNum * 8) + 7u) & ~7u;
        AscendC::DataCopy(blk, blockSumGm_, bsFloats);
        AscendC::PipeBarrier<PIPE_ALL>();
        uint32_t m = 0;
        for (uint32_t c = 0; c < t_->blockNum; c++) m += (uint32_t)blk.GetValue(c * 8);
        m = VoxMinU(m, t_->maxVoxels);
        // 每 8 个槽位（8×32B=256B）一批读入，减少同步开销
        for (uint32_t v = 0; v < m; v += 8) {
            uint32_t cnt8 = VoxMinU(8, m - v);
            AscendC::DataCopy(blk, scrGm_[(int64_t)v * 8], (int32_t)(cnt8 * 8));
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t j = 0; j < cnt8; j++) {
                int32_t vid = blk.GetValue((int32_t)j * 8);
                if (vid != (int32_t)(v + j)) continue;  // 槽位未写（安全兜底）
                int32_t zv = blk.GetValue((int32_t)j * 8 + 1);
                int32_t yv = blk.GetValue((int32_t)j * 8 + 2);
                int32_t xv = blk.GetValue((int32_t)j * 8 + 3);
                int32_t cntV = blk.GetValue((int32_t)j * 8 + 4);
                __gm__ int32_t* cp = coordPtr_ + (uint64_t)(v + j) * 3;
                cp[0] = zv;
                cp[1] = yv;
                cp[2] = xv;
                numPtPtr_[v + j] = cntV;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::TPipe* pipe_;
    const VoxelizationTilingData* t_;
    uint32_t core_;
    uint32_t startBin_;
    uint32_t endBin_;
    __gm__ int32_t* localCntPtr_;
    __gm__ int32_t* vidPtr_;
    __gm__ int32_t* ptLocalPosPtr_;
    __gm__ int32_t* blockSumPtr_;
    __gm__ int32_t* numVoxelsPtr_;
    __gm__ int32_t* scrPtr_;
    AscendC::GlobalTensor<int32_t> scrGm_;
    AscendC::GlobalTensor<int32_t> blockSumGm_;
    AscendC::GlobalTensor<float> pointsGm_;
    __gm__ float* pointsPtr_;
    AscendC::GlobalTensor<int32_t> localCntGm_;
    AscendC::GlobalTensor<int32_t> ptLocalPosGm_;
    AscendC::GlobalTensor<int32_t> vidGm_;
    __gm__ float* voxPtr_;
    __gm__ int32_t* coordPtr_;
    __gm__ int32_t* numPtPtr_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRaw_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufX_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufZ_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufInt_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufT_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufW_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufXF_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufYF_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufZF_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufXI_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufYI_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufZI_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufBin_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufDivX_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufDivY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufDivZ_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufZero_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOffset_;
};

extern "C" __global__ __aicore__ void voxelization(
    GM_ADDR points, GM_ADDR voxels, GM_ADDR coords, GM_ADDR num_points,
    GM_ADDR num_voxels, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(VoxelizationTilingData);
    GET_TILING_DATA(tilingData, tiling);

    AscendC::TPipe pipe;
    KernelVoxelization op(&pipe);
    op.Init(points, voxels, coords, num_points, num_voxels, &tilingData);
    op.Process();
}
