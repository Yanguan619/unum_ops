#include "kernel_operator.h"
#include "bev_pool_tiling.h"

/* BevPool kernel — LSS Splat 阶段的 segment-sum scatter。
 * 输入（已按 rank 排序）：feats [N, C]，coords [N, 4]=(x,y,z,batch)，
 * interval_starts [K]，interval_lengths [K]。
 * 输出：out [B, D, H, W, C]。
 *
 * 每个 interval 对应一个唯一 voxel，其所有点映射到同一输出位置。
 * 多核按 interval 分区，无跨核写冲突。
 */

__aicore__ inline uint32_t BevMinU(uint32_t a, uint32_t b) { return a < b ? a : b; }

class KernelBevPool {
public:
    __aicore__ inline KernelBevPool(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR feats, GM_ADDR coords, GM_ADDR starts,
                                GM_ADDR lengths, GM_ADDR out,
                                const BevPoolTilingData* tiling)
    {
        t_ = tiling;
        core_ = AscendC::GetBlockIdx();
        startInt_ = core_ * tiling->intervalsPerCore;
        endInt_ = BevMinU(startInt_ + tiling->intervalsPerCore, tiling->numIntervals);

        featsPtr_ = (__gm__ float*)((__gm__ uint8_t*)feats);
        coordPtr_ = (__gm__ int32_t*)((__gm__ uint8_t*)coords);
        startsPtr_ = (__gm__ int32_t*)((__gm__ uint8_t*)starts);
        lengthsPtr_ = (__gm__ int32_t*)((__gm__ uint8_t*)lengths);
        outPtr_ = (__gm__ float*)((__gm__ uint8_t*)out);

        featsGm_.SetGlobalBuffer(featsPtr_, (int64_t)tiling->numPoints * tiling->numChannels);
        outGm_.SetGlobalBuffer(outPtr_, (int64_t)tiling->gridTotal * tiling->numChannels);
        startsGm_.SetGlobalBuffer(startsPtr_, (int64_t)tiling->numIntervals);
        lengthsGm_.SetGlobalBuffer(lengthsPtr_, (int64_t)tiling->numIntervals);
        coordsGm_.SetGlobalBuffer(coordPtr_, (int64_t)tiling->numPoints * 4);

        const uint32_t C = t_->numChannels;
        pipe_->InitBuffer(bufAcc_, C * sizeof(float));
        pipe_->InitBuffer(bufAcc1_, C * sizeof(float));
        pipe_->InitBuffer(bufStarts_, BEV_TILE_POINTS * sizeof(int32_t));
        pipe_->InitBuffer(bufLengths_, BEV_TILE_POINTS * sizeof(int32_t));
        pipe_->InitBuffer(bufCoords_, BEV_MAX_TILE_POINTS * 4 * sizeof(int32_t));
        pipe_->InitBuffer(bufChunk_, (int64_t)t_->tilePoints * ((int64_t)t_->numChannels) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (!t_->channelAligned) {
            for (uint32_t iv = startInt_; iv < endInt_; iv++) {
                ProcessIntervalScalar(iv);
            }
            return;
        }
        uint32_t iv = startInt_;
        while (iv < endInt_) {
            iv = ProcessBlock(iv);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline uint64_t OutputOffset(uint32_t start) const
    {
        int32_t bx = coordPtr_[(int64_t)start * 4 + 0];
        int32_t by = coordPtr_[(int64_t)start * 4 + 1];
        int32_t bz = coordPtr_[(int64_t)start * 4 + 2];
        int32_t batch = coordPtr_[(int64_t)start * 4 + 3];
        uint64_t off = ((uint64_t)batch * t_->gridD + (uint64_t)bz) * t_->gridH + (uint64_t)by;
        off = off * t_->gridW + (uint64_t)bx;
        return off * t_->numChannels;
    }

    __aicore__ inline bool InBounds(uint32_t start) const
    {
        int32_t bx = coordPtr_[(int64_t)start * 4 + 0];
        int32_t by = coordPtr_[(int64_t)start * 4 + 1];
        int32_t bz = coordPtr_[(int64_t)start * 4 + 2];
        int32_t batch = coordPtr_[(int64_t)start * 4 + 3];
        if (bx < 0 || bx >= (int32_t)t_->gridW) return false;
        if (by < 0 || by >= (int32_t)t_->gridH) return false;
        if (bz < 0 || bz >= (int32_t)t_->gridD) return false;
        if (batch < 0 || batch >= (int32_t)t_->gridB) return false;
        return true;
    }

    // 标量路径：C 非 8 对齐时的正确性回退，单核。
    __aicore__ inline void ProcessIntervalScalar(uint32_t iv)
    {
        int32_t start = startsPtr_[iv];
        int32_t length = lengthsPtr_[iv];
        if (length <= 0) {
            return;
        }
        if (!InBounds((uint32_t)start)) {
            return;
        }
        if ((uint32_t)(start + length) > t_->numPoints) {
            return;
        }
        uint64_t outOff = OutputOffset((uint32_t)start);
        uint32_t C = t_->numChannels;
        __gm__ float* dst = outPtr_ + outOff;
        for (uint32_t c = 0; c < C; c++) {
            dst[c] = 0.0f;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t r = 0; r < length; r++) {
            const __gm__ float* row = featsPtr_ + (int64_t)(start + r) * C;
            for (uint32_t c = 0; c < C; c++) {
                dst[c] += row[c];
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 块处理（对齐向量化路径）：
    // 排序后的 feats 中，连续 interval 的点行在 GM 上连续 → 一个块可用一次
    // 大的 DataCopy 装载多行，避免每 interval 一次小的 GM 读（延迟受限主因）。
    // OOB interval 的行也占据 GM 空间，必须计入 rowOff 以保持 DataCopy 的连续性。
    // 返回下一个待处理的 interval 下标。
    __aicore__ inline uint32_t ProcessBlock(uint32_t iv0)
    {
        const uint32_t C = t_->numChannels;
        const uint32_t cap = t_->tilePoints;
        AscendC::LocalTensor<float> chunk = bufChunk_.Get<float>();
        AscendC::LocalTensor<float> acc = bufAcc_.Get<float>();
        AscendC::LocalTensor<float> acc1 = bufAcc1_.Get<float>();
        AscendC::LocalTensor<int32_t> startsUb = bufStarts_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> lengthsUb = bufLengths_.Get<int32_t>();

        // ---- Load phase: 累积连续 interval 的点行直到块容量 ----
        // 所有 ln > 0 的 interval（包括 OOB）都计入 rowOff，
        // 因为它们的行在 GM 中连续排列。
        // 批量 DataCopy 读 starts/lengths 到 UB，避免逐 interval GM 标量读（scalar 瓶颈）
        uint32_t iv = iv0;
        uint32_t avail = BevMinU(BEV_TILE_POINTS, endInt_ - iv0);
        AscendC::DataCopy(startsUb, startsGm_[iv0], (int32_t)avail);
        AscendC::DataCopy(lengthsUb, lengthsGm_[iv0], (int32_t)avail);
        AscendC::PipeBarrier<PIPE_MTE2>();
        uint32_t availIdx = 0;
        int64_t rowOff = 0;
        int64_t firstStart = -1;
        while (iv < endInt_ && availIdx < avail) {
            int32_t s = startsUb.GetValue(availIdx);
            int32_t ln = lengthsUb.GetValue(availIdx);
            if (ln <= 0 || (uint32_t)(s + ln) > t_->numPoints) {
                iv++;
                availIdx++;
                continue;
            }
            if (firstStart < 0) {
                firstStart = s;
            }
            if ((int64_t)ln > (int64_t)cap - rowOff) {
                break;
            }
            rowOff += ln;
            iv++;
            availIdx++;
        }
        if (rowOff == 0) {
            // 首 interval 就超容量（或全为空）→ 单独处理这个大 interval。
            return ProcessLargeInterval(iv0);
        }

        // 单次大 DataCopy 装载整个块的多行（连续 GM 区域，含 OOB 行）。
        AscendC::DataCopy(chunk, featsGm_[(int64_t)firstStart * C], (int32_t)(rowOff * C));
        AscendC::PipeBarrier<PIPE_MTE2>();

        // 批量读 coords 到 UB（compute phase 用 UB 替代 GM 标量读，消除 InBounds/OutputOffset 的标量瓶颈）
        AscendC::LocalTensor<int32_t> coordsUb = bufCoords_.Get<int32_t>();
        AscendC::DataCopy(coordsUb, coordsGm_[(int64_t)firstStart * 4], (int32_t)(rowOff * 4));
        AscendC::PipeBarrier<PIPE_MTE2>();

        // ---- Compute phase: 逐 interval 在 UB 内累加并写出 ----
        // 双缓冲：两个 acc 交替。当前 interval 的 Duplicate+Add(V pipe) 与
        // 上一 interval 的 DataCopy(out)(MTE3 pipe) 重叠，隐藏 MTE3 写延迟。
        int64_t coff = 0;
        uint32_t bufIdx = 0;
        bool pendingWrite = false;
        for (uint32_t j = iv0; j < iv; j++) {
            int32_t s = startsUb.GetValue(j - iv0);
            int32_t ln = lengthsUb.GetValue(j - iv0);
            if (ln <= 0 || (uint32_t)(s + ln) > t_->numPoints) {
                continue;
            }
            // UB coords：x,y,z,batch 分别在第 0/1/2/3 列，偏移 (s - firstStart)*4
            int32_t bx = coordsUb.GetValue((int64_t)(s - firstStart) * 4 + 0);
            int32_t by = coordsUb.GetValue((int64_t)(s - firstStart) * 4 + 1);
            int32_t bz = coordsUb.GetValue((int64_t)(s - firstStart) * 4 + 2);
            int32_t batch = coordsUb.GetValue((int64_t)(s - firstStart) * 4 + 3);
            if (bx < 0 || bx >= (int32_t)t_->gridW) continue;
            if (by < 0 || by >= (int32_t)t_->gridH) continue;
            if (bz < 0 || bz >= (int32_t)t_->gridD) continue;
            if (batch < 0 || batch >= (int32_t)t_->gridB) continue;
            {
                uint64_t outOff = ((uint64_t)batch * t_->gridD + (uint64_t)bz) * t_->gridH + (uint64_t)by;
                outOff = (outOff * t_->gridW + (uint64_t)bx) * t_->numChannels;
                AscendC::LocalTensor<float> cur = (bufIdx == 0) ? acc : acc1;
                AscendC::Duplicate<float>(cur, 0.0f, (int32_t)C);
                for (int32_t r = 0; r < ln; r++) {
                    AscendC::Add(cur, cur, chunk[(int64_t)(coff + r) * C], (int32_t)C);
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (pendingWrite) {
                    AscendC::PipeBarrier<PIPE_MTE3>();
                }
                AscendC::DataCopy(outGm_[outOff], cur, (int32_t)C);
                pendingWrite = true;
                bufIdx ^= 1;
            }
            coff += ln;
        }
        if (pendingWrite) {
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        return iv;
    }

    // 单个超大 interval（长度超过块容量）：分片装载累加。
    __aicore__ inline uint32_t ProcessLargeInterval(uint32_t iv)
    {
        const uint32_t C = t_->numChannels;
        const uint32_t cap = t_->tilePoints;
        int32_t start = startsPtr_[iv];
        int32_t length = lengthsPtr_[iv];
        if (length <= 0 || !InBounds((uint32_t)start)) {
            return iv + 1;
        }
        if ((uint32_t)(start + length) > t_->numPoints) {
            return iv + 1;
        }
        uint64_t outOff = OutputOffset((uint32_t)start);
        AscendC::LocalTensor<float> chunk = bufChunk_.Get<float>();
        AscendC::LocalTensor<float> acc = bufAcc_.Get<float>();
        AscendC::Duplicate<float>(acc, 0.0f, (int32_t)C);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t rowBase = start;
        int32_t rowsLeft = length;
        while (rowsLeft > 0) {
            uint32_t tt = BevMinU(cap, (uint32_t)rowsLeft);
            AscendC::DataCopy(chunk, featsGm_[(int64_t)rowBase * C], (int32_t)(tt * C));
            AscendC::PipeBarrier<PIPE_MTE2>();
            for (uint32_t i = 0; i < tt; i++) {
                AscendC::Add(acc, acc, chunk[i * C], (int32_t)C);
            }
            AscendC::PipeBarrier<PIPE_V>();
            rowBase += (int32_t)tt;
            rowsLeft -= (int32_t)tt;
        }
        AscendC::DataCopy(outGm_[outOff], acc, (int32_t)C);
        AscendC::PipeBarrier<PIPE_MTE3>();
        return iv + 1;
    }

private:
    AscendC::TPipe* pipe_;
    const BevPoolTilingData* t_;
    uint32_t core_;
    uint32_t startInt_;
    uint32_t endInt_;
    __gm__ float* featsPtr_;
    __gm__ int32_t* coordPtr_;
    __gm__ int32_t* startsPtr_;
    __gm__ int32_t* lengthsPtr_;
    __gm__ float* outPtr_;
    AscendC::GlobalTensor<float> featsGm_;
    AscendC::GlobalTensor<float> outGm_;
    AscendC::GlobalTensor<int32_t> startsGm_;
    AscendC::GlobalTensor<int32_t> lengthsGm_;
    AscendC::GlobalTensor<int32_t> coordsGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufAcc_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufAcc1_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufStarts_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufLengths_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufCoords_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufChunk_;
};

extern "C" __global__ __aicore__ void bev_pool(GM_ADDR feats, GM_ADDR coords,
                                               GM_ADDR interval_starts,
                                               GM_ADDR interval_lengths,
                                               GM_ADDR out, GM_ADDR workspace,
                                               GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BevPoolTilingData);
    GET_TILING_DATA(tilingData, tiling);

    AscendC::TPipe pipe;
    KernelBevPool op(&pipe);
    op.Init(feats, coords, interval_starts, interval_lengths, out, &tilingData);
    op.Process();
}
