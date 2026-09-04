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

        pipe_->InitBuffer(bufAcc_, BEV_MAX_CHANNELS * sizeof(float));
        pipe_->InitBuffer(bufChunk_, BEV_TILE_POINTS * BEV_MAX_CHANNELS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t iv = startInt_; iv < endInt_; iv++) {
            ProcessInterval(iv);
        }
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

    __aicore__ inline void ProcessInterval(uint32_t iv)
    {
        int32_t start = startsPtr_[iv];
        int32_t length = lengthsPtr_[iv];
        if (length <= 0) {
            return;
        }
        if (!InBounds((uint32_t)start)) {
            return;
        }

        uint64_t outOff = OutputOffset((uint32_t)start);
        uint32_t C = t_->numChannels;
        AscendC::LocalTensor<float> acc = bufAcc_.Get<float>();

        if (t_->channelAligned) {
            AscendC::LocalTensor<float> chunk = bufChunk_.Get<float>();
            AscendC::Duplicate<float>(acc, 0.0f, (int32_t)C);
            AscendC::PipeBarrier<PIPE_ALL>();

            int32_t rowBase = start;
            int32_t rowsLeft = length;
            while (rowsLeft > 0) {
                uint32_t t = BevMinU(BEV_TILE_POINTS, (uint32_t)rowsLeft);
                AscendC::DataCopy(chunk, featsGm_[(int64_t)rowBase * C], (int32_t)(t * C));
                AscendC::PipeBarrier<PIPE_ALL>();
                for (uint32_t i = 0; i < t; i++) {
                    AscendC::Add(acc, acc, chunk[i * C], (int32_t)C);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                rowBase += (int32_t)t;
                rowsLeft -= (int32_t)t;
            }
            AscendC::DataCopy(outGm_[outOff], acc, (int32_t)C);
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            // 标量路径：C 非 8 对齐时的正确性回退。
            // 直接在 GM 上累加，避免 VECCAL 位置 UB 的 GetValue/SetValue 在 310P 上非确定。
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufAcc_;
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