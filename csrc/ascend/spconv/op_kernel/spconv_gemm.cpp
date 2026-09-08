#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "spconv_gemm_tiling.h"

using namespace AscendC;

class KernelSpconvGemmCube {
public:
    __aicore__ inline KernelSpconvGemmCube(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR feats, GM_ADDR weight, GM_ADDR bias, GM_ADDR params,
                                GM_ADDR out, GM_ADDR workspace, const SpconvGemmTilingData* tiling)
    {
        __gm__ int32_t* p = (__gm__ int32_t*)((__gm__ uint8_t*)params);
        M_ = (uint32_t)p[0];
        N_ = (uint32_t)p[1];
        K_ = (uint32_t)p[2];
        hasBias_ = (uint32_t)p[3];

        gmFeats_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(feats), (int64_t)M_ * K_);
        gmWeight_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight), (int64_t)K_ * N_);
        gmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out), (int64_t)M_ * N_);
        if (hasBias_) {
            gmBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias), (int64_t)N_);
        }

        using AType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
        using BType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
        using CType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
        using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;

        mm_.Init(&tiling->mm, pipe_);
    }

    __aicore__ inline void Process()
    {
        mm_.SetTensorA(gmFeats_, false);
        mm_.SetTensorB(gmWeight_, false);
        if (hasBias_) {
            mm_.SetBias(gmBias_);
        } else {
            mm_.DisableBias();
        }
        mm_.SetTail(-1, -1, -1);
        mm_.IterateAll(gmOut_);
        mm_.End();
    }

private:
    TPipe* pipe_;
    uint32_t M_, N_, K_, hasBias_;
    GlobalTensor<half> gmFeats_;
    GlobalTensor<half> gmWeight_;
    GlobalTensor<float> gmOut_;
    GlobalTensor<float> gmBias_;

    using AType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    Matmul<AType, BType, CType, BiasType, CFG_MDL> mm_;
};

extern "C" __global__ __aicore__ void spconv_gemm(GM_ADDR feats, GM_ADDR weight,
                                                    GM_ADDR bias, GM_ADDR params,
                                                    GM_ADDR out, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(SpconvGemmTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelSpconvGemmCube op(&pipe);
    op.Init(feats, weight, bias, params, out, workspace, &tilingData);
    op.Process();
}