#include <cstring>
#include "../op_kernel/spconv_gemm_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "adv_api/matmul/matmul_tiling.h"

namespace optiling {

using namespace matmul_tiling;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SpconvGemmTilingData* tiling = context->GetTilingData<SpconvGemmTilingData>();
    if (tiling == nullptr) return ge::GRAPH_FAILED;
    memset(tiling, 0, sizeof(SpconvGemmTilingData));

    const gert::StorageShape* featsShape = context->GetInputShape(0);
    const gert::Shape& fs = featsShape->GetStorageShape();
    if (fs.GetDimNum() != 2) return ge::GRAPH_FAILED;
    uint32_t M = (uint32_t)fs.GetDim(0);       // N（输出体素数）
    uint32_t K = (uint32_t)fs.GetDim(1);       // K_flat

    const gert::StorageShape* weightShape = context->GetInputShape(1);
    const gert::Shape& ws = weightShape->GetStorageShape();
    if (ws.GetDimNum() != 2 || ws.GetDim(0) != (int64_t)K) return ge::GRAPH_FAILED;
    uint32_t N = (uint32_t)ws.GetDim(1);       // C_out

    const gert::StorageShape* biasShape = context->GetInputShape(2);
    const gert::Shape& bs = biasShape->GetStorageShape();
    uint32_t biasSize = (bs.GetDimNum() >= 1) ? (uint32_t)bs.GetDim(0) : 0;
    uint32_t hasBias = (biasSize == N) ? 1 : 0;

    if (M == 0 || N == 0 || K == 0) return ge::GRAPH_FAILED;

    // 平台信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling matmulTiling(ascendcPlatform);

    // Cube 输入：A(fp16) B(fp16) C(fp32) Bias(fp32)
    matmulTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    matmulTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    matmulTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    matmulTiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    matmulTiling.SetBias(hasBias == 1);
    matmulTiling.SetShape((int32_t)M, (int32_t)N, (int32_t)K);
    matmulTiling.SetOrgShape((int32_t)M, (int32_t)N, (int32_t)K);
    matmulTiling.SetALayout(1, (int32_t)K, (int32_t)K, 1, 0);
    matmulTiling.SetBLayout(1, (int32_t)N, (int32_t)N, 1, 0);
    matmulTiling.SetCLayout(1, (int32_t)N, (int32_t)N, 1, 0);

    if (matmulTiling.GetTiling(tiling->mm) == -1) {
        return ge::GRAPH_FAILED;
    }

    tiling->M = M;
    tiling->N = N;
    tiling->K = K;
    tiling->hasBias = hasBias;

    // 核数：使用 AIC（Cube）核数，匹配 MIX_AIC_1_2 任务类型
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t blockNum = (platform != nullptr) ? platform->GetCoreNumAic() : 1;
    if (blockNum < 1) blockNum = 1;
    tiling->blockNum = blockNum;
    context->SetBlockDim(blockNum);

    // workspace 大小
    size_t* wsSizes = context->GetWorkspaceSizes(1);
    if (wsSizes == nullptr) return ge::GRAPH_FAILED;
    matmul_tiling::SysTilingTempBufSize bufSize;
    if (MatmulGetTmpBufSizeV2(tiling->mm, bufSize) == 0) {
        wsSizes[0] = (size_t)bufSize.l1Size + (size_t)bufSize.ubSize + (size_t)bufSize.l0cSize;
    } else {
        wsSizes[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* fs = context->GetInputShape(0);
    const gert::Shape* ws = context->GetInputShape(1);
    int64_t M = (fs->GetDimNum() >= 1) ? fs->GetDim(0) : 0;
    int64_t N = (ws->GetDimNum() >= 2) ? ws->GetDim(1) : 0;

    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) return ge::GRAPH_FAILED;
    outShape->SetDimNum(2);
    outShape->SetDim(0, M);
    outShape->SetDim(1, N);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class SpconvGemm : public OpDef {
public:
    explicit SpconvGemm(const char* name) : OpDef(name)
    {
        this->Input("feats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("params")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(SpconvGemm);

}  // namespace ops

IMPL_OP_INFERSHAPE(SpconvGemm)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);