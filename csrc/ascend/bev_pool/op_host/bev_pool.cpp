#include <cstring>
#include "../op_kernel/bev_pool_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    BevPoolTilingData* tiling = context->GetTilingData<BevPoolTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* featsShape = context->GetInputShape(0);
    const gert::Shape& fs = featsShape->GetStorageShape();
    uint32_t numPoints = (fs.GetDimNum() >= 2 && fs.GetDim(0) > 0) ? (uint32_t)fs.GetDim(0) : 0;
    uint32_t numChannels = (fs.GetDimNum() >= 2) ? (uint32_t)fs.GetDim(1) : 0;

    const gert::StorageShape* startsShape = context->GetInputShape(2);
    const gert::Shape& ss = startsShape->GetStorageShape();
    uint32_t numIntervals = (ss.GetDimNum() >= 1 && ss.GetDim(0) > 0) ? (uint32_t)ss.GetDim(0) : 0;

    if (numChannels > BEV_MAX_CHANNELS) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t* bAttr = attrs->GetInt(0);
    const int64_t* dAttr = attrs->GetInt(1);
    const int64_t* hAttr = attrs->GetInt(2);
    const int64_t* wAttr = attrs->GetInt(3);
    uint32_t gridB = (bAttr != nullptr) ? (uint32_t)*bAttr : 1;
    uint32_t gridD = (dAttr != nullptr) ? (uint32_t)*dAttr : 1;
    uint32_t gridH = (hAttr != nullptr) ? (uint32_t)*hAttr : 1;
    uint32_t gridW = (wAttr != nullptr) ? (uint32_t)*wAttr : 1;
    uint64_t gridTotal64 = (uint64_t)gridB * (uint64_t)gridD * (uint64_t)gridH * (uint64_t)gridW;
    if (gridTotal64 > UINT32_MAX || gridTotal64 == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t gridTotal = (uint32_t)gridTotal64;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = (platform != nullptr) ? platform->GetCoreNumAiv() : BEV_MAX_CORES;
    uint32_t blockNum = (coreNum < BEV_MAX_CORES) ? coreNum : BEV_MAX_CORES;
    bool channelAligned = (numChannels % 8 == 0);

    uint32_t intervalsPerCore = (numIntervals + blockNum - 1) / blockNum;

    // 块处理路径的块容量（行数）：按实际 C 自适应，尽量用满 UB chunk 预算。
    // chunk 预算 ~ (128KB) / 4 字节 = 32768 float。
    uint32_t tilePoints = (numChannels > 0) ? (32768u / numChannels) : 1;
    tilePoints = (tilePoints < BEV_MAX_TILE_POINTS) ? tilePoints : BEV_MAX_TILE_POINTS;
    if (tilePoints < 1) { tilePoints = 1; }
    if (!channelAligned) { tilePoints = BEV_TILE_POINTS; }

    memset(tiling, 0, sizeof(BevPoolTilingData));
    tiling->numPoints = numPoints;
    tiling->numChannels = numChannels;
    tiling->numIntervals = numIntervals;
    tiling->gridB = gridB;
    tiling->gridD = gridD;
    tiling->gridH = gridH;
    tiling->gridW = gridW;
    tiling->gridTotal = gridTotal;
    tiling->intervalsPerCore = intervalsPerCore;
    tiling->blockNum = blockNum;
    tiling->channelAligned = channelAligned ? 1 : 0;
    tiling->tilePoints = tilePoints;

    context->SetBlockDim(blockNum);
    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ws[0] = 0;  // 不需要 workspace
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t* bAttr = attrs->GetInt(0);
    const int64_t* dAttr = attrs->GetInt(1);
    const int64_t* hAttr = attrs->GetInt(2);
    const int64_t* wAttr = attrs->GetInt(3);
    int64_t gridB = (bAttr != nullptr) ? *bAttr : 1;
    int64_t gridD = (dAttr != nullptr) ? *dAttr : 1;
    int64_t gridH = (hAttr != nullptr) ? *hAttr : 1;
    int64_t gridW = (wAttr != nullptr) ? *wAttr : 1;

    const gert::Shape* fs = context->GetInputShape(0);
    int64_t numChannels = (fs->GetDimNum() >= 2) ? fs->GetDim(1) : 1;

    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 输出实际是 2D [gridTotal, C]（Python 层 view+permute 回 [B,D,H,W,C]），
    // 与 aclnn/register 返回的 tensor 形状保持一致，避免形状验证不匹配。
    outShape->SetDimNum(2);
    outShape->SetDim(0, gridB * gridD * gridH * gridW);
    outShape->SetDim(1, numChannels);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class BevPool : public OpDef {
public:
    explicit BevPool(const char* name) : OpDef(name)
    {
        this->Input("feats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("coords")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("interval_starts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("interval_lengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("batch").Int();
        this->Attr("depth").Int();
        this->Attr("height").Int();
        this->Attr("width").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(BevPool);

}  // namespace ops

IMPL_OP_INFERSHAPE(BevPool)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);