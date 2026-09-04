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
    uint32_t gridTotal = gridB * gridD * gridH * gridW;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = (platform != nullptr) ? platform->GetCoreNumAiv() : BEV_MAX_CORES;
    uint32_t blockNum = (coreNum < BEV_MAX_CORES) ? coreNum : BEV_MAX_CORES;
    bool channelAligned = (numChannels % 8 == 0);
    // 标量路径（C 非 8 倍数）用单核：310P 标量 GM 写操作在多核间无缓存一致性
    if (!channelAligned) {
        blockNum = 1;
    }

    uint32_t intervalsPerCore = (numIntervals + blockNum - 1) / blockNum;

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
    outShape->SetDimNum(5);
    outShape->SetDim(0, gridB);
    outShape->SetDim(1, gridD);
    outShape->SetDim(2, gridH);
    outShape->SetDim(3, gridW);
    outShape->SetDim(4, numChannels);

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