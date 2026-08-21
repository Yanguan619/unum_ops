#include <cstring>
#include <cstdio>

#include "../op_kernel/voxelization_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static uint32_t RoundUp32(uint32_t v) { return (v + 31u) & ~31u; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    VoxelizationTilingData* tiling = context->GetTilingData<VoxelizationTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* in_shape = context->GetInputShape(0);
    const gert::Shape& s = in_shape->GetStorageShape();
    uint32_t numPoints = (s.GetDimNum() >= 2 && s.GetDim(0) > 0) ? (uint32_t)s.GetDim(0) : 0;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const auto* vs = attrs->GetListFloat(0);
    const auto* pcr = attrs->GetListFloat(1);
    const int64_t* maxNumPtsAttr = attrs->GetInt(2);
    const int64_t* maxVoxelsAttr = attrs->GetInt(3);
    if (vs == nullptr || pcr == nullptr || vs->GetSize() < 3 || pcr->GetSize() < 6) {
        return ge::GRAPH_FAILED;
    }
    const float* vsData = vs->GetData();
    const float* pcrData = pcr->GetData();

    // 与 numpy 参考 (VoxelGeneratorV2) 一致：fp32 下 ((range[3:]-range[:3])/voxel_size) 截断取整
    int32_t gridX = (int32_t)((pcrData[3] - pcrData[0]) / vsData[0]);
    int32_t gridY = (int32_t)((pcrData[4] - pcrData[1]) / vsData[1]);
    int32_t gridZ = (int32_t)((pcrData[5] - pcrData[2]) / vsData[2]);
    if (gridX <= 0 || gridY <= 0 || gridZ <= 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t gridTotal = (uint32_t)gridX * (uint32_t)gridY * (uint32_t)gridZ;
    uint32_t maxNumPoints = (uint32_t)((maxNumPtsAttr != nullptr) ? *maxNumPtsAttr : 32);
    uint32_t maxVoxels = (uint32_t)((maxVoxelsAttr != nullptr) ? *maxVoxelsAttr : 40000);

    // 核数按运行时实际 vector 核（310P 故障态只报 7 核），上限 VOXEL_MAX_CORES
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t coreNum = (platform != nullptr) ? platform->GetCoreNumAiv() : VOXEL_MAX_CORES;
    uint32_t blockNum = (coreNum < VOXEL_MAX_CORES) ? coreNum : VOXEL_MAX_CORES;

    uint32_t padN = RoundUp32(numPoints);
    // sync 保留区域（硬件 SyncAll 不使用，仅保持布局稳定）
    uint64_t syncBytes = (uint64_t)blockNum * 2 * 8 * sizeof(int32_t);
    uint64_t offLocalCnt = syncBytes;
    uint64_t offVid = offLocalCnt + (uint64_t)blockNum * gridTotal * sizeof(int32_t);
    uint64_t offPtLocalPos = offVid + (uint64_t)gridTotal * sizeof(int32_t);
    uint64_t offBlockSum = offPtLocalPos + (uint64_t)blockNum * padN * sizeof(int32_t);
    // coords/npts 暂存区：每 vid 一个 32B 槽位（8 int32），容量 padN（vid < M <= numPoints <= padN）
    uint64_t offScratch = offBlockSum + (uint64_t)blockNum * 8 * sizeof(int32_t);
    uint64_t workspaceSize = offScratch + (uint64_t)padN * 8 * sizeof(int32_t);

    memset(tiling, 0, sizeof(VoxelizationTilingData));
    tiling->numPoints = numPoints;
    tiling->padNumPoints = padN;
    tiling->blockNum = blockNum;
    tiling->gridX = (uint32_t)gridX;
    tiling->gridY = (uint32_t)gridY;
    tiling->gridZ = (uint32_t)gridZ;
    tiling->gridTotal = gridTotal;
    tiling->maxNumPoints = maxNumPoints;
    tiling->maxVoxels = maxVoxels;
    // binsPerCore 向上取整到 8（32B），保证每核 bin 区间不共享同一个 8-int32 DataCopy 块
    uint32_t binsPerCore = (gridTotal + blockNum - 1) / blockNum;
    binsPerCore = (binsPerCore + 7u) & ~7u;
    tiling->binsPerCore = binsPerCore;
    tiling->voxelSizeX = vsData[0];
    tiling->voxelSizeY = vsData[1];
    tiling->voxelSizeZ = vsData[2];
    tiling->pcrX0 = pcrData[0];
    tiling->pcrY0 = pcrData[1];
    tiling->pcrZ0 = pcrData[2];
    tiling->invVoxelSizeX = 1.0f / vsData[0];
    tiling->invVoxelSizeY = 1.0f / vsData[1];
    tiling->invVoxelSizeZ = 1.0f / vsData[2];
    tiling->offSync = 0;
    tiling->offLocalCnt = offLocalCnt;
    tiling->offVid = offVid;
    tiling->offPtLocalPos = offPtLocalPos;
    tiling->offBlockSum = offBlockSum;
    tiling->offScratch = offScratch;
    tiling->workspaceSize = workspaceSize;

    context->SetBlockDim(blockNum);
    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ws[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    {
        static bool first = true;
        if (first) {
            first = false;
            FILE* f = fopen("/tmp/infershape_debug.log", "w");
            if (f) {
                fprintf(f, "[DEBUG] InferShape called\n");
                fclose(f);
            }
        }
    }
    // 输出按 max_voxels 容量分配（M 为运行期数据相关值，由 num_voxels[0] 给出）
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        FILE* f = fopen("/tmp/infershape_debug.log", "a");
        if (f) { fprintf(f, "[DEBUG] InferShape: attrs is null\n"); fclose(f); }
        return ge::GRAPH_FAILED;
    }
    const int64_t* maxNumPtsAttr = attrs->GetInt(2);
    const int64_t* maxVoxelsAttr = attrs->GetInt(3);
    int64_t maxV = (maxVoxelsAttr != nullptr) ? *maxVoxelsAttr : 40000;
    int64_t maxP = (maxNumPtsAttr != nullptr) ? *maxNumPtsAttr : 32;
    {
        FILE* f = fopen("/tmp/infershape_debug.log", "a");
        if (f) { fprintf(f, "[DEBUG] InferShape: maxV=%ld, maxP=%ld, maxNumPtsAttr=%p, maxVoxelsAttr=%p\n", maxV, maxP, (void*)maxNumPtsAttr, (void*)maxVoxelsAttr); fclose(f); }
    }

    gert::Shape* voxels_shape = context->GetOutputShape(0);
    if (voxels_shape == nullptr) {
        FILE* f = fopen("/tmp/infershape_debug.log", "a");
        if (f) { fprintf(f, "[DEBUG] InferShape: voxels_shape is null\n"); fclose(f); }
        return ge::GRAPH_FAILED;
    }
    voxels_shape->SetDimNum(3);
    voxels_shape->SetDim(0, maxV);
    voxels_shape->SetDim(1, maxP);
    voxels_shape->SetDim(2, 4);

    gert::Shape* coords_shape = context->GetOutputShape(1);
    if (coords_shape == nullptr) {
        FILE* f = fopen("/tmp/infershape_debug.log", "a");
        if (f) { fprintf(f, "[DEBUG] InferShape: coords_shape is null\n"); fclose(f); }
        return ge::GRAPH_FAILED;
    }
    coords_shape->SetDimNum(2);
    coords_shape->SetDim(0, maxV);
    coords_shape->SetDim(1, 3);

    gert::Shape* num_points_shape = context->GetOutputShape(2);
    if (num_points_shape == nullptr) {
        FILE* f = fopen("/tmp/infershape_debug.log", "a");
        if (f) { fprintf(f, "[DEBUG] InferShape: num_points_shape is null\n"); fclose(f); }
        return ge::GRAPH_FAILED;
    }
    num_points_shape->SetDimNum(1);
    num_points_shape->SetDim(0, maxV);

    gert::Shape* num_voxels_shape = context->GetOutputShape(3);
    if (num_voxels_shape == nullptr) {
        FILE* f = fopen("/tmp/infershape_debug.log", "a");
        if (f) { fprintf(f, "[DEBUG] InferShape: num_voxels_shape is null\n"); fclose(f); }
        return ge::GRAPH_FAILED;
    }
    num_voxels_shape->SetDimNum(1);
    num_voxels_shape->SetDim(0, 1);
    {
        FILE* f = fopen("/tmp/infershape_debug.log", "a");
        if (f) { fprintf(f, "[DEBUG] InferShape: SUCCESS\n"); fclose(f); }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_INT32);
    context->SetOutputDataType(3, ge::DT_INT32);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Voxelization : public OpDef {
public:
    explicit Voxelization(const char* name) : OpDef(name)
    {
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("coords")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("num_points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("num_voxels")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("voxel_size").ListFloat();
        this->Attr("point_cloud_range").ListFloat();
        this->Attr("max_num_points").Int();
        this->Attr("max_voxels").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(Voxelization);

}  // namespace ops

IMPL_OP_INFERSHAPE(Voxelization)
    .InferShape(ge::InferShape)
    .InferDataType(ge::InferDataType);