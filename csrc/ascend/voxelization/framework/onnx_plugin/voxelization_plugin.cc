/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/operator.h"
#include "register/register.h"
#include <nlohmann/json.hpp>
#include <cstdio>

using namespace ge;
using json = nlohmann::json;

namespace domi {
namespace {
const int kTypeFloat = 1;
const int kTypeInt = 2;
const int kTypeFloats = 6;
const int kTypeInts = 8;
}

Status ParseOnnxParamsVoxelization(const ge::Operator& op_src, ge::Operator& op_dest) {
  fprintf(stderr, "[DEBUG] ParseOnnxParamsVoxelization called\n");
  std::vector<float> voxel_size;
  std::vector<float> point_cloud_range;
  int64_t max_num_points = 100;
  int64_t max_voxels = 40000;

  bool parsed = false;
  AscendString attrs_string;
  if (ge::GRAPH_SUCCESS == op_src.GetAttr("attribute", attrs_string)) {
    fprintf(stderr, "[DEBUG] Got attribute JSON string: %s\n", attrs_string.GetString());
    json attrs = json::parse(attrs_string.GetString());
    for (json attr : attrs["attribute"]) {
      if (attr["name"] == "voxel_size" && attr["type"] == kTypeFloats) {
        voxel_size = attr["floats"].get<std::vector<float>>();
      } else if (attr["name"] == "point_cloud_range" && attr["type"] == kTypeFloats) {
        point_cloud_range = attr["floats"].get<std::vector<float>>();
      } else if (attr["name"] == "max_num_points" && attr["type"] == kTypeInt) {
        max_num_points = attr["i"].get<int64_t>();
      } else if (attr["name"] == "max_voxels" && attr["type"] == kTypeInt) {
        max_voxels = attr["i"].get<int64_t>();
      }
    }
    parsed = (!voxel_size.empty() && !point_cloud_range.empty());
  } else {
    fprintf(stderr, "[DEBUG] GetAttr attribute failed\n");
  }

  if (!parsed) {
    fprintf(stderr, "[DEBUG] Trying direct GetAttr\n");
    op_src.GetAttr("voxel_size", voxel_size);
    op_src.GetAttr("point_cloud_range", point_cloud_range);
    op_src.GetAttr("max_num_points", max_num_points);
    op_src.GetAttr("max_voxels", max_voxels);
    fprintf(stderr, "[DEBUG] Direct: voxel_size=%zu, point_cloud_range=%zu\n", voxel_size.size(), point_cloud_range.size());
    parsed = (voxel_size.size() >= 3 && point_cloud_range.size() >= 6);
  }

  if (!parsed) {
    fprintf(stderr, "[DEBUG] Parse failed, returning FAILED\n");
    return FAILED;
  }

  fprintf(stderr, "[DEBUG] Parse succeeded, setting attrs\n");
  op_dest.SetAttr("voxel_size", voxel_size);
  op_dest.SetAttr("point_cloud_range", point_cloud_range);
  op_dest.SetAttr("max_num_points", max_num_points);
  op_dest.SetAttr("max_voxels", max_voxels);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Voxelization")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("custom::1::Voxelization"),
                   ge::AscendString("ai.onnx::13::Voxelization"),
                   ge::AscendString("ai.onnx::11::Voxelization"),
                   ge::AscendString("Voxelization")})
    .ParseParamsByOperatorFn(ParseOnnxParamsVoxelization)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
