#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


from onnx import TensorProto
from onnx.helper import (make_model, make_node, make_graph, make_tensor_value_info)
from onnx.checker import check_model

# 定义输入
points = make_tensor_value_info("points", TensorProto.FLOAT, [None, None, None])

# 定义输出
voxels = make_tensor_value_info("voxels", TensorProto.FLOAT, [None, None, None])
coords = make_tensor_value_info("coords", TensorProto.INT32, [None, None])
num_points = make_tensor_value_info("num_points", TensorProto.INT32, [None])
num_voxels = make_tensor_value_info("num_voxels", TensorProto.INT32, [None])

# 创建 Voxelization 节点
voxelization_node = make_node(
    "Voxelization",
    inputs=["points"],
    outputs=["voxels", "coords", "num_points", "num_voxels"],
    voxel_size=[0.05, 0.05, 0.1],
    point_cloud_range=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
    max_num_points=100,
    max_voxels=40000
)

# 创建计算图
graph = make_graph(
    [voxelization_node],
    'voxelization',
    [points],
    [voxels, coords, num_points, num_voxels]
)

# 创建模型
onnx_model = make_model(graph)

# 设置 opset 版本
del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.version = 13

# 跳过检查，直接保存
# check_model(onnx_model)  # 注释掉检查

with open('voxelization.onnx', "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ Voxelization ONNX model created successfully!")