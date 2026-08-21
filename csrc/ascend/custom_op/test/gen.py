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

X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None, None])
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])
nodel = make_node("LeakyRelu", ["X"], ["Y"], alpha=0.1)

graph = make_graph([nodel], 'leakyrelu', [X], [Y])

onnx_model = make_model(graph)

del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.version = 11

check_model(onnx_model)

with open('leaky_relu.onnx', "wb") as f:
    f.write(onnx_model.SerializeToString())
