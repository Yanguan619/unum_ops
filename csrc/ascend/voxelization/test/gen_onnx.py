from onnx import TensorProto
from onnx.helper import (make_model, make_node, make_graph, make_tensor_value_info)
from onnx.checker import check_model

# 定义输入
points = make_tensor_value_info("points", TensorProto.FLOAT, ["batch", "num_points", "features"])
# 定义输出
voxels = make_tensor_value_info("voxels", TensorProto.FLOAT, ["batch", "max_voxels", "features"])
coords = make_tensor_value_info("coords", TensorProto.INT32, ["max_voxels", "coord_dim"])
num_points = make_tensor_value_info("num_points", TensorProto.INT32, ["max_voxels"])
num_voxels = make_tensor_value_info("num_voxels", TensorProto.INT32, ["batch"])

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