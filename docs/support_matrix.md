# 310P 接口支持矩阵

> 由 `scripts/gen_support_matrix.py` 从 `unum_ops.INTERFACES_310P` 自动生成，请勿手改。

| 子包 | 接口 | 310P 可用 | 说明 |
|------|------|-----------|------|
| `bev_pool` | `bev_pool` | ✅ | AscendC 自定义算子，需先编译 .so |
| `bev_pool` | `bev_pool_torch` | ✅ | 纯 torch scatter_add 实现 |
| `bev_pool` | `BevPoolOutput` | ✅ | 纯 python 数据结构 |
| `bev_pool_v3` | `bev_pool_v3` | ✅ | AscendC 自定义算子，需先编译 .so |
| `bev_pool_v3` | `BevPoolV3Output` | ✅ | 纯 python 数据结构 |
| `infllm_v2` | `infllmv2_attn_stage1_ref_torch` | ✅ | 纯 torch 参考实现 |
| `infllm_v2` | `infllmv2_attn_stage1_triton` | ❌ | triton kernel，需 CUDA |
| `infllm_v2` | `infllmv2_attn_stage1_triton_v2` | ❌ | triton kernel，需 CUDA |
| `infllm_v2` | `infllmv2_attn_stage1` | ❌ | triton kernel 别名（= triton_v2），需 CUDA |
| `infllm_v2` | `max_pooling_1d_varlen_ref_triton` | ❌ | triton kernel，需 CUDA |
| `infllm_v2` | `max_pooling_1d_varlen` | ❌ | triton kernel 别名（= ref_triton），需 CUDA |
| `pointnet2` | `furthest_point_sample` | ✅ | 纯 torch 循环；CPU eager 走 numpy 就地快路径(~8x)，trace 时回退 torch；NPU 时 CPU fallback |
| `pointnet2` | `furthest_point_sample_onnx` | ✅ | 纯 torch 循环，ONNX 可导出（不包 autograd.Function） |
| `pointnet2` | `furthest_point_sample_torch` | ✅ | 纯 torch 循环 |
| `pointnet2` | `gather_operation` | ✅ | 纯 torch.gather |
| `pointnet2` | `three_nn` | ✅ | 纯 torch cdist+topk |
| `pointnet2` | `three_interpolate` | ✅ | 纯 torch，k=3 循环 |
| `pointnet2` | `three_interpolate_torch` | ✅ | 纯 torch，k=3 循环 |
| `pointnet2` | `grouping_operation` | ✅ | 纯 torch，nsample 循环 |
| `pointnet2` | `grouping_operation_torch` | ✅ | 纯 torch，nsample 循环 |
| `pointnet2` | `ball_query` | ✅ | 纯 torch cdist+where，NPU 时 CPU fallback |
| `pointnet2` | `cylinder_query` | ✅ | 纯 torch matmul+where，NPU 时 CPU fallback |
| `pointnet2` | `cylinder_query_torch` | ✅ | 纯 torch matmul+where |
| `pointnet2` | `three_interpolate_onnx` | ✅ | 纯 torch，ONNX 可导出 |
| `pointnet2` | `grouping_operation_onnx` | ✅ | 纯 torch，ONNX 可导出 |
| `pointnet2` | `cylinder_query_onnx` | ✅ | 纯 torch 全向量化，ONNX 可导出 |
| `pointnet2` | `QueryAndGroup` | ✅ | 纯 torch nn.Module |
| `pointnet2` | `GroupAll` | ✅ | 纯 torch nn.Module |
| `pointnet2` | `CylinderQueryAndGroup` | ✅ | 纯 torch nn.Module |
| `pointnet2` | `knn` | ✅ | 纯 torch cdist+topk |
| `sparse_kernel_extension` | `get_block_table_ref_torch` | ✅ | 纯 torch 参考实现 |
| `sparse_kernel_extension` | `get_block_table_ref_triton` | ❌ | triton kernel，需 CUDA |
| `sparse_kernel_extension` | `get_block_table_ref_triton_v2` | ❌ | triton kernel，需 CUDA |
| `sparse_kernel_extension` | `get_block_table_ref_triton_v3` | ❌ | triton kernel，需 CUDA |
| `sparse_kernel_extension` | `get_block_table_v2` | ❌ | triton kernel 别名（= ref_triton_v2），需 CUDA |
| `sparse_kernel_extension` | `get_block_table_v3` | ❌ | triton kernel 别名（= ref_triton_v3），需 CUDA |
| `spconv` | `SparseConvTensor` | ✅ | 纯 torch |
| `spconv` | `SparseModule` | ✅ | 纯 torch |
| `spconv` | `SparseSequential` | ✅ | 纯 torch |
| `spconv` | `SparseReLU` | ✅ | 纯 torch |
| `spconv` | `SparseBatchNorm1d` | ✅ | 纯 torch |
| `spconv` | `SparseLinear` | ✅ | 纯 torch |
| `spconv` | `SparseConv3dAdapter` | ✅ | 纯 torch |
| `spconv` | `SubMConv3dAdapter` | ✅ | 纯 torch |
| `spconv` | `SubMConv3d` | ✅ | 纯 torch |
| `spconv` | `SubMConv2d` | ✅ | 纯 torch |
| `spconv` | `SparseConv3d` | ✅ | 纯 torch |
| `spconv` | `SparseConv2d` | ✅ | 纯 torch |
| `spconv` | `SparseInverseConv3d` | ✅ | 纯 torch |
| `spconv` | `SparseInverseConv2d` | ✅ | 纯 torch |
| `spconv` | `SparseConv3dCPU` | ✅ | SparseConv3d 兼容旧名 |
| `spconv` | `SubMConv3dCPU` | ✅ | SubMConv3d 兼容旧名 |
| `spconv` | `VoxelGeneratorV2` | ✅ | 纯 numpy |
| `spconv` | `VoxelGenerator` | ✅ | 纯 numpy |
| `voxelization` | `voxelization` | ✅ | AscendC 自定义算子，需先编译 .so |
| `voxelization` | `VoxelizationOutput` | ✅ | 纯 python 数据结构 |
| `voxelization` | `voxelization_torch` | ✅ | 纯 torch 实现 |
| `voxelization` | `voxelization_torch_ref` | ✅ | 纯 torch 参考实现 |

共 58 个接口，310P 直接可用 48 个。
