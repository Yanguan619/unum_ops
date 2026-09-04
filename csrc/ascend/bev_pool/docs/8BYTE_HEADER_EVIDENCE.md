# BevPool "8 字节偏移" 的根因修正记录

> 日期：2026-09-03　设备：Ascend310P7　CANN: 9.0.0

## 结论（修正版）

**之前以为 aclnn 框架会给 tensor GM_ADDR 加 8 字节 header，这是错误的推断。**

真正根因是 `op_extension/bev_pool_torch.cpp` 里调用 `aclCreateTensor` 时**参数错位**：
第 5 个参数 `offset`（int64）被误传为 `ACL_FORMAT_ND`（枚举值 = 2），导致框架把数据指针偏移了
`offset × itemsize = 2 × 4 = 8 字节`，kernel 收到的指针因此偏移 8 字节。

## 根因分析

CANN 9.0.0 `acl_meta.h` 中 `aclCreateTensor` 的真实签名：

```c
aclTensor *aclCreateTensor(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
                           const int64_t *stride, int64_t offset,    // ← 第5参是 offset
                           aclFormat format,                          // ← 第6参才是 format
                           const int64_t *storageDims, uint64_t storageDimsNum,
                           void *tensorData);
```

错误调用（之前 bev_pool / voxelization 都是这样）：

```cpp
aclCreateTensor(shape.data(), ndim, dataType, strides.data(),
                ACL_FORMAT_ND,        // ← 被当作 offset 传入！ACL_FORMAT_ND = 2
                ACL_FORMAT_ND,        // ← 被当作 format（碰巧对了）
                shape.data(), ndim, ptr);
```

`ACL_FORMAT_ND = 2`（见 `acl_base_rt.h:175`），所以 offset=2，数据指针偏移 2×itemsize 字节。
对 float32（itemsize=4）恰好是 8 字节 —— 这就是"8 字节偏移"的真相。

vllm-ascend 的 `ConvertType`（`csrc/aclnn_torch_adapter/op_api_common.h:429`）正确传参：

```cpp
auto acl_tensor = aclCreateTensor(
    at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type,
    at_tensor.strides().data(), at_tensor.storage_offset(),  // ← offset 传 storage_offset
    format,
    storageDims.data(), storageDims.size(),
    const_cast<void *>(at_tensor.storage().data()));          // ← 传 storage().data()
```

所以 vllm-ascend 的 310P 算子 kernel 不需要任何指针偏移。

## 修复内容

1. `op_extension/bev_pool_torch.cpp`：`MakeTensor` 改为显式传 `storage_offset`（contiguous 时为 0）
   + 用 `storage().data()` 作为 storageData。
2. `op_kernel/bev_pool.cpp`：移除所有 `- 8` 指针偏移。
3. `test/aclnn_test.cpp`：同样修正 `aclCreateTensor` 的 offset 参数为 0。

## 验证

- `test/aclnn_test.cpp`：修复后 `maxDiff=0`，PASS。
- kernel 不再做任何 `-8` 偏移，与 vllm-ascend 的 310P 算子实现一致。

## 遗留

`voxelization` 的 op_extension 也有同样的 `aclCreateTensor` 参数错位 bug（传给 offset 的是
`ACL_FORMAT_ND`），因此它的 kernel 也做了 `-8` 补偿。两者是"互相抵消的错误"。建议后续
一并修正 voxelization（kernel 去掉 -8 + extension 正确传 offset），保持与 vllm-ascend 一致。
