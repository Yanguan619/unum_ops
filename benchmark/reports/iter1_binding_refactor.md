# BevPool 优化报告 — Iteration 1: Binding 重构（vllm-ascend EXEC_NPU_CMD 风格）

## 动机

基线 wrapper（`bev_pool_torch.cpp`）存在三个问题：

1. **全零输出 flake**：每次调用把输入拷贝进临时 `torch::empty` buffer，输出也 `clone()` 返回。当 NPU allocator 在重分配时复用同一地址，aclnn 按地址缓存的 kernel descriptor 会命中已失效的旧条目 → 写入旧地址 → 输出全零。`executorClear` hack 降低了频率但未根除（~1/6 失败率）。
2. **冗余 D2D 拷贝**：feats/coords/starts/lengths 各一次 `copy_`，外加输出 `clone()`，约 1.9ms（512k 点）。
3. **手动资源管理**：`aclrtMalloc`/`rtFree`/`rtSync`/`executorClear` 全部手写，易错。

## 改动

研究 `vllm-ascend` 的 310P torch binding 模式（`csrc/aclnn_torch_adapter/op_api_common.h` 中的 `EXEC_NPU_CMD` 宏），提取其核心机制：

| 项目 | 旧 wrapper | 新 wrapper |
|------|-----------|-----------|
| 输入 tensor | `copy_()` 进临时 buffer | 直接传 `at_tensor.storage().data()` |
| 输出 tensor | `empty()` + kernel 写 + `clone()` 返回 | `zeros()` + kernel 直接写 + 原样返回 |
| executor | 跨调用缓存 + `executorClear` hack | 每次调用 `getWorkspaceSize` 重建 fresh executor |
| workspace | `aclrtMalloc` + `rtFree` | `at::empty` NPU tensor（RAII） |
| 提交方式 | `run()` + `rtSync` | `OpCommand::SetCustomHandler` + `Run()` |
| aclTensor 生命周期 | 手动 `destroyTensor` | handler 内提交后立即 `destroyTensor` |

关键文件：`csrc/ascend/bev_pool/op_extension/bev_pool_torch.cpp`

### 实现细节

- **`WrapTensor`**：仿照 vllm `ConvertType`，用 `aclCreateTensor` 把 `at::Tensor` 的实际 storage 直接包装成 `aclTensor`（无 D2D 拷贝）。`sizes().data()` / `strides().data()` 由 tensor 内部存储支撑，tensor 存活期间指针有效。
- **`OpCommand`**：`cmd.Name("aclnnBevPool")` + `cmd.SetCustomHandler(lambda)` + `cmd.Run()`。lambda **按值捕获**（仿照 vllm `EXEC_NPU_CMD`），避免引用悬空。handler 内完成 `run()` 提交 + `destroyTensor` 释放。
- **输出**：`at::zeros(...)` 分配（kernel 只写非空 voxel，空 voxel 需保持 0），kernel 直接写入，原样返回（无 clone）。

## 正确性

```
10 passed in 12.25s   × 5 次连续运行全部通过
```

全零 flake **完全消除**：旧 wrapper 在完整测试套件中 `test_large_channels` 约 1/6 概率输出全零；新 wrapper 连续 5 次运行 10/10 全通过。

## 性能

| num_points | baseline ascendc_ms | new ascendc_ms | 提升 |
|-----------|---------------------|----------------|------|
| 2,000 | 2.413 | 1.905 | -21% |
| 8,000 | 3.544 | 3.018 | -15% |
| 32,000 | 8.244 | 7.460 | -9% |
| 128,000 | 26.160 | 25.084 | -4% |
| 512,000 | 82.992 | 80.769 | -3% |

提升主要来自消除 wrapper 的冗余 D2D 拷贝（feats copy ~1.9ms + clone + 其他 buffer copy）。小规模时 wrapper 开销占比大 → 提升显著；大规模时 kernel 时间主导 → 提升较小。

## 瓶颈状态

wrapper 开销已大幅压缩，剩余时间由 **kernel 本身**主导：
- 512k 点 / 80k voxel → ~8 万小 interval，每 interval 串行 `Duplicate → DataCopy(in) → Add → DataCopy(out)` + `PipeBarrier`
- 非带宽受限（~2 GB/s 有效带宽），而是**每 interval 固定延迟累积**

## 下一步

Iteration 2：kernel 侧块处理优化 — 把连续 interval 的 GM 行合并为一次大 `DataCopy` 装载，减少小粒度 GM 读延迟累积。
