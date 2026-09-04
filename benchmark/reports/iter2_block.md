# BevPool 优化报告 — Iteration 2: Kernel 块处理 + Binding 重构 + Rank 修复

## 改动概述

本轮包含三项改动：

1. **Binding 重构**（Iteration 1）：仿照 vllm-ascend `EXEC_NPU_CMD` 模式，直接传用户 tensor storage（无 D2D 拷贝），每次调用 fresh executor（无缓存），通过 `OpCommand::SetCustomHandler` + `Run()` 提交。消除了全零输出 flake 和 ~1.9ms wrapper 开销。详见 `iter1_binding_refactor.md`。

2. **Kernel 块处理**（Iteration 2）：对齐路径（C % 8 == 0）从逐 interval 串行改为块处理 —— 一次大 `DataCopy` 装载连续多 interval 的 GM 行，减少小粒度 GM 读延迟累积。

3. **Rank 计算修复**：Python wrapper 的 rank 公式 `x*(W*D*B) + y*(D*B) + z*B + batch` 在 W < H 时产生碰撞（不同 voxel 被归入同一 interval）。修正为 `x + y*W + z*W*H + batch*W*H*D`（与参考实现的 flat index 一致）。

## Kernel 块处理改动

文件：`csrc/ascend/bev_pool/op_kernel/bev_pool.cpp`

### ProcessBlock 逻辑

- **Load phase**：累积连续 interval 的点行直到块容量（`tilePoints`）。所有 `ln > 0` 的 interval（包括 OOB）都计入 `rowOff`，因为它们的行在排序后的 feats GM 中连续排列。
- **DataCopy**：一次大 `DataCopy(chunk, featsGm_[firstStart*C], rowOff*C)` 装载整个块的多行（连续 GM 区域，含 OOB 行）。
- **Compute phase**：逐 interval 在 UB 内用 `Duplicate` → `Add` → `DataCopy` 累加并写出。`coff` 对所有 `ln > 0` 的 interval 前进（包括 OOB），保持与 chunk 内行偏移对齐。仅 in-bounds interval 执行累加 + 写出。

### 关键修复

1. **OOB 连续性**：旧代码在 load phase 跳过 OOB interval（`continue` 不计入 `rowOff`），导致 `DataCopy` 的 GM 跨度不覆盖 OOB 行 → 后续 in-bounds interval 的行偏移错误。新代码将 OOB interval 的 length 计入 `rowOff`，保持 `DataCopy` 的 GM 连续性。

2. **PIPE_ALL 屏障**：310P 上定向管道屏障（`PIPE_V`/`PIPE_MTE2`/`PIPE_MTE3`）不足以保证跨管道数据可见性。多核场景下，V pipe 的 `Add` 结果对 MTE3 pipe 的 `DataCopy` 不可见 → 输出部分为零。全部改用 `PIPE_ALL` 屏障后正确。

3. **Tiling 修复**：`static_assert` 从错误的 52 修正为 48（12×uint32）。`Init` 中 `t->` typo 修正为 `t_->`。`tilePoints` 按 `min(32768/C, 256)` 自适应。

## 正确性

```
10 passed in 12.22s   × 5 次连续运行全部通过
30/30 随机参数压力测试全部通过（W<H 场景覆盖）
```

- 单核 ProcessBlock：max_diff=0.0 ✓（验证块处理逻辑正确）
- 多核 ProcessBlock + PIPE_ALL：10/10 全通过 ✓
- 全零 flake：连续 5 次无复现 ✓
- 30 轮随机参数（B/D/H/W/C/N 全随机，含 W<H、C 非 8 对齐、N=0 边缘）：30/30 全通过 ✓

### Rank 计算修复详情

旧公式 `x*(W*D*B) + y*(D*B) + z*B + batch` 在 W < H 时有 rank 碰撞：
- 例：W=3, H=8 时，点 (x=2,y=3) 和 (x=1,y=6) 的 rank 都是 9
- 碰撞导致不同 voxel 的点被归入同一 interval，kernel 只写其中一个 voxel → 另一 voxel 输出全零
- 新公式 `x + y*W + z*W*H + batch*W*H*D` 对所有合法坐标严格单射，无碰撞

## 性能

| num_points | baseline ascendc_ms | iter1 (binding) ascendc_ms | iter2 (binding+block) ascendc_ms | vs baseline |
|-----------|---------------------|---------------------------|----------------------------------|-------------|
| 2,000 | 2.413 | 1.905 | 1.847 | -23% |
| 8,000 | 3.544 | 3.018 | 2.844 | -20% |
| 32,000 | 8.244 | 7.460 | 6.870 | -17% |
| 128,000 | 26.160 | 25.084 | 23.430 | -10% |
| 512,000 | 82.992 | 80.769 | 77.396 | -7% |

块处理在中等规模（8k-128k 点）提升显著（-17%~-10% vs baseline），因为减少了小粒度 GM 读延迟累积。大规模（512k）提升较小（-6%），因为 ~80k interval 的固定开销仍主导，且 `PIPE_ALL` 屏障引入额外同步开销。

## 瓶颈分析

- **每 interval 固定开销仍主导**：512k 点 / 80k voxel → ~80k interval，每 interval 仍需 `Duplicate → Add → DataCopy(out)` + `PIPE_ALL` 屏障。块处理减少了 GM 读次数（多 interval 合并一次读），但写和计算仍逐 interval 串行。
- **PIPE_ALL 开销**：全管道排空比定向屏障更保守，但 310P 上定向屏障不保证跨管道可见性，`PIPE_ALL` 是正确性必需。
- **UB 容量限制**：`tilePoints=256`（C=80 时），每块最多 256 行。512k 点 / 80k interval → 平均每 interval ~6.4 行 → 每块约 40 interval，GM 读次数从 80k 降至 ~2k。但写次数仍为 80k。

## 编译缓存注意事项

310P 的 TBE 编译系统在 `/data/kernel_meta` 和 `/workspace/unum_ops/kernel_meta` 缓存编译结果。修改 kernel 源码后必须清除这两个目录 + `build_out`，否则编译器复用旧二进制。`rebuild_install.sh` 需在 `build.sh` 前清除缓存。

## 下一步

1. **减少写延迟**：多 interval 写同一 voxel 的情况（多对一映射）可合并写出。
2. **双缓冲预取**：块间重叠 load/compute（需解决 `PIPE_ALL` 全局屏障限制）。
3. **interval 合并**：相邻 interval 映射到同一 voxel 时可合并为一个累加区间。
