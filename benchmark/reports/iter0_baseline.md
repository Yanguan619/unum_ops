# BevPool AscendC 算子优化报告 — 基线

## 环境与测试条件

- 设备：Ascend 310P（8 AICore/chip）
- 配置：B=1 D=8 H=100 W=100 C=80，grid=80000 voxel
- 正确性：`test/test_bev_pool.py` 全部 10 项
- 性能：`benchmark/bench_bev_pool.py`（warmup=5, repeat=20, torch.npu 计时）

## 正确性

```
10 passed in 12.07s
```

## 性能

| num_points | torch_ms | ascendc_ms |
|-----------|----------|-----------|
| 2,000 | 30.230 | 2.413 |
| 8,000 | 28.461 | 3.544 |
| 32,000 | 31.296 | 8.244 |
| 128,000 | 72.011 | 26.160 |
| 512,000 | 144.530 | 82.992 |

## 瓶颈分析

- 8 核全部启用，带宽约 2 GB/s，**远低于 GM 带宽上限** → 非带宽受限，而是**每 interval 固定开销/延迟受限**。
- 512k 点散布在 8 万 voxel 上 → ~8 万个小 interval，每个 interval 串行执行：
  `Duplicate(acc) → DataCopy(in) → 若干 Add → DataCopy(out)`，并多次 `PipeBarrier<PIPE_ALL>` 全管道排空，**无跨 interval 的读写重叠（MLP）**。
- 每 interval 的 GM 读延迟 + 写延迟 + barrier 串行累积，成为主要瓶颈。

## 优化方向

1. 动态 chunk 容量（按实际 C 分配，避免按 C=512 固定 64KB）
2. 用定向 PipeBarrier 替换 PIPE_ALL，减少全管道排空
3. interval 数据预取（双 chunk 缓冲），隐藏 GM 读延迟
4. 异步 copyout（双 acc 缓冲），隐藏 GM 写延迟 / 跨 interval 重叠
