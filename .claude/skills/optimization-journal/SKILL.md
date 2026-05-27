---
name: unum-ops-triton-optimization
description: >
  记录 unum_ops/infllm_v2 项目中 max_pooling_1d_varlen Triton 算子的全部
  优化历程。包含每次尝试的假设、实现、性能数据和分析结论（哪些有效、哪些无效
  及其原因）。当用户询问该算子的优化历史、性能瓶颈、调参经验，或需要参考类似
  的 Triton 优化策略时，应加载此 skill。
---

# max_pooling_1d_varlen Triton 算子优化日志

## 1. 问题描述

优化 `src/unum_ops/infllm_v2/max_pooling_1d_varlen.py` 中的 `max_pooling_1d_varlen_ref_triton` 函数，使其在所有序列长度（64–8192）上达到或超越已安装的 CUDA C 扩展的性能，同时保持 16 个回归测试全部通过（最大绝对差 = 0.0）。

### 算子逻辑

输入 score 张量形状 `(num_heads, total_q, max_k)`，输出 block_score 形状 `(num_heads, total_q, max_blocks)`，其中 `max_blocks = ceil(max_seqlen_q / block_size)`。

对每个查询位置 q 和块 bk，在 KSIZE=5 的滑动窗口上取最大值：

```python
win_start = max(0, bk * STRIDE_POOL - PADDING)   # PADDING=1, STRIDE_POOL=4
win_end   = min(win_start + KSIZE, k_len)
output[h, q, bk] = max(score[h, q, win_start:win_end])
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `kernel_size` | 32 | 输入核大小（kernel 内部未直接使用） |
| `kernel_stride` | 16 | 池化步长 |
| `block_size` | 64 | 输出块大小 |
| `num_heads` | 8 | 注意力头数 |
| `batch_size` | 1 | batch 中的序列数 |
| `KSIZE` | 5 | 滑动窗口大小 |
| `STRIDE_POOL` | 4 | 内部步长 (block_size / kernel_stride) |
| `flat_stride` | seqlen/16 | 内存寻址的行步长 |

### 核心性能瓶颈

score 张量的实际内存布局是 `(num_heads, total_q, max_k)`，相邻 query 之间的 stride 是 `max_k`（seqlen=8192 时为 16384）。但 kernel 使用 `flat_stride = max_seqlen_q / kernel_stride`（seqlen=8192 时为 512）来寻址。这导致 2D 加载时每行（query）之间相隔 512 个元素（1024 字节），无法合并访存。

对于 seqlen=8192，实测 HBM 带宽利用率仅 ~4.6%（Triton）vs ~10%（CUDA C），峰值带宽 1555 GB/s 的 A100 上 Triton 仅达到 ~72 GB/s。

---

## 2. 优化尝试完整记录

### 尝试 0：纯 Python 循环（优化前基线）

**假设**：N/A —— 这是最初的实现，尚未使用 Triton。

**实现**：嵌套 Python for 循环遍历所有 (head, q, block) 组合，每次用 PyTorch 切片 + `.max()` 计算一个输出元素。

**结果**：灾难性缓慢。

| seqlen | 耗时 (ms) |
|--------|-----------|
| 64     | 49        |
| 128    | 210       |

**结论**：❌ 不可用。比 CUDA 慢 5000–26000 倍。

**原因**：每次 `.max()` 调用都触发一次 GPU kernel launch → CPU↔GPU 同步开销巨大。

---

### 尝试 1：1D Grid + 每个 (head, q_abs) 循环处理所有 block

**假设**：单个 Triton kernel 可以为每个 (head, q_abs) 用一个循环处理所有 block，消除 Python 开销。

**实现**：

- 1D grid: `num_heads * total_q` 个 program（seqlen=8192 时 8×8192=65536 个）
- 每个 program 处理一个 (head, q_abs) 的所有 block
- 对每个 block 加载 5 个 key 值并计算 max

**关键代码**：

```python
pid = tl.program_id(0)
head = pid // total_q
q_abs = pid % total_q
for bk in range(max_blocks):
    # 计算窗口、加载、max、存储
```

**结果**：比纯 Python 快 8000–26000 倍，小 seqlen 匹配 CUDA。

| seqlen | CUDA(ms) | Triton(ms) | 比值 |
|--------|----------|------------|------|
| 64     | 0.010    | 0.006      | 0.6× |
| 128    | 0.011    | 0.008      | 0.8× |
| 256    | 0.015    | 0.009      | 0.6× |
| 512    | 0.022    | 0.015      | 0.7× |
| 1024   | 0.034    | 0.027      | 0.8× |
| 2048   | 0.061    | 0.082      | 1.3× |
| 4096   | 0.149    | 0.335      | 2.2× |
| 8192   | 0.431    | ~1.37*     | 3.2× |

*近似值，初始实现的精确数据未保留。

**结论**：✅ 巨大改进，但 65536 个 program 太多，launch 开销和 occupancy 不佳。

---

### 尝试 2：2D Grid + 每个 (head, bk) 分 tile 循环（"2D Strided Load" 方案）

**假设**：将 query 分组为 tile，每个 program 处理一个 (head, bk) 的所有 tile。减少 program 数量，利用 Triton 的 2D 向量化加载。

**实现**：

- Grid: `num_heads * max_blocks`（8×128=1024 个 program）
- 每个 program 以 `BLOCK_Q` 为 tile 大小遍历 `total_q`
- 对每个 tile 执行 2D strided load，然后用 `tl.max(vals, axis=1)` 归约

**关键代码**：

```python
offs_q = tl.arange(0, BLOCK_Q)
offs_k = tl.arange(0, KSIZE_POW2)
for q_start in range(0, total_q, BLOCK_Q):
    # batch search 确定 k_len, off_bq
    base = head * total_q * flat_stride + q_abs * flat_stride
    offs = base[:, None] + win_start + offs_k[None, :]
    vals = tl.load(score_ptr + offs, ...)  # 2D strided load
    max_val = tl.max(vals, axis=1)
```

**BLOCK_Q=256, num_warps=4 时的结果**：

| seqlen | CUDA(ms) | Triton(ms) | 比值 |
|--------|----------|------------|------|
| 64     | 0.010    | **0.006**  | **0.6×** |
| 128    | 0.011    | **0.008**  | **0.7×** |
| 256    | 0.015    | **0.009**  | **0.6×** |
| 512    | 0.022    | **0.014**  | **0.6×** |
| 1024   | 0.034    | **0.027**  | **0.8×** |
| 2048   | **0.061** | 0.071      | 1.2× |
| 4096   | **0.150** | 0.251      | 1.7× |
| 8192   | **0.431** | 0.945      | 2.2× |

**结论**：✅ **最佳架构方案**，此后所有调优均基于此。

**成功原因**：

- 2D `tl.load` 会生成按行的 strided load 循环，Triton 编译器能较好地处理
- `tl.max(vals, axis=1)` 是一个 shuffle-based 归约，效率高
- 1024 个 program 比 65536 个合理得多
- BLOCK_Q=256 提供了足够的线程级并行

**仍然落后 CUDA 的原因**：

- 2D load 本质上是 stride 访问：warp 内的相邻线程读取地址相隔 `flat_stride` 个元素
- seqlen=8192 时 `flat_stride=512`，意味着每 128 字节 cache line 中只有 ~10 字节被使用
- 实测 Triton 带宽 ~72 GB/s（峰值 1555 GB/s 的 4.6%）
- CUDA C kernel 也受同样问题影响（~156 GB/s, 10%），但通过更好的指令调度实现了 2× 优势

---

### 尝试 3：逐 Key 位置 1D 加载 + 逐元素 tl.maximum

**假设**：用 KSIZE 个独立的 1D strided load 替代 2D load，用 `tl.maximum` 累加。KSIZE=5 很小，1D load 可能编译成更简单的代码。

**实现**：

```python
max_val = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
for k_off in range(KSIZE):
    k_pos = win_start + k_off
    mask_k = q_mask & (k_pos < win_end)
    vals = tl.load(score_ptr + base + k_pos, mask=mask_k, other=float("-inf"))
    max_val = tl.maximum(max_val, vals)
```

**结果**：

| seqlen | CUDA(ms) | Triton(ms) | 比值 |
|--------|----------|------------|------|
| 8192   | **0.431** | 1.371      | 3.2× |

所有 seqlen 下均比 2D 方案差。

**结论**：❌ **不要使用此方案**。

**原因**：

- 每个 1D load 仍然是 strided 访问（与 2D load 相同的问题）
- 但需要 KSIZE=5 次独立的 load 指令（vs 2D 方案的一次）
- `tl.maximum` 调用 KSIZE 次增加了计算开销
- Triton 编译器无法像优化 2D load + tl.max 那样优化这个模式

---

### 尝试 4：按 Query Tile 组织 Program（内部循环 bk）

**假设**：按 (head, query_tile) 分配 program，每个 program 遍历所有 bk。相邻 bk 的 key 窗口有重叠（5 个中有 1 个），可能改善 L1/L2 缓存复用。

**实现**：

```python
num_q_tiles = tl.cdiv(total_q, BLOCK_Q)
head = pid // num_q_tiles
q_tile = pid % num_q_tiles
# batch search 在 bk 循环外部（每个 program 只做一次）
for bk in range(max_blocks):
    win_start = max(0, bk * STRIDE_POOL - PADDING)
    vals = tl.load(score_ptr + offs, ...)
    max_val = tl.max(vals, axis=1)
    # 存储
```

**结果**：

| seqlen | CUDA(ms) | Triton(ms) | 比值 |
|--------|----------|------------|------|
| 8192   | **0.431** | 0.961      | 2.2× |

与按 bk 的方案几乎一致。

**结论**：❌ 对此问题无效。

**原因**：

- 相邻 bk 之间窗口重叠仅 5 个中的 1 个元素（key 位置 3–4），缓存复用可忽略
- 同一个 query tile 在所有 bk 上的数据跨度为 `BLOCK_Q × flat_stride × max_blocks` 字节 = 64×512×128×2 = 8 MB，远超 L1/L2 容量（192 KB / 40 MB on A100）
- 按 bk 和按 query_tile 的方式具有相同的工作总量和访存模式

---

### 尝试 5：BLOCK_Q 大小调优

**假设**：`BLOCK_Q`（每次处理的 query 数量）是关键调优参数。减小 BLOCK_Q 降低寄存器压力、可能提高 occupancy；增大 BLOCK_Q 减少循环开销、可能提高指令级并行。

**配置**：按 bk 方案（尝试 2），变化 BLOCK_Q，num_warps=4（默认）。

**结果**：

| BLOCK_Q | Programs | Tiles/program | 总迭代 | Seqlen=8192 |
|---------|----------|---------------|--------|-------------|
| 64      | 1024     | 128           | 131072 | **0.913ms** |
| 128     | 1024     | 64            | 65536  | 0.931ms |
| 256     | 1024     | 32            | 32768  | 0.951ms |
| 512     | 1024     | 16            | 16384  | 0.989ms |

**分析**：

- **BLOCK_Q=64 最佳**（0.913ms）：每次 tile 加载 64×8=512 个 bf16 元素；每个线程在归约期间持有 64/32=2 个值。低寄存器压力 → 高 occupancy。
- BLOCK_Q=128：每个线程持有 4 个值，寄存器压力略增。
- BLOCK_Q=256：每个线程持有 8 个值，寄存器压力显著。
- BLOCK_Q=512：每个线程持有 16 个值，寄存器压力过高 → occupancy 严重受损。

**结论**：✅ **BLOCK_Q=64 最优**。平衡点在寄存器压力与迭代次数之间。

---

### 尝试 6：num_warps 调优

**假设**：更多 warp 每个 program 可以通过更好的 warp 级调度隐藏访存延迟。

**配置**：BLOCK_Q=64，按 bk 方案。

**结果**：

| num_warps | Seqlen=8192 |
|-----------|-------------|
| 4（默认）  | **0.913ms** |
| 8          | 1.032ms     |

**结论**：❌ 更多 warp 反而更差。

**原因**：每个额外的 warp 增加寄存器文件分区。8 warp 时每个 warp 的寄存器减半，可能导致 spill 到 local memory。kernel 已有 1024 个并发 program，并行度充足。

---

### 尝试 7：BATCH_SIZE 改为 constexpr

**假设**：将 `batch_size` 设为 `tl.constexpr` 可以让 Triton 编译专用化 kernel，展开 batch search 循环，消除常见 `batch_size=1` 情况下的循环开销。

**结果**：正确性保持（16/16 测试）。但每个不同的 `batch_size` 值触发独立编译，增加了首次调用的延迟。

**性能影响**：可忽略（<1% 差异）。`batch_size=1` 时循环只有一次平凡迭代，Triton 编译器本就可以优化好。

**结论**：❌ 不值得。每次编译的开销超过微小性能收益。保持 `batch_size` 为运行时参数。

---

### 尝试 8：Shared Memory 作为暂存

**假设**：将 `(BLOCK_Q, WINDOW_SIZE)` 的 score 数据 tile 用合并的 1D 加载读入 shared memory，然后从 shared memory 计算每行的 max。如果合并加载可行，可能改善访存效率。

**分析**：要合并加载，相邻线程必须访问相邻内存地址。score 张量的偏移计算方式为：

```python
offset = head * total_q * flat_stride + q * flat_stride + k
```

线程 `i = q * KSIZE_POW2 + k` 访问 `head * total_q * flat_stride + (i // KSIZE_POW2) * flat_stride + (i % KSIZE_POW2)`。相邻线程：

- 线程 i=0: offset = 0
- 线程 i=1: offset = 1
- ...
- 线程 i=7: offset = 7
- 线程 i=8: offset = flat_stride + 0 = 512（与线程 7 相隔 505！）

因此在每组 KSIZE_POW2=8 的线程内访问是连续的，但组之间有 `flat_stride - 7` 的间隔。这不是合并访问——与直接 2D load 相同的问题。

**结论**：❌（分析性结论，未实际实现）。基础的 stride 访问模式无法通过 shared memory staging 修复，除非改变张量布局。

---

### 尝试 9：Flat Index Space（每个输出元素一个线程）

**假设**：用 `(num_heads * total_q * max_blocks,)` 的 grid，每个 program 计算一个输出元素。每个 program 只需加载 5 个连续的 key 值 → 简单代码，最小寄存器压力。

**分析**：

- Grid 大小：8 × 8192 × 128 = 8,388,608 个 program
- 每个 program 加载 5 元素 × 2 字节 = 10 字节，写入 1 元素 = 2 字节
- A100 有 108 个 SM，每个 SM 最多 ~2048 个常驻线程
- 每个 program 是一个 warp → 每个 SM 最多 64 个并发 program
- 需要 ~131K 个 wave 才能完成所有 8.4M program → launch 开销巨大

**结论**：❌（分析性结论，未实现）。program 数量爆炸，warp 之间无法合并访存。

---

### 尝试 10：使用实际 Tensor Stride 替代 flat_stride

**假设**：也许 kernel 应该使用 score 张量的实际 stride（`max_k`）而不是 `flat_stride`。

**分析**：

- score 张量 contiguous，shape `(num_heads, total_q, max_k)`，query 间 stride = `max_k`（seqlen=8192 时为 16384）
- 但 CUDA C kernel 和 Triton kernel 都使用 `flat_stride = 512`
- 两者输出一致（16 个测试 0.0 max diff）
- 这意味着 `flat_stride` 约定是代码库的有意设计，而非 bug

实际代码库中 `flat_stride = max_seqlen_q / kernel_stride = 8192 / 16 = 512`，对应于每个 query 的 stride 位置数。

**结论**：❓ 这是代码架构设计决策，非性能问题。不要修改。

---

## 3. 最终最佳配置

### Kernel 架构

- **Program 分解**：按 (head, bk)，遍历 query tile
- **Grid**: `num_heads × max_blocks`（8 × 128 = 1024 programs）
- **BLOCK_Q**: 64
- **num_warps**: 4（默认）
- **batch_size**: 运行时参数（非 constexpr）

### 最终性能

| seqlen | CUDA(ms) | Triton(ms) | 比值 |
|--------|----------|------------|------|
| 64     | 0.010    | **0.006**  | **0.6×** |
| 128    | 0.011    | **0.008**  | **0.7×** |
| 256    | 0.015    | **0.010**  | **0.7×** |
| 512    | 0.022    | **0.015**  | **0.7×** |
| 1024   | 0.035    | **0.031**  | **0.9×** |
| 2048   | **0.061** | 0.067      | 1.1× |
| 4096   | **0.149** | 0.242      | 1.6× |
| 8192   | **0.431** | 0.913      | 2.1× |

### 正确性

- 所有 16 个回归测试通过（max abs diff = 0.0）
- 覆盖 5 个 batch_size（1, 2, 4, 8, 16）× 3 种 num_heads（4, 8, 16）
- 对比基线：已安装的 CUDA C 扩展（`infllm_v2.max_pooling_1d`）

---

## 4. 核心经验总结

### Triton 调优的关键认知

1. **寄存器压力是第一约束**：BLOCK_Q=64 优于更大值，尽管循环次数更多。每次 tile 中的每个元素在 `tl.max` 归约期间都占用一个寄存器。32 线程/warp 时，BLOCK_Q=64 = 2 元素/线程，BLOCK_Q=256 = 8 元素/线程。

2. **2D `tl.load` + `tl.max(axis=1)` 快于逐元素循环**：2D strided load 显著优于逐 key 位置 1D 加载。Triton 编译器对向量化的 2D load+reduce 生成更好的代码。

3. **更多 warp ≠ 更快**：4→8 warp 时更差。1024+ 并发 program 已饱和 GPU。

4. **Program 分解影响不大**：按 bk 和按 query tile 方案性能几乎一致。内部循环结构（每次迭代处理多少数据）比外部循环层次更重要。

5. **constexpr 有代价**：仅在值真正影响代码生成且编译器无法自动优化时使用。

### 根本瓶颈

seqlen=8192 时 2.1× 差距的根本原因是 **stride 低效的访存模式**。这不是 Triton 特有的问题——CUDA C kernel 也有相同的访存模式，仅达到 ~10% HBM 带宽利用率。Triton 和 CUDA 之间的 2× 差距可能来自：

- CUDA C 允许更精确的指令调度（warp 级原语、手动展开）
- CUDA C 编译时带 `-arch=sm_80` 的架构特定优化
- CUDA C kernel 可能使用向量化访存指令（`half2`, `float4`）

### 进一步优化建议

如需缩小 2.1× 差距：

1. **改变 score 张量布局**：如果可能，将 query 维度的 stride 改为 1，使 2D load 完全合并。但需要全代码库的修改。

2. **用 CUDA C 实现 kernel**：现有的扩展基础设施（`C.cpython-311-x86_64-linux-gnu.so`）说明可行。

3. **使用 `tl.dot` + mask 矩阵**：对结构良好的问题，矩阵乘法可超越显式 max 操作。但窗口太小（KSIZE=5），不太可能有效。

4. **与上游算子融合**：如果 score 张量由前一个操作生成，将池化融入该 kernel，避免物化完整的 score 张量。

---

## 5. 所有尝试汇总

| # | 尝试 | 结论 | 最佳 Seqlen=8192 | 原因 |
|---|------|------|------------------|------|
| 0 | 纯 Python 循环 | ❌ | 49–210ms | CPU↔GPU 同步开销 |
| 1 | 1D grid, 65536 programs | ✅→❌ | ~1.3ms | program 太多 |
| 2 | **2D grid, 按 bk 分 tile** | **✅** | **0.945ms** | **最佳架构** |
| 3 | 逐 key 1D 加载 | ❌ | 1.371ms | 更多指令，更差的编译 |
| 4 | 按 query_tile 组织 program | ❌ | 0.961ms | 缓存复用可忽略 |
| 5 | **BLOCK_Q 调优** | **✅** | **0.913ms** | **寄存器压力权衡** |
| 6 | num_warps=8 | ❌ | 1.032ms | 寄存器不足 |
| 7 | BATCH_SIZE constexpr | ❌ | ~相同 | 编译开销 > 收益 |
| 8 | Shared memory 暂存 | ❌（分析） | — | 相同的 stride 模式 |
| 9 | 每个输出元素一个线程 | ❌（分析） | — | 840 万 program 不可能 |
| 10 | tensor stride vs flat_stride | ❓ | — | 架构约定 |

## 6. 文件位置

- **Triton kernel + wrapper**: `src/unum_ops/infllm_v2/max_pooling_1d_varlen.py`
- **正确性测试**（16 例）：`test/test_max_pooling_1d.py`
- **性能基准**：`benchmark/test_max_polling_1d.py`
- **CUDA C 参考**（编译后的 .so）：`.venv/lib/python3.11/site-packages/infllm_v2/C.cpython-311-x86_64-linux-gnu.so`
- **CUDA C wrapper**：`.venv/lib/python3.11/site-packages/infllm_v2/max_pooling_1d.py`
