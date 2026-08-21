# Voxelization 算子性能优化记录

> 设备：Ascend310P7 (Health=Warning, 单核模式)　CANN: 9.0.0
> 数据：KITTI 000008.bin (17221 points, FOV filtered)
> golden：M=3941 (VoxelGeneratorV2)

## 测试方法

- warmup 3 次 + bench 20 次取平均
- 测试程序：`test/aclnn_test.cpp`（aclnn API 调用 + aclrtSynchronizeStream 计时）

## 优化记录

### 基线 (52.0 ms)

- per-point DataCopy 读-改-写 localCnt（每次 8-int32 块）+ 每步 PipeBarrier
- InitWorkspace 标量循环清零 214272 个 bin（~21ms）
- ScatterPoints 重复 LoadTile + ComputeTile（第二遍遍历全部点）
- 正确性：PASS (3941/3941)

### 优化1: bin-tile-centric (610.8 ms → 回退)

- 每个 bin tile (1024 bins) 加载 localCnt 到 UB，遍历所有 point tile 域内增减
- 失败原因：210 个 bin tile × 17 个 point tile = 3570 次 LoadTile+ComputeTile，比 point-centric 的 17 次多 210 倍
- 正确性：FAIL (标量写 ptLocalPos 跨 bin-tile 不可见)

### 优化2: DataCopy 清零 + 去 PipeBarrier (49.5 ms)

- InitWorkspace 从标量循环改为 DataCopy 批量清零（210 块 × 1024 元素）
- 去除 per-point DataCopy 之间的 PipeBarrier（DataCopy 本身有隐式同步）
- 正确性：PASS (3941/3941)
- 收益：52.0 → 49.5 ms (5% ↑)

### 优化3: 缓存 ComputeTile 结果 (47.5 ms)

- CountPoints 阶段 ComputeTile 后，将 bin/xi/yi/zi 缓存到 voxels output tensor 的 workspace 区域
- ScatterPoints 阶段用 DataCopy 读缓存替代 ComputeTile（省 17 次 ComputeTile ≈ 13 次 vector API 调用 × 1024 元素）
- 正确性：PASS (3941/3941)
- 收益：49.5 → 47.5 ms (4% ↑，累计 52.0 → 47.5 ms, 9% ↑)

### 优化4: tile size 2048 (50.6 ms → 回退)

- VOXEL_TILE_POINTS 从 1024 增大到 2048，LoadTile 次数减半（17→9）
- UB 占用从 ~89KB 增到 ~178KB（接近 192KB 上限）
- 失败原因：Gather offset 表 + DataCopy 数据量翻倍，每次 LoadTile 开销增大抵消了次数减少的收益
- 正确性：PASS，但性能退化
- 结论：1024 是 310P 上该算子的最优 tile size

### 优化5: 多核并行 (33.2 ms, 2核)

- 用 `SyncAll<true>(syncGm_, syncUb_, blockNum)` 替代 `NotifyEvent`（syncGm_ 在 voxels output tensor 内，MTE2 可访问）
- workspace 数据全部放在 voxels output tensor 末尾（MTE2 可访问）
- 各核按 bin 分区并行处理（binsPerCore = gridTotal / blockNum）
- `blockSum` 用 DataCopy 写/读（跨核 MTE3/MTE2 可见）
- `ptLocalPos` per-core 独立区域（避免多核同时写同一 point tile 的 posBlk）
- ScatterPoints 重新 LoadTile+ComputeTile（不用缓存，避免多核 DataCopy 写同一 GM 地址竞态）
- 正确性：PASS (3941/3941)，偶发 4/3941 npts=0（SyncAll 在 Warning 态偶发不完整）
- 收益：47.5 → 33.2 ms (30% ↑，累计 52.0 → 33.2 ms, 36% ↑)

#### 多核同步方案对比

| 方案 | 核数 | 稳定性 | 耗时 | 备注 |
|------|------|--------|------|------|
| SyncAll<true> | 1 | ✓ | 47.5ms | 单核基线 |
| SyncAll<true>(syncGm_, syncUb_) | 2 | ✓ 3/5 | 33.2ms | 最优 |
| SyncAll<true>(syncGm_, syncUb_) | 4 | ✗ 2/5 | 22.9ms | 偶发竞态 |
| SyncAll<true>(syncGm_, syncUb_) | 8 | ✗ 0/5 | 14.3ms | 6/3941 coords 差异 |
| NotifyEvent/WaitEvent | 2 | ✓ 10/10 | 33.2ms | 同步等价 |
| NotifyEvent/WaitEvent | 8 | ✗ | 14.3ms | 同 SyncAll |

结论：310P7 Warning 态下，8核的 `SyncAll` 同步不稳定（偶发 4-6/3941 voxel 的 pos==0 点未处理）。2核 `SyncAll` 稳定性约 60%（偶发 4/3941）。根因是 `SyncAll` 内部的 `__sync_all_stub` 硬件指令在 Warning 态下不完全可靠。

> ⚠️ 上述结论已被推翻：偶发丢点根因不是 `SyncAll`，而是多核并发标量写输出 tensor 同一 32B 缓存行的 store 丢失（详见优化6）。

### 优化6: 单写者最终写 + 硬件 SyncAll (14.9 ms, 8核)

- **根因确认**：`ScatterPoints` 阶段所有核并发标量写 `coords/npts` 输出（`pos==0` 分支）——多个核写同一 32B 缓存行的不同字段时 store 互相覆盖丢失，表现为确定性丢 14 个 vid 的 npts、coords[3743] 丢写
- **修复 1（单写者）**：`ScatterPoints` 的 `pos==0` 分支改为 MTE3 DataCopy 写全局 scratch 槽位 `[vid,zv,yv,xv,cntV,0,0,0]`（每 vid 一个 32B 槽，vid 唯一，无跨核写竞争）；新增 `WriteCoordsNpts()` 由 core 0 单写者执行：MTE2 读 blockSum 重建 M，每 8 槽一批 MTE2 读 scratch，校验槽内 vid 后标量写输出
- **修复 2（同步）**：尝试纯软件 GM 标志轮询 barrier（MTE3 写 flag + MTE2 轮询 + 2 参 `DataCacheCleanAndInvalid`），实测在 310P7 上 MTE2 轮询读陈旧 flag，导致单次 launch 挂起或 30-109 s/launch（目标 15ms）——**MTE2/MTE3 跨核可见性不可靠**，且 3 参 DCCI 在 kernel 中不可编译
- **最终方案**：改回硬件 `AscendC::SyncAll<true>()`（ffts 跨核 flag + wait_flag_dev），配合单写者方案后 **24 launch 全 PASS、无挂起**，证明原"SyncAll 不稳定"是误判——真正的竞态是标量写缓存行，与 SyncAll 无关
- 正确性：PASS (3941/3941)，连续 10+ 次运行稳定
- 收益：52.0 → 14.9 ms (71% ↑)

#### 多核性能（8 核，修正后）

| 核数 | 耗时 |
|------|------|
| 2 | 33.89 ms |
| 4 | 23.62 ms |
| 8 | 14.97 ms |

## 最终性能

| 版本 | 核数 | 耗时 | 正确性 | 优化内容 |
|------|------|------|--------|---------|
| 基线 (单核) | 1 | 52.0 ms | PASS | — |
| 优化2 (单核) | 1 | 49.5 ms | PASS | DataCopy 清零 + 去 PipeBarrier |
| 优化3 (单核) | 1 | 47.5 ms | PASS | 缓存 ComputeTile 结果 |
| 优化5 (多核) | 2 | 33.2 ms | 偶发丢点 | 多核并行 + SyncAll |
| **优化6 (8核)** | **8** | **14.9 ms** | **PASS 稳定** | **单写者修复 + 硬件 SyncAll** |

**最终最优：14.9 ms**（8核），基线 52.0ms → 14.9ms，提升 **71%**，多核并行从"8核 SyncAll 不稳定"修正为"8核稳定 PASS"。
