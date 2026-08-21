# 310P (Ascend310P / DAV_2201) 算子开发踩坑总结与设计原则

> 日期：2026-08-20　设备：Ascend310P7 (Health=Warning)　CANN: 9.0.0
> 算子：Voxelization (registry-invoke)　验证：PASS (M=3941, 3941/3941 点集匹配)

---

## 1. 问题清单

### P1. aclnn 框架对所有 tensor GM_ADDR 加 8 字节 header

**现象**：kernel 写入 `num_voxels[0] = 99999`，host 侧 `aclrtMemcpy` 读到的是 0。进一步发现所有 output tensor 的数据都偏移了 2 个 int32（8 字节）。

**根因**：aclnn 框架（`NnopbaseRunWithWorkspace`）在将 `aclTensor` 的 device 指针传递给 kernel 之前，会在指针前添加 8 字节 header（可能用于 tensor 元数据）。kernel 收到的 `GM_ADDR voxels` 实际指向 `devPtr + 8`，而不是 `devPtr`。

**验证方法**：kernel 中写 `num_voxels[0] = 77777`，host 从 `nvoxDev` 读 `nvoxHost[0..7]`，发现 77777 出现在 `nvoxHost[2]`（偏移 8 字节）。

**影响**：所有通过 `aclCreateTensor` 创建的 tensor（input + output）的 `GM_ADDR` 都有 8 字节偏移。workspace（通过 `aclrtMalloc` + `aclnnVoxelization(ws, ...)` 传入）**不受影响**。

**修复**：kernel Init 中对所有 tensor 指针减 8：
```cpp
points = (__gm__ uint8_t*)points - 8;
voxels = (__gm__ uint8_t*)voxels - 8;
coords = (__gm__ uint8_t*)coords - 8;
numPointsOut = (__gm__ uint8_t*)numPointsOut - 8;
numVoxelsOut = (__gm__ uint8_t*)numVoxelsOut - 8;
```

---

### P2. workspace GM 不被 MTE2 引擎映射（DDR 地址越界）

**现象**：`AscendC::DataCopy(cntBlk, localCntGm_[0], 8)`（MTE2 读 workspace GM）导致 aicore exception："The DDR address of the MTE instruction is out of range"。但 `AscendC::DataCopy(localCntGm_[0], zeroI, cnt)`（MTE3 写 workspace GM）正常工作。从 output tensor GM 做 DataCopy 读也正常。

**根因**：`aclrtMalloc` 分配的 workspace 内存地址范围与 aclnn 框架内部管理的 tensor 内存地址范围不同。MTE2（DataCopy GM→UB）只能访问 tensor 内存区域，不能访问 `aclrtMalloc` 分配的 workspace 区域。MTE3（DataCopy UB→GM）的地址校验更宽松，可以写 workspace。

**验证方法**：
- 从 `voxPtr_`（output tensor）DataCopy 读 8 个 int32 → **成功**，`npts[0]` 读到正确值
- 从 `localCntPtr_`（workspace GM）DataCopy 读 8 个 int32 → **MTE 越界崩溃**
- 两者地址前缀不同：tensor `0xe800012...`，workspace `0xe7fffc...`

**影响**：所有需要跨阶段（跨方法调用）DataCopy 读写 workspace 的操作都会崩溃。

**修复**：把 workspace 数据放到 voxels output tensor 末尾（MTE2 可访问区域）：
```cpp
uint32_t voxTotalFloats = maxVox * maxPts * 4;
__gm__ int32_t* wsBase = (__gm__ int32_t*)voxels + (voxTotalFloats - wsFloats);
localCntPtr_ = wsBase + offLocalCnt/4 + core_ * gridTotal;
```

---

### P3. 标量 GM 读写跨方法调用不可见

**现象**：`InitWorkspace()` 中标量写 `localCntPtr_[b] = 0`，在 `CountPoints()` 中标量读 `localCntPtr_[b]` 读到的是 -1（未初始化值），而不是 0。`asm volatile("" ::: "memory")` 也无法解决。`volatile` 限定符在 310P 上不兼容 `GlobalTensor::SetGlobalBuffer`。

**根因**：310P 标量单元的 GM 写操作通过标量写缓冲（scalar write buffer），该缓冲不会自动刷新到 DDR。`PipeBarrier<PIPE_ALL>()` 同步的是 MTE2/MTE3/Vector/Cube 管线，不包括标量写缓冲的刷新。因此标量写在跨方法调用时不可见。

**验证方法**：
- 同一方法内 `localCntPtr_[0] = 42; numPtPtr_[0] = localCntPtr_[0];` → 读到 42 ✓
- `InitWorkspace()` 中 `localCntPtr_[0] = 0`，`BlockCount()` 中读 `localCntPtr_[0]` → 读到 -1 ✗
- `DataCopy` (MTE3) 写 workspace 后，标量读 → 读到 MTE3 写入的值 ✓（MTE3→scalar 可见）
- 标量写 workspace 后，`DataCopy` (MTE2) 读 → **崩溃**（P2）

**影响**：任何需要跨阶段读写 workspace 数据的标量操作都不可靠。

**修复**：所有 workspace 读写改为 `DataCopy`（MTE2 读 / MTE3 写），按 8-int32（32B）块操作：
```cpp
// 读：MTE2 从 output tensor GM 读到 UB
AscendC::DataCopy(cntBlk, localCntGm_[blkBase], 8);
int32_t v = cntBlk.GetValue(idx);
cntBlk.SetValue(idx, v + 1);
// 写：MTE3 从 UB 写回 output tensor GM
AscendC::DataCopy(localCntGm_[blkBase], cntBlk, 8);
```

---

### P4. GET_TILING_DATA 在 310P 上编译期嵌入 tiling

**现象**：`GET_TILING_DATA(tilingData, tiling)` 宏在 310P 上将 tiling data 烘焙为编译期常量数组，不从 runtime 的 `tiling` GM_ADDR 读取。

**根因**：`get_op_tiling.py` 中 310P 分支：
```python
class_body += f"    uint8_t __ascendc_arr_##tiling_data[{tiling_size}] = {{{tiling_arr_data_str}}};"
class_body += f"    {tiling_struct} tiling_data = convert_from_bytes<{tiling_struct}>(...);"
```
`tiling_arg` 参数被完全忽略。tiling data 在 ATC 编译时由 host tiling 函数计算并嵌入 kernel 二进制。

**影响**：tiling data 实际值正确（已验证 numPoints=17221, gridX=432 等），但开发者不能在 runtime 动态修改 tiling。如果同一 kernel binary 需要支持不同输入 shape，必须重新编译。

**结论**：310P 的 dynamic shape 支持有限。tiling data 在编译时确定，runtime 不可变。开发时需注意 tiling 函数的编译期调用参数必须覆盖所有 runtime 场景。

---

### P5. 多核并发标量写输出 tensor 同一缓存行 → store 丢失（SyncAll 被误判"不稳定"）

**现象**：多核并行后 `SyncAll<true>(syncGm_, syncUb_, blockNum)` 偶发丢点：确定性地丢 14 个 vid 的 npts（含 3743），`coords[3743]` 整体丢写。当时误判为"SyncAll 在 Warning 态不稳定"，改为单核+软件轮询。

**根因（后来确认）**：丢点不是 SyncAll 的问题，而是 `ScatterPoints` 阶段所有核并发**标量写** `coords/npts` 输出 tensor——多个核写同一 32B 缓存行（不同 4B 字段）时，标量 store 相互覆盖丢失。标量写与 `SyncAll` 硬件同步无直接关系，`SyncAll` 本身稳定。

**验证方法**：
- 8 核 `SyncAll` + 标量写输出：确定性丢 14 个 vid 的 npts、coords[3743] 丢写
- 8 核 `SyncAll` + **单写者最终写**（core 0 独写输出，其他核写每 vid 独占的 scratch 槽位）：24 launch 全 PASS，无挂起
- 8 核软件 GM 标志轮询 barrier（不用 SyncAll）：单次 launch 挂起或 30-109 s/launch（见 P10）

**修复**：`ScatterPoints` 的 `pos==0` 分支改为 MTE3 DataCopy 写全局 scratch 槽位 `[vid,zv,yv,xv,cntV,0,0,0]`（每 vid 一个 32B 槽，vid 唯一，无跨核写竞争）；`WriteCoordsNpts()` 仅 core 0 执行，MTE2 分批读 scratch 后标量写输出。最终用硬件 `AscendC::SyncAll<true>()` 做两次核间同步，性能 14.9ms（8 核），连续运行稳定。

**教训**：多核算子中**同一缓存行的写者必须唯一**（单写者模式），否则标量/向量 store 在硬件层静默丢写，且表现与同步原语无关，极易误判为同步问题。

---

### P10. MTE2 轮询 GM 标志的软件 barrier 在 310P7 上读陈旧值（挂起/极慢）

**现象**：用纯软件 barrier（每核 MTE3 写自己的 GM 标志槽，其他核 MTE2 轮询全部标志 + `PipeBarrier`）替代 SyncAll 后，单次 launch 挂起或耗时 30-109 s/launch（目标 ~15ms），且行为非确定（首次 launch 有时快有时挂）。

**根因**：MTE2 的 GM 读在 L1/L2 缓存中命中陈旧值，看不到其他核 MTE3 刚写入的标志。尝试 `DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE>`（2 参形式）在每次轮询前失效缓存行，编译通过但**未解决**挂起/极慢。3 参形式（带 `DcciDst`）在 310P kernel 中不可编译（ccec 未定义 3 参 guard 所需宏）。

**结论**：310P7 上 MTE2/MTE3 跨核可见性不可靠，**不要用 GM 标志轮询做核间同步**。核间同步应使用硬件 `SyncAll<true>()`（ffts 跨核 flag + wait_flag_dev，硬件级保证新鲜读）。CANN 内部 `SoftSyncAllImpl` 也用硬件事件（`SetFlag<HardEvent::MTE2_S>`/`WaitFlag`）而非纯轮询。

---

### P6. DataCopyPad 在 310P 上不支持

**现象**：`AscendC::DataCopyPad(raw, pointsGm_[off * 4], params, padParams)` 编译时报 deprecated 警告，运行时产生 MTE 越界错误。

**根因**：310P (DAV_2201) 的 MTE 引擎不支持 `DataCopyPad` 的非对齐填充功能。CANN 头文件中标记为 "unsupported API on current device"。

**修复**：用 `DataCopy`（要求 32B 对齐）+ 标量循环处理尾块。整 tile 用 `DataCopy`，不满 tile 的末尾用标量 `SetValue` 逐点读 GM。

---

### P7. AtomicAdd 在 310P 上不可用

**现象**：`AscendC::AtomicAdd<int32_t>(&localCntPtr_[b], 1)` 编译报错 "no template named 'AtomicAdd'"。

**根因**：`AtomicAdd` 仅在 `__NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3510` 上可用。310P（`__NPU_ARCH__ == 2201`）不支持 GM 原子操作。

**影响**：需要原子更新的 histogram/scatter 场景必须用 DataCopy 读-改-写替代。

---

### P8. Muls 倒数预乘 vs Div 精度差异

**现象**：用 `Muls(tmp, xf, invVoxelSizeX, cnt)` 替代 `Div(tmp, xf, dx, cnt)` 导致 M=3940（少 1 个 voxel），验证 FAIL。

**根因**：`1.0f / 0.16f` 在 float32 中不是精确值（6.2500001...），乘法与除法在边界点（坐标恰好落在 voxel 边界上）的 floor 结果不同，导致个别点被分到不同的 voxel。

**修复**：改回 `Div`，保留 `bufDivX_/bufDivY_/bufDivZ` UB buffer 存储除数。接受 `Div` 的性能代价换取精度一致性。

**教训**：P0 优化中的 "Div→Mul 倒数预乘" 在需要 `floor` 精确匹配的场景**不安全**。只有在精度容差允许（如 softmax 的 exp 计算）时才能使用。

---

### P9. UB buffer 溢出导致 MTE 越界

**现象**：`bufW_` 大小为 `VOXEL_TILE_POINTS * sizeof(int32_t)` = 4096B，但 `alignedCnt = (cnt + 7) & ~7` 可达 1032，DataCopy 写 4128B 超出 buffer。

**修复**：buffer 大小加 8 元素余量：`pipe_->InitBuffer(bufW_, (VOXEL_TILE_POINTS + 8) * sizeof(int32_t))`。

---

## 2. 310P 算子开发设计原则

### 原则 1：GM 内存分类——tensor GM vs workspace GM

**规则**：310P 上存在两类 GM 内存，MTE2 引擎的访问能力不同：
- **tensor GM**（通过 `aclCreateTensor` 传入的 input/output）→ MTE2 可读、MTE3 可写 ✓
- **workspace GM**（通过 `aclrtMalloc` + `aclnnXxxOp(ws, ...)` 传入）→ **MTE2 不可读** ✗、MTE3 可写 ✓

**设计要求**：
- 需要跨阶段 DataCopy 读写的数据，**必须放在 output tensor 的 GM 中**（如末尾未使用区域）
- workspace GM 只能用于 MTE3 写（如初始化清零），**不能用于 MTE2 读**
- 标量 GM 读写只能用于 output tensor，不能用于 workspace（标量写 workspace 不刷到 DDR）

### 原则 2：标量 GM 操作的局限

**规则**：310P 标量单元的 GM 写不自动刷新到 DDR，跨方法调用不可见。`asm volatile` 和 `volatile` 均无法解决。

**设计要求**：
- 跨阶段共享的 workspace 数据，**必须用 DataCopy（MTE2/MTE3）读写**，不能用标量 `__gm__ T*` 读写
- DataCopy 读写粒度为 8 个 int32（32B），按 `bin & ~7u` 对齐
- 单方法内的标量 GM 读写可见（如直接写 output tensor），可安全使用

### 原则 3：aclnn 框架 8 字节 tensor 指针偏移

**规则**：aclnn（`NnopbaseRunWithWorkspace`）对所有通过 `aclCreateTensor` 创建的 tensor GM_ADDR 添加 8 字节 header。workspace 不受影响。

**设计要求**：
- kernel Init 中对所有 tensor 类型的 GM_ADDR **统一减 8**：
  ```cpp
  points = (__gm__ uint8_t*)points - 8;
  voxels = (__gm__ uint8_t*)voxels - 8;
  // ... 所有 input/output tensor
  ```
- workspace GM_ADDR **不减 8**
- 测试程序从 tensor 的 `aclrtMalloc` 返回指针直接 `aclrtMemcpy` 读取（无需偏移）

### 原则 4：核间同步用硬件 SyncAll，禁用 GM 标志轮询；写者须唯一

**规则**：
- 310P7 上**硬件 `SyncAll<true>()` 可用且稳定**（配合单写者方案，24 launch 全 PASS）；早期"Warning 态挂死"的结论是误判，实际是并发标量写缓存行的丢写（P5）
- **不要用软件 GM 标志轮询**做核间同步：MTE2 读不到其他核 MTE3 刚写的标志（陈旧读），导致挂起或 30-109s/launch（P10）

**设计要求**：
- 核间同步只依赖硬件 `SyncAll<true>()`（`__sync_all_stub` 硬件跨核 flag）
- 需要跨核共享的数据，**写者必须唯一**：每核写自己独占的区域（如每 vid 一个 32B scratch 槽位），最终由单写者核统一写出
- 若必须用 GM 标志同步，参考 CANN `SoftSyncAllImpl` 的硬件事件模式（`copy_gm_to_ubuf` + `SetFlag<HardEvent::MTE2_S>`/`WaitFlag`），不要用 `PipeBarrier` + 轮询

### 原则 5：API 可用性矩阵

| API | 310P (2201) | 910B (2201) | 350 (3510) | 替代方案 |
|-----|:-----------:|:-----------:|:----------:|---------|
| DataCopyPad | ✗ | ✗ | ✓ | DataCopy + 标量尾块 |
| AtomicAdd | ✗ | ✗ | ✓ | DataCopy 读-改-写 |
| Gather | ✓ | ✓ | ✓ | — |
| GET_TILING_DATA | 编译期嵌入 | 编译期嵌入 | runtime 读取 | — |
| SyncAll | ✓ | — | — | — |

**设计要求**：开发前查阅 `__NPU_ARCH__` 条件编译，确认目标 API 可用。不可用的 API 在编译期就报错，不会在运行时产生静默错误。

### 原则 6：精度敏感场景的运算选择

**规则**：`Muls(x, 1/v)` 与 `Div(x, v)` 在 float32 下对 `floor` 操作可能产生不同结果（边界点差异），因为 `1/v` 不是精确值。

**设计要求**：
- 需要 `Cast<FLOOR>` 后做整数比较/索引的场景（如 voxelization 的网格坐标计算）→ **必须用 Div**
- 不涉及 floor 取整的场景（如 softmax、layer norm）→ 可以用 `Muls` 倒数预乘
- P0 优化 "Div→Mul" 需标注精度风险，必须做 E2E 精度验证

### 原则 7：UB buffer 大小留余量

**规则**：DataCopy 要求 32B（8 × int32）对齐，`alignedCnt = (cnt + 7) & ~7` 可能比 `cnt` 多 7 个元素。

**设计要求**：
- 所有用于 DataCopy 的 UB buffer 大小 = `VOXEL_TILE_POINTS + 8`（多 8 个元素余量）
- `pipe_->InitBuffer(bufX_, (VOXEL_TILE_POINTS + 8) * sizeof(T))`
- UB 总用量 = Σ(buffer sizes) ≤ 192KB (DAV_2201)

### 原则 8：GET_TILING_DATA 编译期嵌入

**规则**：310P 的 `GET_TILING_DATA` 宏将 tiling data 烘焙为编译期常量，不从 runtime GM 读取。

**设计要求**：
- tiling 函数在编译时被调用，其计算结果嵌入 kernel 二进制
- 同一 kernel binary 的 tiling data 不可在 runtime 改变
- 如需支持不同输入 shape，必须重新编译 kernel
- tiling struct 的字段顺序和大小必须 host/kernel 完全一致（无 padding 空洞）

### 原则 9：Gather 去交错可用

**规则**：310P 支持 `Gather` Level 2 API，可用于 UB 内的 stride 去交错。

**设计要求**：
- 预计算 offset 表（`offset[i] = i * stride_bytes`），在 Init 时一次性填入 UB
- 用 `srcBaseAddr` 参数区分通道（如 X=0, Y=4, Z=8, I=12 字节偏移）
- 整 tile：DataCopy 交错数据到 UB → 4 次 Gather 提取各通道
- 尾 tile：标量读 GM 填 raw buffer → Gather（或直接标量填各通道）

### 原则 10：测试程序设计

**设计要求**：
- 测试程序用 `aclrtMalloc` 分配 workspace，大小由 `aclnnXxxOpGetWorkspaceSize` 返回值决定
- `aclrtMemcpy` 从 tensor 的 `aclrtMalloc` 返回指针直接读取（kernel 内部已减 8）
- 写输出文件时按实际 `num_voxels` 截断，不写满 `maxVoxels` 容量
- golden 参考用 `VoxelGeneratorV2`（spconv）生成，对比 coords + num_points + voxel 点集

---

## 3. 310P 算子开发检查清单

- [ ] kernel Init 中对所有 tensor GM_ADDR 减 8（aclnn header）
- [ ] workspace GM 不用于 DataCopy 读（MTE2 不可访问）
- [ ] 跨阶段共享数据放在 output tensor GM 末尾
- [ ] 所有 workspace 读写用 DataCopy（MTE2/MTE3），按 8-int32 块
- [ ] 单方法内标量 GM 读写可用（仅限 output tensor）
- [ ] `DataCopyPad` 不可用，用 `DataCopy` + 标量尾块
- [ ] `AtomicAdd` 不可用，用 DataCopy 读-改-写
- [ ] 核间同步用硬件 `SyncAll`，不用 GM 标志轮询（MTE2 陈旧读会挂起）
- [ ] 同一缓存行的写者唯一（单写者最终写，其他核写独占槽位）
- [ ] `Div` 不可替换为 `Muls(倒数)` 当涉及 `Cast<FLOOR>` 精度匹配时
- [ ] UB buffer 大小 = tile_size + 8（DataCopy 对齐余量）
- [ ] GET_TILING_DATA 编译期嵌入，tiling struct 无 padding
- [ ] Gather offset 表在 Init 时预计算，srcBaseAddr 区分通道
- [ ] 测试程序从 tensor 基指针读取（kernel 已减 8），按实际 M 截断输出
