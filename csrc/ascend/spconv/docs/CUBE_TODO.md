# Cube Mm 路径 TODO

## 目标
用 Ascend 310P 的 Cube 单元（`Mmad`）实现 spconv GEMM `out = A @ B + bias`，
达到或超越 NPU einsum 的性能（当前向量管线慢 3~10x）。

## 计算规约
```
out[n, o] = sum_{k,i} feats[k, n, i] * weight[k, i, o] + bias[o]
```
等价于单发 dense GEMM：
- A = all_feats.reshape(N, K*Cin)  —— (N, KC) row-major，已连续，无需 permute
- B = wT.reshape(K*Cin, Cout)      —— (KC, Cout) row-major，已连续
- out = A @ B + bias                —— (N, Cout)

## 310P Cube 硬件约束

### 流水线（3-stage）
```
GM ──LoadData2DGM2L1Cal──→ L1 (__cbuf__) ──LoadData2DL12L0ACal/L0BCal──→ L0A/L0B (__ca__/__cb__)
──MmadCal──→ L0C (__cc__) ──DataCopy──→ GM
```

### 关键限制
1. **GM→L0A/L0B 直接加载不支持**（`dav_c310` 中 `LoadData2DGM2L0ACal/L0BCal` 均 assert false）
2. **必须经 L1 中转**：GM→L1 (V1 `LoadData2DParams`)，然后 L1→L0A/L0B (V1 `LoadData2DParams`)
3. **对齐要求**（从 `LoadDataWithStrideImpl` 断言摘录）：
   - `mExtension % 16 == 0`（M 维 fractal 块对齐）
   - `kExtension * sizeof(T) % 32 == 0`（K 维 fractal 块对齐）
   - L0A/L0B 地址按 `VALUE_512` (512B) 对齐
   - L1 源地址按 `ONE_BLK_SIZE` (32B) 对齐
4. **Fractal 格式**：fp32 的 fractal 块为 16×16，`c0=16`，`dSize=2`

## 实现步骤

### Step 1: L1 缓冲分配
- L1 缓冲用 `TBuffAddr` + `TPosition::A1` / `TPosition::B1`
- 容量：`TOTAL_L0A_SIZE` / `TOTAL_L0B_SIZE` 来自硬件常量
- 存储 A tile 和 B 矩阵的 fractal 格式数据

参考代码（`kernel_operator_gemm_base_impl.h`）：
```cpp
TBuffAddr tbufa;
tbufa.logicPos = static_cast<uint8_t>(TPosition::A1);
l1a.SetAddr(tbufa);
l1a.InitBuffer(0, TOTAL_L1_SIZE / sizeof(PrimT<T>));
```

### Step 2: GM → L1 加载
函数：`LoadData2DGM2L1Cal(__cbuf__ T* dst, __gm__ T* src, const LoadData2DParams& params)`

`LoadData2DParams` 参数含义（从 `LoadData2DGM2L1Cal` 源码反推）：
```cpp
// 内部转换为：
uint16_t mStartPosition = 0;
uint16_t kStartPosition = params.startIndex;    // K-block fractal 起始索引
uint8_t  mStep = params.srcStride;              // M-step（每 repeat 的 M-blocks 数）
uint8_t  kStep = params.repeatTimes;            // K-blocks 数
int16_t  srcStride = params.srcStride;          // GM 行 stride（元素为单位）
uint16_t dstStride = params.dstGap + 1;         // L1 行 stride
```

**参数计算**（以 A 矩阵为例，M=N, K=KC）：
- `startIndex = 0`（从第一个 fractal 块开始）
- `repeatTimes = ceil(K / 16)`（K 维 fractal 块数）
- `srcStride = K`（GM 行 stride = 每行元素数）
- `dstGap = 16 - 1`（L1 行 stride = 16 元素）
- 注意：M 维需要按 16 分块，每块调用一次 LoadData

### Step 3: L1 → L0A 加载
函数：`LoadData2DL12L0ACal(__ca__ T* dst, __cbuf__ T* src, const LoadData2DParams& params)`

参数含义与 Step 2 类似：
- `startIndex`：L1 中 K-block 起始索引
- `repeatTimes`：K-blocks 数
- `srcStride`：L1 中 M-step
- `dstGap`：L0A 中行 stride - 1

### Step 4: Mmad
```cpp
MmadParams mmadParams;
mmadParams.m = mTile;      // M_tile (必须是 16 的倍数)
mmadParams.n = n;           // Cout (已对齐到 16)
mmadParams.k = k;           // K_blk (已对齐到 16)
mmadParams.unitFlag = 0;    // 普通矩阵乘
mmadParams.cmatrixSource = 0;  // 结果存 L0C
mmadParams.cmatrixInitVal = 1; // L0C 初始化为 0

MmadCal(l0c, l0a, l0b, mmadParams);
```

### Step 5: L0C → GM 写出
```cpp
// L0C 到 UB 或 GM
DataCopy(outGm, l0c, size);
```

## 代码位置
- 内核：`csrc/ascend/spconv/op_kernel/spconv_gemm.cpp`
- 核心函数：`ProcessCube()` 新方法，与现有 `Process()` 并列
- Tiling 扩展：`spconv_gemm_tiling.h` 需添加 cube 相关的 tiling 字段

## 关键参考文件
```
/usr/local/Ascend/cann-9.0.0/aarch64-linux/ascendc/include/basic_api/impl/dav_c310/kernel_operator_mm_impl.h
    - LoadData2DGM2L1Cal (line 46)
    - LoadData2DL12L0ACal (line 64)
    - LoadData2DL12L0BCal (line 85)
    - MmadCal (line 511-540)

/usr/local/Ascend/cann-9.0.0/aarch64-linux/ascendc/include/basic_api/impl/kernel_operator_gemm_base_impl.h
    - GetPingPongBuffer (line 263)
    - GetSingleThreadBuffer (line 291)
    - LoadL0A (line 190)
    - LoadL0B (line 162)
    - MmadFunc (line 217)
    - GemmExecNm (line 410)
    - GemmExecNmPingPong (line 342)

/usr/local/Ascend/cann-9.0.0/aarch64-linux/ascendc/include/basic_api/interface/kernel_operator_mm_intf.h
    - LoadData 2D/3D APIs
    - Mmad 签名
    - SetMMRowMajor / SetMMColumnMajor

/usr/local/Ascend/cann-9.0.0/aarch64-linux/ascendc/include/basic_api/interface/kernel_struct_mm.h
    - LoadData2DParams struct
    - MmadParams struct
```

## 验证方法
1. 先写一个最小 Cube 测试：固定 M=16, K=16, N=16，验证 GM→L1→L0A→Mmad→L0C→GM 完整通路
2. 逐步放大 M（tile over rows），验证多 tile 累加
3. 整合到现有 spconv_gemm 内核，`Process()` 中分支选择 cube 路径
4. 与 einsum 的端到端性能对比

## 已知风险
- 310P fp32 cube 的精确吞吐量未知（需实测）
- L1 到 L0A/L0B 的 stride/gap 参数容易算错
- 多核场景下 L1 资源竞争（每个核独立 L1）
- deprecated `Gemm` API 可能在未来 CANN 版本中移除