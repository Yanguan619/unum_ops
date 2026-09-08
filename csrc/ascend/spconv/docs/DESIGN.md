# SpconvGemm AscendC 算子设计文档

## 概述

`SpconvGemm` 是稀疏卷积（spconv）特征聚合阶段的自定义 AscendC 算子。
它计算 `out[n, o] = bias[o] + sum_{k,i} feats[k, n, i] * weight[k, i, o]`，
即 gather 后 GEMM + bias。

## 调用约定

- feats: (K, N, C_in) float32 — 已按邻居表 gather，无效邻居已置 0
- weight: (K, C_in, C_out) float32 — spconv 的 _wT 布局
- bias: (C_out,) float32
- params: (6,) int32 — [K, N, C_in, C_out, hasBias, blockNum]
- out: (N, C_out) float32

## 310P 适配

Ascend 310P 上，GET_TILING_DATA 宏将 tiling struct 烘焙为编译期常量（P4），
因此 kernel 不依赖 tiling 中的 shape 参数。所有 shape 参数（K, N, Cin, Cout）
通过 `params` 输入在运行时从 GM 读取。

输出写通过 MTE3 DataCopy 完成，避免 310P 标量 GM 写缓冲不可见问题（P3）。

## 性能现状（诚实记录）

当前内核为**标量实现**（逐体素逐通道标量累加），在 Ascend 310P 上相比
NPU 原生 einsum **慢 50~100x**（einsum 使用 Cube 单元矩阵乘）。

因此 `conv.py::_gather` 的 drop-in 默认走 einsum；设置
`UNUM_SPCONV_USE_ASCENDC=1` 才启用 AscendC 内核（正确性已由 75 个测试验证）。

### 为什么向量化路径不可用

尝试过三种向量化方案均失败（310P 硬件限制）：
1. `Muls/Add` + LocalTensor 视图（`wUb[i*Cout]`）——结果错误
2. `Muls/Add` + GetValue/SetValue 手动拷贝权重行到独立 buffer——结果错误
3. `Muls/Add` + 标量 GM 读 feats——结果错误

根因推测：标量单元（GetValue/SetValue/GM 读）写入的数据经标量管线，
向量单元（Muls/Add）读取时存在可见性/一致性问题（类似 P3 的标量写缓冲）。
`PipeBarrier<PIPE_ALL>()` 也无法解决。

### 优化方向

- **Cube 单元**（`AscendC::Mm`）：`out = sum_k feats_k @ weight_k^T` 用矩阵乘指令，
  可匹配或超越 einsum。L0A/L0B/L0C buffer + LoadData 配置较复杂，需专项开发。
- **PyTorch 原生 matmul 融合**：在 torch 侧用 `bmm + reduce` 代替逐点 einsum，
  已是当前 einsum 路径的实现，无需自定义内核。

## 多核分区

- blockNum = 8（固定，对应 310P AIV 核数）
- 每个核处理 rowsPerCore = ceil(N / 8) 行
- 分区在 kernel 内根据 params[5] 和 GetBlockIdx() 计算，不依赖烘焙 tiling

## 目录结构

```
csrc/ascend/spconv/
├── CMakeLists.txt / CMakePresets.json / build.sh / rebuild_install.sh / custom_op.json
├── op_host/spconv_gemm.cpp          # TilingFunc, InferShape, InferDataType, OpDef
├── op_kernel/spconv_gemm.cpp        # AscendC 内核（标量路径 + 多核 + MTE3 输出）
├── op_kernel/spconv_gemm_tiling.h   # tiling struct（仅用于 SetBlockDim）
├── op_extension/                     # aclnn 绑定 + TORCH_LIBRARY 注册
│   ├── spconv_gemm_ops.h
│   ├── spconv_gemm_torch.cpp
│   ├── register.cpp
│   └── CMakeLists.txt
```

## Python 集成

`src/unum_ops/spconv/ascendc.py` 提供：
- `spconv_gemm(feats, weight, bias)` — autograd 封装，backward 用 CPU einsum
- `available()` — 检测扩展是否可加载

`conv.py` 的 `_gather` 方法在 NPU 上自动路由到 AscendC 内核。