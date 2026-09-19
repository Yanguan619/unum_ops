# AGENTS.md

## Project

`unum_ops` — 算子集合 + 纯 torch/numpy 跨硬件实现 + AscendC算子。

所有子模块 lazy import，缺失依赖静默跳过。设备自动检测 (`cpu`/`cuda`/`npu_310p`/`npu_910`)。

## Operator migration order

Cuda → Torch-Native → AscendC (never Triton unless explicitly asked).

## Architecture

```
unum_ops/
├── src/unum_ops/       — Python 算子封装（lazy import，每算子一个子包；含编译产物 _libs/）
├── csrc/
│   ├── cuda/           — CUDA 内核
│   └── ascend/         — AscendC 内核（每算子一个独立 vendor）
├── test/               — pytest 测试（conftest.py 提供 @pytest.mark.npu / npu fixture 自动跳过）
├── benchmark/          — 性能测试 (bench_utils.py 提供 triton-free 替代)
├── scripts/            — check.sh 三档流水线 / perf_gate.py 性能门禁 / gen_support_matrix.py
├── 3rd/                — 打包的第三方依赖
├── docs/               — 用户文档（mkdocs；support_matrix.md 自动生成，勿手改）
└── .opencode/          — agent 产出的 TODO/PLAN/DEBUG/output，而存放到不是`/tmp/opencode/`下
```

- 每个 AscendC 算子是一个独立 OPP vendor，安装在 `$ASCEND_HOME_PATH/opp/vendors/`。`.so` 由 `hatch_build.py` 在 310P 上 pip install 时自动编译；Python 侧按 环境变量 → 包内 `_libs/` → 源码树 顺序查找。
- benchmark报告放在benchmark/reports/。