# unum_ops

算子集合 + 纯 torch/numpy 跨硬件实现 + AscendC 算子。

- 所有子模块 lazy import，缺失依赖静默跳过
- 设备自动检测: `cpu` / `cuda` / `npu_310p` / `npu_910`
- 算子迁移顺序: Cuda → Torch-Native → AscendC

接口在 310P 上的可用性见 [310P 接口支持矩阵](support_matrix.md)
（由 `scripts/gen_support_matrix.py` 从 `unum_ops.INTERFACES_310P` 自动生成）。

## 本地构建文档

```bash
pip install mkdocs
python scripts/gen_support_matrix.py
mkdocs serve
```
