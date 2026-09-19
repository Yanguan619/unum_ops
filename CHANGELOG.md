# Changelog

本项目所有显著变更记录于此。
格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- `bev_pool_v3` 注册进 `INTERFACES_310P`，其 OPP vendor（bevpoolV3）加入 `hatch_build.py` 自动编译清单
- `test/test_registry.py`：守护接口注册表与 AscendC vendor 编译清单的 meta-test
- `test/conftest.py`：`@pytest.mark.npu` / `@pytest.mark.cuda` 自动跳过标记与 `device` fixture
- `scripts/check.sh`：fast / full / bench 三档本地流水线（无远程 CI 依赖）
- `scripts/perf_gate.py`：benchmark 性能回归门禁（基线固化 + 阈值对比，基线在 `benchmark/reports/perf_baseline.json`）
- `scripts/gen_support_matrix.py` + `mkdocs.yml` + `docs/`：从 `INTERFACES_310P` 自动生成 310P 接口支持矩阵
- `CHANGELOG.md`

### Changed

- pre-commit 分层：commit 阶段只跑 lint + registry 快检（秒级），全量测试与 benchmark 移至 pre-push 阶段；`default_install_hook_types: [pre-commit, pre-push]`
- `test/test_atb.py` → `test/test_atb_attention.py`、`test/test_pa.py` → `test/test_paged_attention.py`：语义化命名，移除调试打印，补形状/有限性断言
- 补登 13 个漏登接口：pointnet2 `*_torch` 变体 ×4、infllm_v2 triton 别名 ×2、sparse_kernel_extension `get_block_table_v2/v3`、spconv `SparseConv3dCPU`/`SubMConv3dCPU` 兼容旧名、voxelization `voxelization_torch(_ref)`
- 子包 `bev_pool` / `bev_pool_v3` / `voxelization` 补 `__all__` 导出清单
- benchmark 报告统一存放于 `benchmark/reports/`（不再放 `docs/`）
