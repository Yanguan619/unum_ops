# Agent Instructions

## Build & Test Commands

```bash
# bev_pool: full rebuild + install + test
bash csrc/ascend/bev_pool/rebuild_install.sh
python -m pytest test/test_bev_pool.py -q

# voxelization: build OPP + install + op_extension + test
cd csrc/ascend/voxelization && rm -rf build_out /data/kernel_meta && bash build.sh && \
  PKG_DIR="build_out/_CPack_Packages/Linux/External/custom_opp_openEuler_aarch64.run/packages/vendors/voxelization" && \
  cp -a "$PKG_DIR/." /usr/local/Ascend/cann-9.0.0/opp/vendors/voxelization/ && \
  cd op_extension && rm -rf build && mkdir build && cd build && \
  TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") && \
  cmake .. -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0 -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" && \
  make -j4 && cd /workspace/unum_ops && \
  python -m pytest test/test_voxelization.py -q

# voxelization benchmark
python -m pytest benchmark/bench_voxelization.py -v

# spconv tests
python -m pytest test/test_spconv.py -q

# stability test
python -m pytest test/test_stability.py -v

# all tests
python -m pytest test/test_spconv.py test/test_bev_pool.py test/test_voxelization.py -q
```

## Completed

- [x] **bev_pool Python wrapper rank 精度**: `ranks.float().argsort()` → `ranks.argsort()` (int64)
- [x] **bev_pool binding 缺少输入验证**: 加 14 个 TORCH_CHECK (dtype/device/contiguous/shape)
- [x] **bev_pool kernel 无 start 下标越界检查**: 3 处加 `start+length > numPoints` 防护
- [x] **host tiling gridTotal uint32 溢出**: uint64 计算 + 校验 `<= UINT32_MAX`
- [x] **build.sh 缺少 set -e 且版本比较有 bug**: 加 set -e + `sort -V` 版本比较
- [x] **rebuild_install.sh 硬编码路径**: 动态 ASCEND_HOME_PATH + 动态查找 PKG_DIR
- [x] **测试缺少边缘覆盖**: 16 个测试 (OOB/C=0/dtype/非连续/rank int64/网格维度)
- [x] **spconv 正确性 bug**: SubM padding/stride/dilation, 2D _triple, dense()/from_dense() device, VoxelGenerator, _gather edge
- [x] **voxelization 算子**: 独立 vendor + dlopen 统一加载, 20 测试通过
- [x] **voxelization 性能优化**: wrapper 免 D2D 拷贝, 固定开销 ~7ms→~2.5ms (-64%)
- [x] **README.md 虚假声明**: 重写 README，准确描述 AscendC/torch 算子
- [x] **pyproject.toml 控制台脚本**: 添加 `main()` 函数; 更新 description
- [x] **死代码**: 删除 NPUBridge、NPUStorageImpl、op_api_common.h、utils.h
- [x] **长稳压测**: `test_stability.py` 1500 次迭代 0 错误
- [x] **`.gitignore` 排除 `benchmark/output/`**
- [x] **voxelization 测试补边界**: 空输入/全部越界/max_voxels截断/max_num_points截断/坐标顺序/非均匀voxel_size
- [x] **bev_pool benchmark 补多维度**: 不同 C/D/H/W 组合
- [x] **spconv benchmark**: `benchmark/bench_spconv.py` 回归基准
- [x] **voxelization benchmark 补参数**: 不同 voxel_size/PCR 组合
- [x] **test_bev_pool.py 去重**: 删除重复的 `test_all_oob_points`
- [x] **voxelization 修复回归测试**: `test_small_max_voxels_clean_error` + `test_large_grid_clean_error` 断言干净报错而非崩溃

## Known Bugs (已修复)

- [x] **voxelization kernel max_voxels < 64 时 aicore 异常(507015)**: 根因是 workspace 放在 voxels 输出 buffer 内，小 max_voxels 时 buffer 容量不足。host 侧增加 `workspaceSize > voxBufferBytes` 校验，返回 GRAPH_FAILED 干净报错代替崩溃。
- [x] **voxelization kernel 小 voxel_size 大网格 aicore 异常**: 同根因（gridTotal 过大 → workspace 超 voxels buffer）。同一校验解决。

## Voxelization 性能

| num_points | ascendc_ms | numpy_ms | 加速比 |
|-----------|-----------|---------|--------|
| 5,000 | 2.44 | 3.74 | 1.5x |
| 20,000 | 6.06 | 15.30 | 2.5x |
| 50,000 | 12.81 | 39.40 | 3.1x |
| 100,000 | 23.41 | 80.43 | 3.4x |

## Remaining TODOs (by priority)

全部完成。项目当前状态：

- 3 个 AscendC 算子：bev_pool ✅ / voxelization ✅ / spconv ✅
- 91 个测试全部通过，1500 次长稳压测 0 错误
- 多 vendor 独立部署，统一 dlopen 加载，无冲突
- pip install 时自动编译 AscendC 扩展 .so
- 持续维护：`AGENTS.md` 记录所有已知问题和修复状态