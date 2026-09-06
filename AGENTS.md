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

## Voxelization 性能

| num_points | ascendc_ms | numpy_ms | 加速比 |
|-----------|-----------|---------|--------|
| 5,000 | 2.44 | 3.74 | 1.5x |
| 20,000 | 6.06 | 15.30 | 2.5x |
| 50,000 | 12.81 | 39.40 | 3.1x |
| 100,000 | 23.41 | 80.43 | 3.4x |

## Remaining TODOs (by priority)

### 🟡 Medium

- [ ] **README.md 虚假声明**: "此项目算子全部为 torch/triton 实现" → `bev_pool` 是 AscendC

- [ ] **.so 加载路径脆弱** (`src/unum_ops/bev_pool/__init__.py:23-25`): `_SO_REL` 相对源码树，`pip install` 到 site-packages 后找不到

- [ ] **pyproject.toml 控制台脚本不存在** (`pyproject.toml:30`): `unum_ops = "unum_ops:main"` 但 `main()` 不存在

- [ ] **输出形状不匹配**: `InferShape` 声明 5D `[B,D,H,W,C]`，实际返回 2D `[gridTotal, C]`。CANN 未来版本若加形状验证会崩溃

- [ ] **死代码**: `op_extension/NPUBridge.cpp`, `NPUStorageImpl.cpp` 未在 CMake 中编译，可删除

### 🟢 Low

- [ ] 许可证头不一致（华为 OSL vs BSD 3-Clause vs 无头）
- [ ] `.gitignore` 未排除 `benchmark/output/`