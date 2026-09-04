# Agent Instructions

## Build & Test Commands

```bash
# bev_pool: full rebuild + install + test
bash csrc/ascend/bev_pool/rebuild_install.sh
python -m pytest test/test_bev_pool.py -q

# spconv tests
python -m pytest test/test_spconv.py -q

# all tests
python -m pytest test/test_spconv.py test/test_bev_pool.py -q
```

## Remaining TODOs (by priority)

### 🔴 High

- [ ] **bev_pool Python wrapper rank 精度** (`src/unum_ops/bev_pool/__init__.py:104`): `ranks.float().argsort()` 当 `B*D*H*W > 2^24` 时 float32 精度不足，不同 voxel 碰撞。改为 `ranks.argsort()`（int64 排序）

- [ ] **bev_pool binding 缺少输入验证** (`csrc/ascend/bev_pool/op_extension/bev_pool_torch.cpp:88-95`): 无 dtype、device、contiguous、shape 检查 → 非连续/CPU/float16 输入静默错

- [ ] **bev_pool kernel 无 start 下标越界检查** (`op_kernel/bev_pool.cpp:58-105`): 当 `start > numPoints` 时 coordPtr_/featsPtr_ 读越界

- [ ] **host tiling gridTotal uint32 溢出** (`op_host/bev_pool.cpp:38`): `B*D*H*W` 用 uint32 计算，乘积 > 2^32 时溢出为 0 → kernel 写越界

- [ ] **build.sh 缺少 set -e 且版本比较有 bug** (`build.sh:1,36`): cmake 版本字符串比较 `3.10.0 < 3.19.0` 错误；`cmake --build` 失败不中断

- [ ] **rebuild_install.sh 硬编码路径** (`rebuild_install.sh:7,10`): CANN 版本 `cann-9.0.0`, 架构 `aarch64`, 发行版 `openEuler` 硬编码

- [ ] **测试缺少边缘覆盖** (`test/test_bev_pool.py`): 无 OOB 坐标、C=0、N=0、非连续输入、dtype 异常测试

### 🟡 Medium

- [ ] **README.md 虚假声明**: "此项目算子全部为 torch/triton 实现" → `bev_pool` 是 AscendC

- [ ] **.so 加载路径脆弱** (`src/unum_ops/bev_pool/__init__.py:23-25`): `_SO_REL` 相对源码树，`pip install` 到 site-packages 后找不到

- [ ] **pyproject.toml 控制台脚本不存在** (`pyproject.toml:30`): `unum_ops = "unum_ops:main"` 但 `main()` 不存在

- [ ] **输出形状不匹配**: `InferShape` 声明 5D `[B,D,H,W,C]`，实际返回 2D `[gridTotal, C]`。CANN 未来版本若加形状验证会崩溃

- [ ] **死代码**: `op_extension/NPUBridge.cpp`, `NPUStorageImpl.cpp` 未在 CMake 中编译，可删除

### 🟢 Low

- [ ] 许可证头不一致（华为 OSL vs BSD 3-Clause vs 无头）
- [ ] `.gitignore` 未排除 `benchmark/output/`