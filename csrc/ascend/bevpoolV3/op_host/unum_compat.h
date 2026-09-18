/*
 * Copyright (c) 2026 Yanguan02
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * 兼容垫片：补 DrivingSDK 默认依赖 common/op_host/common.h 的工具宏，
 * 使 v3 算子能在 unum_ops build infra 下编译。
 */
#ifndef UNUM_COMPAT_H
#define UNUM_COMPAT_H

#include <cstdint>
#include <type_traits>
#include <graph/types.h>

// 向上对齐到 n 的整数倍（CeilAlign(x, n) = ((x + n - 1) / n) * n）
#ifndef CeilAlign
#define CeilAlign(x, n) (((x) + (n) - 1) / (n) * (n))
#endif

// 向下对齐到 n 的整数倍（FloorAlign(x, n) = (x / n) * n）
#ifndef FloorAlign
#define FloorAlign(x, n) ((x) / (n) * (n))
#endif

// 上取整除法（DivCeil(x, n) = (x + n - 1) / n）
#ifndef DivCeil
#define DivCeil(x, n) (((x) + (n) - 1) / (n))
#endif

// 提交 tiling 数据到 context：
//   GetTilingData<T>() 拿到 context 内部 buffer 指针（已 SetDataSize），
//   把本地结构体直接复制进去（tiling 结构体均为 POD，赋值即可）。
#ifndef ADD_TILING_DATA
#define ADD_TILING_DATA(context, tiling) do {                                                \
        using _TD = std::remove_reference<decltype(tiling)>::type;                            \
        auto* _td_ptr = (context)->template GetTilingData<_TD>();                             \
        if (_td_ptr == nullptr) return ge::GRAPH_FAILED;                                     \
        *_td_ptr = (tiling);                                                                  \
    } while (0)
#endif

// 仿 CANN 行为：nullptr 时返回 GRAPH_FAILED。
#ifndef CHECK_NULLPTR
#define CHECK_NULLPTR(p) do { if ((p) == nullptr) return ge::GRAPH_FAILED; } while (0)
#endif

#endif  // UNUM_COMPAT_H