#!/bin/bash
set -e
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if [ -z "$BASE_LIBS_PATH" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        if [ -z "$ASCEND_AICPU_PATH" ]; then
            echo "please set env."
            exit 1
        else
            export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
        fi
    else
        export ASCEND_HOME_PATH=$ASCEND_HOME_PATH
    fi
else
    export ASCEND_HOME_PATH=$BASE_LIBS_PATH
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
script_path=$(realpath $(dirname $0))
cd "$script_path"

BUILD_DIR="build_out"
mkdir -p build_out
rm -rf build_out/*
opts=$(python3 $ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py $script_path/CMakePresets.json)
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')
min_version="3.19.0"

if [ "$(printf '%s\n' "$min_version" "$cmake_version" | sort -V | head -n1)" != "$min_version" ] ; then
    cmake -S . -B "$BUILD_DIR" $opts
else
    cmake -S . -B "$BUILD_DIR" --preset=default
fi
cmake --build "$BUILD_DIR" --target binary package -j$(nproc)

# 构建 op_extension (.so)
echo "--- building op_extension ---"
EXT_DIR="$script_path/op_extension"
EXT_BUILD="$EXT_DIR/build"
mkdir -p "$EXT_BUILD"
TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null)
if [ -z "$TORCH_CMAKE" ] || [ ! -d "$TORCH_CMAKE" ]; then
    echo "ERROR: torch not found (python3 -c 'import torch' failed)."
    echo "Please install torch (and torch_npu) in the python3 environment first."
    exit 1
fi
cd "$EXT_BUILD"
cmake "$EXT_DIR" -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH"
cmake --build "$EXT_BUILD" -j$(nproc)
echo "--- op_extension built: $EXT_BUILD/libbev_pool_ops.so ---"