# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} plugin_srcs)

set(NLOHMANN_JSON_URL "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip")
set(NLOHMANN_JSON_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/json")
set(NLOHMANN_JSON_ARCHIVE "${NLOHMANN_JSON_DIR}/include.zip")
set(NLOHMANN_JSON_HEADER "${NLOHMANN_JSON_DIR}/include/nlohmann/json.hpp")

file(MAKE_DIRECTORY "${NLOHMANN_JSON_DIR}")
if(NOT EXISTS "${NLOHMANN_JSON_HEADER}")
    file(DOWNLOAD "${NLOHMANN_JSON_URL}" "${NLOHMANN_JSON_ARCHIVE}"
        STATUS download_status
        LOG download_log
    )
    list(GET download_status 0 download_code)
    if(NOT download_code EQUAL 0)
        message(FATAL_ERROR "Failed to download nlohmann/json include.zip: ${download_status}\n${download_log}")
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${NLOHMANN_JSON_ARCHIVE}" --format=zip
        WORKING_DIRECTORY "${NLOHMANN_JSON_DIR}"
        RESULT_VARIABLE extract_code
        OUTPUT_VARIABLE extract_out
        ERROR_VARIABLE extract_err
    )
    if(NOT extract_code EQUAL 0)
        message(FATAL_ERROR "Failed to extract nlohmann/json include.zip: ${extract_code}\n${extract_out}\n${extract_err}")
    endif()
endif()

npu_op_library(cust_onnx_parsers ONNX_PLUGIN
    ${plugin_srcs}
)

target_include_directories(cust_onnx_parsers PRIVATE "${NLOHMANN_JSON_DIR}/include")

npu_op_package_add(${package_name}
    LIBRARY
        cust_onnx_parsers
)
