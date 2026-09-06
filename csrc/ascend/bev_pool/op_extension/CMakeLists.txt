cmake_minimum_required(VERSION 3.16.0)
project(bev_pool_ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "/usr/local/Ascend/cann-9.0.0")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# 从 Python 可执行文件探测 include 目录（sysconfig 更可靠，
# 不依赖 cmake 的 Development 组件/系统 python3-dev 包）
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PY_INCLUDE_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
string(REGEX REPLACE "\n.*" "" PY_INCLUDE_TMP "${PY_INCLUDE_RAW}")
string(STRIP "${PY_INCLUDE_TMP}" PY_INCLUDE_DIR)
if(NOT EXISTS "${PY_INCLUDE_DIR}/Python.h")
    # fallback: 用 sysconfig.get_config_var
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
            "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))"
        OUTPUT_VARIABLE PY_INC2_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    string(REGEX REPLACE "\n.*" "" PY_INC2_TMP "${PY_INC2_RAW}")
    string(STRIP "${PY_INC2_TMP}" PY_INCLUDE_DIR)
endif()
if(NOT EXISTS "${PY_INCLUDE_DIR}/Python.h")
    # fallback: 让 cmake 搜索
    find_path(PY_INCLUDE_DIR Python.h
        PATHS
            /usr/include/python3.11
            /usr/local/python3.11/include
            /usr/local/python3.11.15/include/python3.11
            /usr/include/python3.12
            /usr/local/include/python3.11
    )
endif()
message(STATUS "PY_INCLUDE_DIR=${PY_INCLUDE_DIR}")
if(NOT EXISTS "${PY_INCLUDE_DIR}/Python.h")
    message(FATAL_ERROR "Python.h not found. Install python3-dev or set PY_INCLUDE_DIR manually.")
endif()

# 自动探测 torch 的 CMake 路径（不依赖用户手动设置 CMAKE_PREFIX_PATH）
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import torch, os; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
string(REGEX REPLACE "\n.*" "" TORCH_CMAKE_TMP "${TORCH_CMAKE_RAW}")
string(STRIP "${TORCH_CMAKE_TMP}" TORCH_CMAKE_PREFIX)
if(TORCH_CMAKE_PREFIX AND EXISTS "${TORCH_CMAKE_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
endif()

find_package(Torch REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
string(REGEX REPLACE "\n.*" "" TORCH_NPU_TMP "${TORCH_NPU_RAW}")
string(STRIP "${TORCH_NPU_TMP}" TORCH_NPU_DIR)

# bev_pool 使用独立 vendor 名（非 customize），op_api 库在 vendors/bev_pool/ 下
set(BEV_POOL_OP_API_DIR "${ASCEND_HOME_PATH}/opp/vendors/bev_pool/op_api")
if(NOT EXISTS "${BEV_POOL_OP_API_DIR}/lib/libcust_opapi.so")
    set(BEV_POOL_OP_API_DIR "${ASCEND_HOME_PATH}/opp/vendors/customize/op_api")
endif()

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_DIR}/include
    ${ASCEND_HOME_PATH}/aarch64-linux/include
    ${BEV_POOL_OP_API_DIR}/include
    ${PY_INCLUDE_DIR}
)

set(SOURCES
    bev_pool_torch.cpp
    register.cpp
)

add_library(bev_pool_ops SHARED ${SOURCES})

target_compile_definitions(bev_pool_ops PRIVATE
    UNUM_VENDOR_DIR="bev_pool"
)

target_link_libraries(bev_pool_ops PRIVATE
    "${TORCH_LIBRARIES}"
    "${TORCH_NPU_DIR}/lib/libtorch_npu.so"
)

set_target_properties(bev_pool_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
