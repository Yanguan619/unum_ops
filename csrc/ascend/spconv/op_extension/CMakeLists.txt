cmake_minimum_required(VERSION 3.16.0)
project(spconv_gemm_ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "/usr/local/Ascend/cann-9.0.0")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

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

set(SPCONV_OP_API_DIR "${ASCEND_HOME_PATH}/opp/vendors/spconv/op_api")
if(NOT EXISTS "${SPCONV_OP_API_DIR}/lib/libcust_opapi.so")
    set(SPCONV_OP_API_DIR "${ASCEND_HOME_PATH}/opp/vendors/customize/op_api")
endif()

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_DIR}/include
    ${ASCEND_HOME_PATH}/aarch64-linux/include
    ${SPCONV_OP_API_DIR}/include
    ${PY_INCLUDE_DIR}
)

set(SOURCES
    spconv_gemm_torch.cpp
    register.cpp
)

add_library(spconv_gemm_ops SHARED ${SOURCES})

target_compile_definitions(spconv_gemm_ops PRIVATE
    UNUM_VENDOR_DIR="spconv"
)

target_link_libraries(spconv_gemm_ops PRIVATE
    "${TORCH_LIBRARIES}"
    "${TORCH_NPU_DIR}/lib/libtorch_npu.so"
)

set_target_properties(spconv_gemm_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)