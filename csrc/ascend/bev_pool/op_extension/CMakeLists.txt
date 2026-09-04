cmake_minimum_required(VERSION 3.16.0)
project(bev_pool_ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "/usr/local/Ascend/cann-9.0.0")
endif()

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX REPLACE "\n.*" "" TORCH_NPU_TMP "${TORCH_NPU_RAW}")
string(STRIP "${TORCH_NPU_TMP}" TORCH_NPU_DIR)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_DIR}/include
    ${ASCEND_HOME_PATH}/aarch64-linux/include
    ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/include
    /usr/local/python3.11.15/include/python3.11
)

set(SOURCES
    bev_pool_torch.cpp
    register.cpp
)

add_library(bev_pool_ops SHARED ${SOURCES})

target_link_libraries(bev_pool_ops PRIVATE
    "${TORCH_LIBRARIES}"
    "${TORCH_NPU_DIR}/lib/libtorch_npu.so"
)

set_target_properties(bev_pool_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
