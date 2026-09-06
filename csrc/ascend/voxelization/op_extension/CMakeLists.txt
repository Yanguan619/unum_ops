cmake_minimum_required(VERSION 3.16.0)
project(voxelization_ops)

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
    OUTPUT_VARIABLE TORCH_NPU_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_DIR}/include
    ${ASCEND_HOME_PATH}/aarch64-linux/include
    /usr/local/python3.11.15/include/python3.11
)

set(SOURCES
    voxelization_torch.cpp
    register.cpp
)

add_library(voxelization_ops SHARED ${SOURCES})

target_compile_definitions(voxelization_ops PRIVATE
    UNUM_VENDOR_DIR="voxelization"
)

target_link_libraries(voxelization_ops PRIVATE
    "${TORCH_LIBRARIES}"
    "${TORCH_NPU_DIR}/lib/libtorch_npu.so"
)

set_target_properties(voxelization_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)