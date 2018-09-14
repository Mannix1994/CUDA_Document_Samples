cmake_minimum_required(VERSION 3.10)

project(CUDA_SAMPLE)

# 设置CUDA版本
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.0)

# 找CUDA
find_package(CUDA REQUIRED)

#
CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_NVCC_FLAGS Auto)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
message(STATUS ${CUDA_NVCC_FLAGS})

# 设置公共源文件
file(GLOB COMMON common.h common.cpp)

# 添加*.cu源文件
file(GLOB CU 2/sample_2.1.cu)

# 添加包含目录
include_directories(${CUDA_INCLUDE_DIRS})

# 添加可执行文件
CUDA_ADD_EXECUTABLE(sample_2.1 # target名
        ${CU}  # 添加*.cu源文件
        ${COMMON}  # 添加
        )

add_subdirectory(2)

add_subdirectory(3)
