cmake_minimum_required(VERSION 3.24)
project(animejanai-inference LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# TensorRT location (default: userland dpkg -x extract on Linux; on Windows
# pass -DAJI_TRT_ROOT=<dir with nvinfer_10.dll, nvinfer_10.lib, include/>).
if(WIN32)
    set(AJI_TRT_ROOT "" CACHE PATH "TensorRT root")
    set(AJI_TRT_INCLUDE "${AJI_TRT_ROOT}/include")
    set(AJI_TRT_LIB "${AJI_TRT_ROOT}")
    set(AJI_NVINFER nvinfer_10)
else()
    set(AJI_TRT_ROOT "$ENV{HOME}/sdk/tensorrt/usr" CACHE PATH "TensorRT root")
    set(AJI_TRT_INCLUDE "${AJI_TRT_ROOT}/include/x86_64-linux-gnu")
    set(AJI_TRT_LIB "${AJI_TRT_ROOT}/lib/x86_64-linux-gnu")
    set(AJI_NVINFER nvinfer)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(aji SHARED src/aji_trt.cpp src/aji_conf.cpp src/kernels.cu)
target_include_directories(aji
    PUBLIC include
    PRIVATE ${AJI_TRT_INCLUDE})
target_link_directories(aji PRIVATE ${AJI_TRT_LIB})
target_link_libraries(aji PRIVATE ${AJI_NVINFER} CUDA::cuda_driver CUDA::cudart)
set_target_properties(aji PROPERTIES
    BUILD_RPATH "${AJI_TRT_LIB}"
    INSTALL_RPATH "${AJI_TRT_LIB}"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden)

add_executable(aji_harness src/harness.c)
target_link_libraries(aji_harness PRIVATE aji CUDA::cudart)
set_target_properties(aji_harness PROPERTIES
    BUILD_RPATH "${AJI_TRT_LIB};$<TARGET_FILE_DIR:aji>")

# Parity test tool: compiles the kernels directly (the shared lib hides
# the launcher symbols), no TensorRT dependency.
add_executable(aji_kernel_test src/kernel_test.c src/kernels.cu)
target_include_directories(aji_kernel_test PRIVATE include src)
target_link_libraries(aji_kernel_test PRIVATE CUDA::cudart)
