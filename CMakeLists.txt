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

# aji: thin dispatcher (what the mpv filter loads as lib=). Parses the
# conf's [global] backend key and forwards the C ABI to the sibling
# backend library. No CUDA/TRT/ORT dependencies of its own.
add_library(aji SHARED src/aji_dispatch.cpp src/aji_conf.cpp)
target_include_directories(aji PUBLIC include)
target_link_libraries(aji PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(aji PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden)

# aji_trt: TensorRT/CUDA backend (the former aji.dll).
add_library(aji_trt SHARED src/aji_trt.cpp src/aji_conf.cpp src/kernels.cu)
target_include_directories(aji_trt
    PUBLIC include
    PRIVATE ${AJI_TRT_INCLUDE})
target_link_directories(aji_trt PRIVATE ${AJI_TRT_LIB})
target_link_libraries(aji_trt PRIVATE ${AJI_NVINFER} CUDA::cuda_driver CUDA::cudart)
set_target_properties(aji_trt PROPERTIES
    BUILD_RPATH "${AJI_TRT_LIB}"
    INSTALL_RPATH "${AJI_TRT_LIB}"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden)
if(NOT WIN32)
    # backends export the same aji_* names as the dispatcher; bind their
    # internal references locally so the global (dispatcher) symbols
    # can't interpose
    target_link_options(aji_trt PRIVATE -Wl,-Bsymbolic)
endif()

add_executable(aji_harness src/harness.c)
target_link_libraries(aji_harness PRIVATE aji CUDA::cudart)
add_dependencies(aji_harness aji_trt)
set_target_properties(aji_harness PROPERTIES
    BUILD_RPATH "${AJI_TRT_LIB};$<TARGET_FILE_DIR:aji>")

# aji_dml: DirectML backend (Windows only). Needs the ONNX Runtime
# DirectML headers (Microsoft.ML.OnnxRuntime.DirectML nupkg) and the
# DirectML header (Microsoft.AI.DirectML nupkg); onnxruntime.dll and
# DirectML.dll are loaded at runtime from the shim's own directory, so
# nothing is import-linked.
if(WIN32)
    set(AJI_ORT_ROOT "" CACHE PATH
        "Microsoft.ML.OnnxRuntime.DirectML package root")
    set(AJI_DML_ROOT "" CACHE PATH
        "Microsoft.AI.DirectML package root")
    add_library(aji_dml SHARED src/aji_dml.cpp src/aji_conf.cpp)
    target_include_directories(aji_dml
        PUBLIC include
        PRIVATE "${AJI_ORT_ROOT}/build/native/include"
                "${AJI_DML_ROOT}/include")
    target_link_libraries(aji_dml PRIVATE d3d12 d3d11 dxgi d3dcompiler)
    set_target_properties(aji_dml PROPERTIES CXX_VISIBILITY_PRESET hidden)

    add_executable(aji_harness_dml src/harness_dml.cpp)
    target_link_libraries(aji_harness_dml PRIVATE aji d3d11)
    add_dependencies(aji_harness_dml aji_dml)
endif()

# Parity test tool: compiles the kernels directly (the shared lib hides
# the launcher symbols), no TensorRT dependency.
add_executable(aji_kernel_test src/kernel_test.c src/kernels.cu)
target_include_directories(aji_kernel_test PRIVATE include src)
target_link_libraries(aji_kernel_test PRIVATE CUDA::cudart)
