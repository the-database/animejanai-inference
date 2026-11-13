cmake_minimum_required(VERSION 3.24)
# Select the default before project() initializes it from nvcc's default GPU.
# Preserve explicit -D overrides and CMake's CUDAARCHS environment override.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    # Ship kernels for every supported GPU generation (Turing..Blackwell), NOT
    # "native": native builds only for the build machine's GPU, so the shipped
    # binary dies with "no kernel image is available for execution on the device"
    # on every other architecture (e.g. built on a 5090/sm120 => fails on a
    # 4090/sm89). The trailing -virtual emits PTX so future archs JIT-compile.
    # Keep in sync with the TensorRT builder packs (sm75/80/86/89/90/100/120).
    # Local dev can still override with -DCMAKE_CUDA_ARCHITECTURES=native.
    set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real 100-real 120-real 120-virtual
        CACHE STRING "CUDA architectures for AJN kernels")
endif()
project(animejanai-inference LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# TensorRT location (default: userland dpkg -x extract on Linux; on Windows
# pass -DAJI_TRT_ROOT=<dir with nvinfer_10.dll, nvinfer_10.lib, include/>).
if(WIN32)
    set(AJI_TRT_ROOT "" CACHE PATH "TensorRT root")
    set(AJI_TRT_INCLUDE "${AJI_TRT_ROOT}/include")
    set(AJI_TRT_LIB "" CACHE PATH "TensorRT import library directory (empty uses AJI_TRT_ROOT)")
    if(NOT AJI_TRT_LIB)
        set(AJI_TRT_LIB "${AJI_TRT_ROOT}")
    endif()
    set(AJI_NVINFER "nvinfer_10" CACHE STRING
        "nvinfer import library name (nvinfer_10 / nvinfer_11)")
else()
    set(AJI_TRT_ROOT "$ENV{HOME}/sdk/tensorrt/usr" CACHE PATH "TensorRT root")
    set(AJI_TRT_INCLUDE "${AJI_TRT_ROOT}/include/x86_64-linux-gnu")
    set(AJI_TRT_LIB "${AJI_TRT_ROOT}/lib/x86_64-linux-gnu")
    set(AJI_NVINFER nvinfer)
endif()

find_package(CUDAToolkit REQUIRED)

# Library names the TensorRT backends import, for preloading them from the
# shared runtime packs that aji-runtime.json names (core/runtime_packs).
# They follow the SDKs this build uses.
if(WIN32)
    set(_pack_cudart "cudart64_${CUDAToolkit_VERSION_MAJOR}.dll")
    set(_pack_nvinfer "${AJI_NVINFER}.dll")
else()
    set(_pack_cudart "libcudart.so.${CUDAToolkit_VERSION_MAJOR}")
    set(_trt_major "")
    if(EXISTS "${AJI_TRT_INCLUDE}/NvInferVersion.h")
        file(STRINGS "${AJI_TRT_INCLUDE}/NvInferVersion.h" _trt_ver
             REGEX "#define (NV_TENSORRT_MAJOR|TRT_MAJOR_ENTERPRISE) +[0-9]+")
        string(REGEX MATCH "(NV_TENSORRT_MAJOR|TRT_MAJOR_ENTERPRISE) +([0-9]+)" _m "${_trt_ver}")
        set(_trt_major "${CMAKE_MATCH_2}")
    endif()
    if(_trt_major STREQUAL "")
        message(WARNING "TensorRT major version not found in ${AJI_TRT_INCLUDE}/NvInferVersion.h; "
                        "aji-runtime.json can't preload nvinfer")
        set(_pack_nvinfer "")
    else()
        set(_pack_nvinfer "lib${AJI_NVINFER}.so.${_trt_major}")
    endif()
endif()
set(AJI_PACK_DEFS AJI_PACK_CUDART="${_pack_cudart}" AJI_PACK_NVINFER="${_pack_nvinfer}")

# aji: thin dispatcher (what the mpv filter loads as lib=). Parses the
# conf's [global] backend key and forwards the C ABI to the sibling
# backend library. No CUDA/TRT/ORT dependencies of its own.
add_library(aji SHARED src/aji_dispatch.cpp src/aji_conf.cpp src/core/fileio.cpp
    src/core/runtime_packs.cpp src/aji_img_dispatch.cpp src/aji_held.cpp src/core/held.cpp)
target_compile_definitions(aji PRIVATE ${AJI_PACK_DEFS})
target_include_directories(aji PUBLIC include)
target_link_libraries(aji PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(aji PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden)

# The TensorRT-independent part of the TensorRT backends (conf parsing, ONNX
# shape probing, file I/O, image tiling, the CUDA kernels), compiled once
# and linked into both aji_trt and aji_trtrtx. Nothing in it depends on
# the TensorRT flavour (AJI_TRTRTX) or includes TensorRT headers.
set(AJI_HALF_SOURCES src/core/half_array.cpp)
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
    list(APPEND AJI_HALF_SOURCES src/core/half_f16c.cpp)
    # Keep AVX instructions out of the baseline CPU-dispatch translation unit.
    set_source_files_properties(src/core/half_array.cpp PROPERTIES
        COMPILE_DEFINITIONS AJI_HAVE_F16C=1)
    set_source_files_properties(src/core/half_f16c.cpp PROPERTIES
        COMPILE_OPTIONS /arch:AVX)
endif()
add_library(aji_trt_common OBJECT src/aji_conf.cpp src/onnx_shape.cpp
    src/core/fileio.cpp src/core/runtime_packs.cpp src/core/tiling.cpp src/core/tiling_gpu.cu src/kernels.cu
    ${AJI_HALF_SOURCES})
target_include_directories(aji_trt_common PRIVATE include)
set_target_properties(aji_trt_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden)

# aji_trt: TensorRT/CUDA backend (the former aji.dll).
add_library(aji_trt SHARED src/aji_trt.cpp src/core/trt_engine.cpp
    src/aji_img_trt.cpp $<TARGET_OBJECTS:aji_trt_common>)
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

# Optional observation module: neither aji nor aji_trt imports it. Its model
# parser, allocations and kernels are loaded only for an addon subscription,
# and unloaded with its last observer.
option(AJI_BUILD_ANALYSIS "Build the optional Windows addon observation module" OFF)
if(AJI_BUILD_ANALYSIS)
    if(NOT WIN32)
        message(FATAL_ERROR "AJI_BUILD_ANALYSIS currently supports Windows only")
    endif()
    string(REPLACE "nvinfer" "nvonnxparser" AJI_NVONNXPARSER "${AJI_NVINFER}")
    # aji_analysis: what the player loads. It imports no CUDA or TensorRT
    # library: it loads cudart/nvinfer/the ONNX parser (from the shared
    # runtime packs when aji-runtime.json names them, like aji.dll does for
    # aji_trt), then the observer aji_analysis_trt, and forwards to it.
    add_library(aji_analysis SHARED src/aji_analysis_loader.cpp
        src/core/fileio.cpp src/core/runtime_packs.cpp)
    target_compile_definitions(aji_analysis PRIVATE ${AJI_PACK_DEFS}
        AJI_PACK_NVONNXPARSER="${AJI_NVONNXPARSER}.dll")
    target_include_directories(aji_analysis PUBLIC include)
    set_target_properties(aji_analysis PROPERTIES CXX_VISIBILITY_PRESET hidden)
    add_library(aji_analysis_trt SHARED src/aji_analysis.cpp src/analysis_kernels.cu src/kernels.cu
        src/core/fileio.cpp src/core/runtime_packs.cpp)
    target_include_directories(aji_analysis_trt PUBLIC include PRIVATE ${AJI_TRT_INCLUDE})
    target_link_directories(aji_analysis_trt PRIVATE ${AJI_TRT_LIB})
    target_link_libraries(aji_analysis_trt PRIVATE ${AJI_NVINFER} ${AJI_NVONNXPARSER} CUDA::cuda_driver CUDA::cudart bcrypt)
    set_target_properties(aji_analysis_trt PROPERTIES CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)
    add_dependencies(aji_analysis aji_analysis_trt)
endif()
if(NOT WIN32)
    # backends export the same aji_* names as the dispatcher; bind their
    # internal references locally so the global (dispatcher) symbols
    # can't interpose
    target_link_options(aji_trt PRIVATE -Wl,-Bsymbolic)
endif()

# aji_trtrtx: the same backend sources built against TensorRT for RTX
# (AJI_TRTRTX): engines build in-process through the builder API (no
# trtexec) and cache under their own names. Optional: skipped unless
# -DAJI_TRTRTX_ROOT=<SDK root> is given. The import library names carry
# the SDK's major/minor on Windows (tensorrt_rtx_1_7.lib), read from its
# NvInferVersion.h so a new SDK needs no extra option.
set(AJI_TRTRTX_ROOT "" CACHE PATH "TensorRT for RTX root (include/ + libs)")
if(AJI_TRTRTX_ROOT)
    # tolerate a backslashed Windows path from a .bat
    file(TO_CMAKE_PATH "${AJI_TRTRTX_ROOT}" _rtx_root)
    if(NOT EXISTS "${_rtx_root}/include/NvInferVersion.h")
        message(FATAL_ERROR "AJI_TRTRTX_ROOT=${AJI_TRTRTX_ROOT}: no TensorRT-RTX SDK "
                            "there (missing include/NvInferVersion.h). See "
                            "docs/BUILD-WINDOWS.md, 2b.")
    endif()
    file(STRINGS "${_rtx_root}/include/NvInferVersion.h" _rtx_ver
         REGEX "#define TRT_(MAJOR|MINOR)_RTX ")
    string(REGEX MATCH "TRT_MAJOR_RTX ([0-9]+)" _m "${_rtx_ver}")
    set(_rtx_major "${CMAKE_MATCH_1}")
    string(REGEX MATCH "TRT_MINOR_RTX ([0-9]+)" _m "${_rtx_ver}")
    set(_rtx_minor "${CMAKE_MATCH_1}")
    if(_rtx_major STREQUAL "" OR _rtx_minor STREQUAL "")
        message(FATAL_ERROR "AJI_TRTRTX_ROOT: no TensorRT-RTX NvInferVersion.h "
                            "in ${_rtx_root}/include")
    endif()
    if(WIN32)
        set(AJI_TRTRTX_LIB "${_rtx_root}")
        set(_rtx_libs tensorrt_rtx_${_rtx_major}_${_rtx_minor}
                      tensorrt_onnxparser_rtx_${_rtx_major}_${_rtx_minor})
    else()
        set(AJI_TRTRTX_LIB "${_rtx_root}/lib")
        set(_rtx_libs tensorrt_rtx tensorrt_onnxparser_rtx)
    endif()
    if(WIN32)
        target_compile_definitions(aji PRIVATE
            AJI_PACK_TRTRTX="tensorrt_rtx_${_rtx_major}_${_rtx_minor}.dll"
            AJI_PACK_TRTRTX_PARSER="tensorrt_onnxparser_rtx_${_rtx_major}_${_rtx_minor}.dll")
    else()
        target_compile_definitions(aji PRIVATE
            AJI_PACK_TRTRTX="libtensorrt_rtx.so.${_rtx_major}"
            AJI_PACK_TRTRTX_PARSER="libtensorrt_onnxparser_rtx.so.${_rtx_major}")
    endif()
    message(STATUS "aji_trtrtx: TensorRT-RTX ${_rtx_major}.${_rtx_minor} "
                   "(${_rtx_libs})")
    add_library(aji_trtrtx SHARED src/aji_trt.cpp src/core/trt_engine.cpp
        src/aji_img_trt.cpp $<TARGET_OBJECTS:aji_trt_common>)
    target_compile_definitions(aji_trtrtx PRIVATE AJI_TRTRTX)
    target_include_directories(aji_trtrtx
        PUBLIC include
        PRIVATE "${_rtx_root}/include")
    target_link_directories(aji_trtrtx PRIVATE ${AJI_TRTRTX_LIB})
    target_link_libraries(aji_trtrtx PRIVATE ${_rtx_libs}
        CUDA::cuda_driver CUDA::cudart)
    set_target_properties(aji_trtrtx PROPERTIES
        BUILD_RPATH "${AJI_TRTRTX_LIB}"
        INSTALL_RPATH "${AJI_TRTRTX_LIB}"
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden)
    if(NOT WIN32)
        target_link_options(aji_trtrtx PRIVATE -Wl,-Bsymbolic)
    endif()
else()
    message(STATUS "AJI_TRTRTX_ROOT not set; skipping aji_trtrtx target")
endif()

# aji_image: image-ABI CLI harness (PNM in/out); any backend via aji.dll.
add_executable(aji_image src/aji_image_cli.cpp $<$<BOOL:${WIN32}>:src/utf8-argv.manifest>)
target_link_libraries(aji_image PRIVATE aji)
add_dependencies(aji_image aji_trt)
if(WIN32)
    add_dependencies(aji_image aji_dml)
endif()
if(TARGET aji_trtrtx)
    add_dependencies(aji_image aji_trtrtx)
endif()

# The .manifest embeds <activeCodePage>UTF-8</activeCodePage> so argv/fopen
# carry non-ASCII paths intact; benchmark.ps1 runs this from the installed
# package, which may live under a non-ASCII path.
add_executable(aji_harness src/harness.c $<$<BOOL:${WIN32}>:src/utf8-argv.manifest>)
target_link_libraries(aji_harness PRIVATE aji CUDA::cudart)
add_dependencies(aji_harness aji_trt)
if(TARGET aji_trtrtx)
    add_dependencies(aji_harness aji_trtrtx)
endif()
set_target_properties(aji_harness PROPERTIES
    BUILD_RPATH "${AJI_TRT_LIB};$<TARGET_FILE_DIR:aji>")

# aji_encode: offline file->file upscale/interpolate CLI. aji_harness's
# libaji call sequence + real libav demux/decode/encode/mux, on TensorRT or
# DirectML (Windows). Built only when the ffmpeg/libav development libraries are present;
# on Windows pass -DAJI_FFMPEG_ROOT=<dir with include/ and lib/>.
if(WIN32)
    set(AJI_FFMPEG_ROOT "" CACHE PATH "ffmpeg dev root (include/, lib/)")
    if(AJI_FFMPEG_ROOT)
        # Same UTF-8 argv manifest as aji_harness above.
        add_executable(aji_encode src/encode.c src/utf8-argv.manifest)
        target_include_directories(aji_encode PRIVATE "${AJI_FFMPEG_ROOT}/include")
        target_link_directories(aji_encode PRIVATE "${AJI_FFMPEG_ROOT}/lib")
        target_link_libraries(aji_encode PRIVATE aji CUDA::cudart CUDA::cuda_driver
            avformat avcodec avutil avfilter swscale
            dxgi dxguid)   # AMF adapter selection
        # --backend directml makes no CUDA call; delay-loading the CUDA DLLs
        # lets aji_encode start on machines without the NVIDIA driver
        # (nvcuda.dll) or the TensorRT component pack (cudart).
        target_link_options(aji_encode PRIVATE /DELAYLOAD:nvcuda.dll
            /DELAYLOAD:cudart64_${CUDAToolkit_VERSION_MAJOR}.dll)
        target_link_libraries(aji_encode PRIVATE delayimp)
        add_dependencies(aji_encode aji_trt)
        set_target_properties(aji_encode PROPERTIES
            BUILD_RPATH "${AJI_TRT_LIB};$<TARGET_FILE_DIR:aji>")
    else()
        message(STATUS "AJI_FFMPEG_ROOT not set; skipping aji_encode target")
    endif()
else()
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFMPEG IMPORTED_TARGET
            libavformat libavcodec libavutil libavfilter libswscale)
    endif()
    if(FFMPEG_FOUND)
        find_package(Threads REQUIRED)    # --preview-session's stdin reader
        add_executable(aji_encode src/encode.c)
        target_link_libraries(aji_encode PRIVATE
            aji CUDA::cudart CUDA::cuda_driver PkgConfig::FFMPEG Threads::Threads)
        add_dependencies(aji_encode aji_trt)
        set_target_properties(aji_encode PROPERTIES
            BUILD_RPATH "${AJI_TRT_LIB};$<TARGET_FILE_DIR:aji>")
    else()
        message(STATUS "ffmpeg/libav not found via pkg-config; skipping aji_encode target")
    endif()
endif()

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
    add_library(aji_dml SHARED src/aji_dml.cpp src/aji_conf.cpp src/onnx_shape.cpp
        src/core/fileio.cpp src/core/ort_dml.cpp
        src/aji_img_dml.cpp src/core/tiling.cpp ${AJI_HALF_SOURCES})
    target_include_directories(aji_dml
        PUBLIC include
        PRIVATE "${AJI_ORT_ROOT}/build/native/include"
                "${AJI_DML_ROOT}/include")
    target_link_libraries(aji_dml PRIVATE d3d12 d3d11 dxgi d3dcompiler)
    set_target_properties(aji_dml PROPERTIES CXX_VISIBILITY_PRESET hidden)

    add_executable(aji_harness_dml src/harness_dml.cpp src/utf8-argv.manifest)
    target_link_libraries(aji_harness_dml PRIVATE aji d3d11)
    add_dependencies(aji_harness_dml aji_dml)
    add_executable(aji_scene_dml_test sanity/scene_dml.cpp)
    target_link_libraries(aji_scene_dml_test PRIVATE aji d3d11)
    add_dependencies(aji_scene_dml_test aji_dml)

    # aji_frame_diff on DirectML, cross-checked bit for bit against
    # kernels.cu where CUDA works (cudart is delay-loaded, so it runs on any
    # DirectML GPU). aji_dml loads ONNX Runtime and DirectML from its own
    # folder, so the test runs from dml-test/ beside copies of the three.
    add_executable(aji_frame_diff_dml_test sanity/frame_diff_dml.cpp src/kernels.cu)
    target_include_directories(aji_frame_diff_dml_test PRIVATE include src)
    target_link_libraries(aji_frame_diff_dml_test PRIVATE aji d3d11 dxgi
        CUDA::cudart delayimp)
    target_compile_definitions(aji_frame_diff_dml_test PRIVATE
        AJI_CUDART_DLL="cudart64_${CUDAToolkit_VERSION_MAJOR}.dll")
    target_link_options(aji_frame_diff_dml_test PRIVATE
        /DELAYLOAD:cudart64_${CUDAToolkit_VERSION_MAJOR}.dll)
    set_target_properties(aji_frame_diff_dml_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dml-test")
    set(_fd_runtime $<TARGET_FILE:aji> $<TARGET_FILE:aji_dml>)
    foreach(dll IN ITEMS "${AJI_ORT_ROOT}/runtimes/win-x64/native/onnxruntime.dll"
                         "${AJI_DML_ROOT}/bin/x64-win/DirectML.dll")
        if(EXISTS "${dll}")
            list(APPEND _fd_runtime "${dll}")
        endif()
    endforeach()
    # a target of its own, run on every build: a rebuilt aji_dml.dll must
    # reach dml-test/ even when the test itself doesn't relink
    add_custom_target(aji_frame_diff_dml_runtime ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/dml-test"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_fd_runtime}
                "${CMAKE_BINARY_DIR}/dml-test"
        DEPENDS aji aji_dml)
    add_dependencies(aji_frame_diff_dml_test aji_frame_diff_dml_runtime)
endif()

# Parity test tool: compiles the kernels directly (the shared lib hides
# the launcher symbols), no TensorRT dependency.
add_executable(aji_kernel_test src/kernel_test.c src/kernels.cu)
target_include_directories(aji_kernel_test PRIVATE include src)
target_link_libraries(aji_kernel_test PRIVATE CUDA::cudart)

# Exact parity of the fused NV12/P010 preprocessor against its former passes.
# The test includes kernels.cu to inspect the private resampling plan.
add_executable(aji_pre_fusion_test sanity/pre_fusion.cu)
target_include_directories(aji_pre_fusion_test PRIVATE include src)
target_link_libraries(aji_pre_fusion_test PRIVATE CUDA::cudart)

# Image-ABI conversion and tiling checks: CPU only, no GPU or models.
add_executable(aji_half_test sanity/half_array_test.cpp ${AJI_HALF_SOURCES})
target_include_directories(aji_half_test PRIVATE src)
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
    add_executable(aji_half_benchmark EXCLUDE_FROM_ALL
        experimental/image-half/half-profile.cpp ${AJI_HALF_SOURCES})
    target_include_directories(aji_half_benchmark PRIVATE src)
endif()
add_executable(aji_tiling_test sanity/tiling_test.cpp src/core/tiling.cpp)
target_include_directories(aji_tiling_test PRIVATE include src)

add_executable(aji_runtime_packs_test sanity/runtime_packs_test.cpp
    src/core/runtime_packs.cpp src/core/fileio.cpp)
target_include_directories(aji_runtime_packs_test PRIVATE include src)

# The held-frame reuse rule (aji_held_*) against synthetic comparisons.
add_executable(aji_held_test sanity/held_test.cpp src/core/held.cpp)
target_include_directories(aji_held_test PRIVATE include src)
target_link_libraries(aji_runtime_packs_test PRIVATE ${CMAKE_DL_LIBS})

add_executable(aji_msb444_test sanity/msb444.cu src/kernels.cu)
target_include_directories(aji_msb444_test PRIVATE include src)
target_link_libraries(aji_msb444_test PRIVATE CUDA::cudart)

# aji_frame_diff's tile and region differences (held-frame reuse).
add_executable(aji_frame_diff_test sanity/frame_diff.cu src/kernels.cu)
target_include_directories(aji_frame_diff_test PRIVATE include src)
target_link_libraries(aji_frame_diff_test PRIVATE CUDA::cudart)

# Regression for concurrent first CUDA runtime loading in separate DLLs.
if(WIN32)
    foreach(probe IN ITEMS aji_cuda_init_a aji_cuda_init_b)
        add_library(${probe} SHARED EXCLUDE_FROM_ALL sanity/cuda_init.cu)
        target_include_directories(${probe} PRIVATE include)
        target_link_libraries(${probe} PRIVATE CUDA::cudart)
    endforeach()
    add_executable(aji_cuda_init_test EXCLUDE_FROM_ALL sanity/cuda_init.cpp)
    add_dependencies(aji_cuda_init_test aji_cuda_init_a aji_cuda_init_b)
endif()

# Self-contained tests for ctest. CI runs `ctest -L cpu` (no GPU on hosted
# runners); `ctest -L gpu` needs an NVIDIA GPU. aji_kernel_test, the harnesses
# and the sanity/*.py drivers take models or fixtures and are run by hand.
enable_testing()
add_executable(aji_rife_conf_test sanity/rife_conf_test.cpp src/aji_conf.cpp)
target_include_directories(aji_rife_conf_test PRIVATE src)
foreach(test IN ITEMS aji_half_test aji_tiling_test aji_runtime_packs_test aji_rife_conf_test
                     aji_held_test)
    add_test(NAME ${test} COMMAND ${test})
    set_tests_properties(${test} PROPERTIES LABELS cpu)
endforeach()
foreach(test IN ITEMS aji_pre_fusion_test aji_msb444_test aji_frame_diff_test)
    add_test(NAME ${test} COMMAND ${test})
    set_tests_properties(${test} PROPERTIES LABELS gpu)
endforeach()
if(WIN32)
    # any DirectML GPU; the CUDA cross-check is skipped without an NVIDIA one
    add_test(NAME aji_frame_diff_dml_test COMMAND aji_frame_diff_dml_test)
    set_tests_properties(aji_frame_diff_dml_test PROPERTIES LABELS gpu)
endif()
