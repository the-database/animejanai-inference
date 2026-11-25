cmake_minimum_required(VERSION 3.24)
# AJI_CUDA=OFF builds what needs no CUDA toolkit: the dispatcher, aji_dml (Windows), the
# host-only half of the native path (ONNX reader, lowering, layout) and the CPU tests. For
# machines without CUDA (Linux on AMD/Intel, Android) and the Vulkan backend's development.
option(AJI_CUDA "Build the CUDA backends (aji_trt, aji_trtrtx), the native CUDA kernels and their tests" ON)
# AJI_VULKAN=ON compiles the Vulkan compute shaders (src/vk/shaders) with glslc and embeds them.
option(AJI_VULKAN "Compile and embed the Vulkan compute shaders (needs glslc)" OFF)
# Select the default before CUDA is enabled, which initializes it from nvcc's default GPU.
# Preserve explicit -D overrides and CMake's CUDAARCHS environment override.
if(AJI_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
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
project(animejanai-inference LANGUAGES C CXX)
if(AJI_CUDA)
    enable_language(CUDA)
endif()

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

if(AJI_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Library names the TensorRT backends import, for preloading them from the
# shared runtime packs that aji-runtime.json names (core/runtime_packs).
# They follow the SDKs this build uses. Without CUDA there are no TensorRT
# backends to preload for: both names are empty.
if(NOT AJI_CUDA)
    set(_pack_cudart "")
    set(_pack_nvinfer "")
elseif(WIN32)
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

# The host-only half of the native path (no CUDA): the ONNX reader, the lowering to a Plan,
# its layout for one size (native/layout) and the weight packing. Compiled into aji_native for
# the CUDA backends, and on its own (aji_native_host) for the CPU tests and the builds without
# CUDA.
set(AJI_NATIVE_HOST_SOURCES src/native/onnx_graph.cpp src/native/lower.cpp
    src/native/layout.cpp src/native/pack.cpp src/native/rife_plan.cpp)
add_library(aji_native_host OBJECT ${AJI_NATIVE_HOST_SOURCES})
target_include_directories(aji_native_host PRIVATE src)
set_target_properties(aji_native_host PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden)

if(AJI_CUDA)
add_library(aji_trt_common OBJECT src/aji_conf.cpp src/onnx_shape.cpp
    src/core/fileio.cpp src/core/runtime_packs.cpp src/core/tiling.cpp src/core/tiling_gpu.cu src/kernels.cu
    ${AJI_HALF_SOURCES})
target_include_directories(aji_trt_common PRIVATE include)
set_target_properties(aji_trt_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden)

# Native tensor-core kernels and their ONNX lowering/executor (src/native), linked into both
# TensorRT backends (used with [global] native=on) and the native tests.
add_library(aji_native OBJECT src/native/span48_project.cu src/native/span_fp8.cu src/native/conv.cu src/native/gconv.cu src/native/tfm.cu src/native/rife.cpp src/native/rife_kernels.cu src/native/esrgan.cpp src/native/tail.cpp src/native/fdat.cpp src/native/fdat_body.cpp src/native/fdat_kernels.cu src/native/tfdat.cpp src/native/tfdat_kernels.cu src/native/hat.cpp src/native/dat.cpp src/native/dat_kernels.cu src/native/sst.cpp src/native/sst_kernels.cu src/native/tfix.cpp src/native/tfix_kernels.cu src/native/img_kernels.cu
    ${AJI_NATIVE_HOST_SOURCES} src/native/executor.cpp src/native/framenet.cpp)
target_include_directories(aji_native PRIVATE src)
target_link_libraries(aji_native PRIVATE CUDA::cudart)
# The kernels need sm_80+ (device_supported refuses older GPUs before any launch), so skip
# sm_75, and the datacenter Hopper/Blackwell targets the players never see: real code for the
# GeForce generations plus compute_80 PTX, which those and future GPUs JIT. Each target is a
# full copy of every kernel (~1.3 MB).
set(AJI_NATIVE_CUDA_ARCHITECTURES 80-real 86-real 89-real 120-real 80-virtual
    CACHE STRING "CUDA architectures for the native kernels (src/native)")
set_target_properties(aji_native PROPERTIES
    CUDA_ARCHITECTURES "${AJI_NATIVE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden)

# aji_trt: TensorRT/CUDA backend (the former aji.dll).
add_library(aji_trt SHARED src/aji_trt.cpp src/core/trt_engine.cpp
    src/aji_img_trt.cpp $<TARGET_OBJECTS:aji_trt_common> $<TARGET_OBJECTS:aji_native>)
target_include_directories(aji_trt
    PUBLIC include
    PRIVATE ${AJI_TRT_INCLUDE} src)
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
        src/aji_img_trt.cpp $<TARGET_OBJECTS:aji_trt_common> $<TARGET_OBJECTS:aji_native>)
    target_compile_definitions(aji_trtrtx PRIVATE AJI_TRTRTX)
    target_include_directories(aji_trtrtx
        PUBLIC include
        PRIVATE "${_rtx_root}/include" src)
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
else()
    message(STATUS "AJI_CUDA=OFF: skipping aji_trt, aji_trtrtx, the native CUDA kernels, "
                   "aji_harness, aji_encode and the CUDA tests")
endif()  # AJI_CUDA

# aji_image: image-ABI CLI harness (PNM in/out); any backend via aji.dll.
add_executable(aji_image src/aji_image_cli.cpp $<$<BOOL:${WIN32}>:src/utf8-argv.manifest>)
target_link_libraries(aji_image PRIVATE aji)
if(TARGET aji_trt)
    add_dependencies(aji_image aji_trt)
endif()
if(WIN32)
    add_dependencies(aji_image aji_dml)
endif()
if(TARGET aji_trtrtx)
    add_dependencies(aji_image aji_trtrtx)
endif()

if(AJI_CUDA)
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
        # CUDA is loaded when first used (encode_cuda_dl.h), as /DELAYLOAD does on Windows: the
        # binary starts without the NVIDIA driver (--backend vulkan on AMD / Intel)
        target_compile_definitions(aji_encode PRIVATE AJI_ENCODE_CUDA_DL=1
            AJI_CUDART_SOVER="${CUDAToolkit_VERSION_MAJOR}")
        target_include_directories(aji_encode SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(aji_encode PRIVATE
            aji ${CMAKE_DL_LIBS} PkgConfig::FFMPEG Threads::Threads)
        add_dependencies(aji_encode aji_trt)
        set_target_properties(aji_encode PROPERTIES
            BUILD_RPATH "${AJI_TRT_LIB};$<TARGET_FILE_DIR:aji>")
    else()
        message(STATUS "ffmpeg/libav not found via pkg-config; skipping aji_encode target")
    endif()
endif()
endif()  # AJI_CUDA: aji_harness, aji_encode

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
    target_link_libraries(aji_harness_dml PRIVATE aji d3d11 dxgi)
    add_dependencies(aji_harness_dml aji_dml)
    add_executable(aji_scene_dml_test sanity/scene_dml.cpp)
    target_link_libraries(aji_scene_dml_test PRIVATE aji d3d11 dxgi)
    add_dependencies(aji_scene_dml_test aji_dml)

    # aji_frame_diff on DirectML, cross-checked bit for bit against
    # kernels.cu where CUDA works (cudart is delay-loaded, so it runs on any
    # DirectML GPU). aji_dml loads ONNX Runtime and DirectML from its own
    # folder, so the test runs from dml-test/ beside copies of the three.
    # It compiles kernels.cu, so it needs AJI_CUDA.
  if(AJI_CUDA)
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
endif()

if(AJI_CUDA)
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
endif()  # AJI_CUDA

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

# The held-frame reuse rule against synthetic comparisons, and its C ABI
# (src/aji_held.cpp) over a stub context.
add_executable(aji_held_test sanity/held_test.cpp src/core/held.cpp src/aji_held.cpp
    src/core/fileio.cpp)
target_include_directories(aji_held_test PRIVATE include src)
target_link_libraries(aji_held_test PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(aji_runtime_packs_test PRIVATE ${CMAKE_DL_LIBS})

# Native path, host only: the packing, the ONNX lowering, the layout and the export forms, on the CPU.
add_executable(aji_native_pack_test sanity/native_pack_test.cpp src/native/pack.cpp)
target_include_directories(aji_native_pack_test PRIVATE src)
add_executable(aji_native_lower_test sanity/native_lower_test.cpp $<TARGET_OBJECTS:aji_native_host>)
target_include_directories(aji_native_lower_test PRIVATE src)
add_executable(aji_native_layout_test sanity/native_layout_test.cpp $<TARGET_OBJECTS:aji_native_host>)
target_include_directories(aji_native_layout_test PRIVATE src)
add_executable(aji_native_forms_test sanity/native_forms_test.cpp $<TARGET_OBJECTS:aji_native_host>)
target_include_directories(aji_native_forms_test PRIVATE src)

if(AJI_CUDA)
add_executable(aji_msb444_test sanity/msb444.cu src/kernels.cu)
target_include_directories(aji_msb444_test PRIVATE include src)
target_link_libraries(aji_msb444_test PRIVATE CUDA::cudart)

# aji_frame_diff's tile and region differences (held-frame reuse).
add_executable(aji_frame_diff_test sanity/frame_diff.cu src/kernels.cu)
target_include_directories(aji_frame_diff_test PRIVATE include src)
target_link_libraries(aji_frame_diff_test PRIVATE CUDA::cudart)

# Native kernels (aji_native, above): the kernels against a CPU reference on the GPU.
add_executable(aji_native_conv_test sanity/native_conv_test.cu $<TARGET_OBJECTS:aji_native>)
target_include_directories(aji_native_conv_test PRIVATE src)
target_link_libraries(aji_native_conv_test PRIVATE CUDA::cudart)
# Real-model gate reuse regression. Pass the three ONNX paths when running on a CUDA GPU.
add_executable(aji_native_gate_reuse_test sanity/native_gate_reuse_test.cpp $<TARGET_OBJECTS:aji_native>)
target_include_directories(aji_native_gate_reuse_test PRIVATE src)
target_link_libraries(aji_native_gate_reuse_test PRIVATE CUDA::cudart)
add_executable(aji_native_tfm_test sanity/native_tfm_test.cu $<TARGET_OBJECTS:aji_native>)
target_include_directories(aji_native_tfm_test PRIVATE src)
target_link_libraries(aji_native_tfm_test PRIVATE CUDA::cudart)
add_executable(aji_native_gconv_test sanity/native_gconv_test.cu $<TARGET_OBJECTS:aji_native>)
target_include_directories(aji_native_gconv_test PRIVATE src)
target_link_libraries(aji_native_gconv_test PRIVATE CUDA::cudart)
# Native image model on a raw tile (parity against an fp32 reference, timing); built on request.
add_executable(aji_native_img_run EXCLUDE_FROM_ALL sanity/native_img_run.cu $<TARGET_OBJECTS:aji_native>)
target_include_directories(aji_native_img_run PRIVATE src)
target_link_libraries(aji_native_img_run PRIVATE CUDA::cudart)
# The native ONNX reader and builders against malformed models (unit cases; with model paths, mutants).
add_executable(aji_native_onnx_test sanity/native_onnx_test.cu $<TARGET_OBJECTS:aji_native>)
target_include_directories(aji_native_onnx_test PRIVATE src)
target_link_libraries(aji_native_onnx_test PRIVATE CUDA::cudart)
add_executable(aji_native_vid_run EXCLUDE_FROM_ALL sanity/native_vid_run.cu $<TARGET_OBJECTS:aji_native>)
target_include_directories(aji_native_vid_run PRIVATE src)
target_link_libraries(aji_native_vid_run PRIVATE CUDA::cudart)
# Native RIFE on a raw input tensor (parity against an fp32 reference, timing); built on request.
add_executable(aji_native_rife_run EXCLUDE_FROM_ALL sanity/native_rife_run.cu $<TARGET_OBJECTS:aji_native>)
target_include_directories(aji_native_rife_run PRIVATE src)
target_link_libraries(aji_native_rife_run PRIVATE CUDA::cudart)
# Whole-model timing on prepared data (experimental/native-span); built on request.
add_executable(aji_native_bench EXCLUDE_FROM_ALL experimental/native-span/lib/native_bench.cu $<TARGET_OBJECTS:aji_native>)
target_include_directories(aji_native_bench PRIVATE src)
target_link_libraries(aji_native_bench PRIVATE CUDA::cudart)

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
endif()  # AJI_CUDA: CUDA tests

# Vulkan compute shaders (src/vk/shaders), compiled to SPIR-V by glslc and embedded in a
# generated source (cmake/aji_spirv.cmake). Shipped libraries embed them, so glslc is a build
# tool only; a missing glslc is an error, never a silent skip.
if(AJI_VULKAN)
    find_program(AJI_GLSLC glslc HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin")
    if(NOT AJI_GLSLC)
        message(FATAL_ERROR "AJI_VULKAN=ON needs glslc (the Vulkan SDK, shaderc, or MSYS2's "
                            "mingw-w64-ucrt-x86_64-shaderc); pass -DAJI_GLSLC=<path>")
    endif()
    message(STATUS "glslc: ${AJI_GLSLC}")
    # glslc processes run at once (Ninja job pool; 0: no limit), for small CI machines
    set(AJI_GLSLC_JOBS 0 CACHE STRING "parallel glslc jobs (0: no limit)")
    if(AJI_GLSLC_JOBS GREATER 0)
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS aji_glslc=${AJI_GLSLC_JOBS})
    endif()
    include(cmake/aji_spirv.cmake)
    aji_spirv_embed(aji_vk_shaders src/vk/shaders/shaders.txt)
    add_library(aji_vk_shaders OBJECT ${AJI_VK_SHADERS_SOURCE})
    target_include_directories(aji_vk_shaders PRIVATE src)
    set_target_properties(aji_vk_shaders PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden)
    add_executable(aji_vk_spirv_test sanity/vk_spirv_test.cpp $<TARGET_OBJECTS:aji_vk_shaders>)
    target_include_directories(aji_vk_spirv_test PRIVATE src)

    # aji_vk's device layer, kernels and executor (src/vk). The Vulkan headers come from the
    # Vulkan SDK or AJI_VK_INCLUDE; the loader is opened at run time, so nothing links Vulkan.
    find_path(AJI_VK_INCLUDE vulkan/vulkan_core.h HINTS "$ENV{VULKAN_SDK}/Include" "$ENV{VULKAN_SDK}/include")
    if(NOT AJI_VK_INCLUDE)
        message(FATAL_ERROR "AJI_VULKAN=ON needs the Vulkan headers: install the Vulkan SDK or pass "
                            "-DAJI_VK_INCLUDE=<dir containing vulkan/vulkan_core.h>")
    endif()
    add_library(aji_vk_core OBJECT src/vk/vk.cpp src/vk/device.cpp src/vk/kernels.cpp src/vk/executor.cpp
                src/vk/gconv.cpp src/vk/esrgan.cpp src/vk/rife.cpp src/vk/tfix.cpp src/vk/tnet.cpp src/vk/fdat.cpp src/vk/hat.cpp src/vk/dat.cpp src/vk/tfdat.cpp src/vk/frame.cpp src/core/fileio.cpp)
    target_include_directories(aji_vk_core PRIVATE src include)
    target_include_directories(aji_vk_core SYSTEM PRIVATE "${AJI_VK_INCLUDE}")
    set_target_properties(aji_vk_core PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden)
    # By hand, with a model: aji_vk_model_test <model.onnx> <w> <h> (see the source). With CUDA it
    # also compares against the CUDA native executor.
    if(AJI_CUDA)
        add_executable(aji_vk_model_test sanity/vk_model_test.cpp $<TARGET_OBJECTS:aji_vk_core>
                       $<TARGET_OBJECTS:aji_vk_shaders> $<TARGET_OBJECTS:aji_native>)
        target_compile_definitions(aji_vk_model_test PRIVATE AJI_VK_TEST_CUDA=1)
        target_link_libraries(aji_vk_model_test PRIVATE CUDA::cudart)
    else()
        add_executable(aji_vk_model_test sanity/vk_model_test.cpp $<TARGET_OBJECTS:aji_vk_core>
                       $<TARGET_OBJECTS:aji_vk_shaders> $<TARGET_OBJECTS:aji_native_host>)
    endif()
    target_include_directories(aji_vk_model_test PRIVATE src)
    target_include_directories(aji_vk_model_test SYSTEM PRIVATE "${AJI_VK_INCLUDE}")
    # By hand: raw NV12 / P010 frames through a model on Vulkan alone, no D3D (vk/frame.h)
    add_executable(aji_harness_vk src/harness_vk.cpp $<TARGET_OBJECTS:aji_vk_core> $<TARGET_OBJECTS:aji_vk_shaders>
                   $<TARGET_OBJECTS:aji_native_host>)
    target_include_directories(aji_harness_vk PRIVATE src include)
    target_include_directories(aji_harness_vk SYSTEM PRIVATE "${AJI_VK_INCLUDE}")
    if(UNIX)
        find_package(Threads REQUIRED)
        target_link_libraries(aji_harness_vk PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    endif()
    # By hand: the Vulkan-frames ABI (aji_create_vk) as a player uses it, a device of its own
    # standing in for the player's, through the dispatcher
    add_executable(aji_vk_frames_test sanity/vk_frames_test.cpp $<TARGET_OBJECTS:aji_vk_core> $<TARGET_OBJECTS:aji_vk_shaders>
                   $<TARGET_OBJECTS:aji_native_host>)
    target_include_directories(aji_vk_frames_test PRIVATE src include)
    target_include_directories(aji_vk_frames_test SYSTEM PRIVATE "${AJI_VK_INCLUDE}")
    target_link_libraries(aji_vk_frames_test PRIVATE aji)
    if(UNIX)
        target_link_libraries(aji_vk_frames_test PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    endif()
    # Every entry of the kernel table against the CPU reference, on the device's family (ctest -L gpu)
    add_executable(aji_vk_conv_test sanity/vk_conv_test.cpp $<TARGET_OBJECTS:aji_vk_core>
                   $<TARGET_OBJECTS:aji_vk_shaders> $<TARGET_OBJECTS:aji_native_host>)
    target_include_directories(aji_vk_conv_test PRIVATE src)
    target_include_directories(aji_vk_conv_test SYSTEM PRIVATE "${AJI_VK_INCLUDE}")
    if(UNIX)
        find_package(Threads REQUIRED)
        target_link_libraries(aji_vk_conv_test PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    endif()
    # The general conv's modes (ESRGAN, RIFE) against a CPU reference (ctest -L gpu)
    add_executable(aji_vk_gconv_test sanity/vk_gconv_test.cpp $<TARGET_OBJECTS:aji_vk_core>
                   $<TARGET_OBJECTS:aji_vk_shaders> $<TARGET_OBJECTS:aji_native_host>)
    target_include_directories(aji_vk_gconv_test PRIVATE src)
    target_include_directories(aji_vk_gconv_test SYSTEM PRIVATE "${AJI_VK_INCLUDE}")
    if(UNIX)
        target_link_libraries(aji_vk_gconv_test PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    endif()

    if(WIN32)
        # aji_vk (backend=Vulkan): aji_dml.cpp built with AJI_VULKAN. The D3D12
        # frame pipeline is DirectML's; each model that lowers to the native
        # kernels runs on the Vulkan executor on the same adapter, the others
        # on DirectML. Ships beside aji_dml with the same ONNX Runtime and
        # DirectML DLLs; Vulkan is loaded at run time. Its image ABI is
        # aji_img_vk.cpp (AJI_IMG_VULKAN), all Vulkan.
        add_library(aji_vk SHARED src/aji_dml.cpp src/aji_conf.cpp src/onnx_shape.cpp src/vk/video.cpp
            src/core/ort_dml.cpp src/aji_img_vk.cpp src/core/tiling.cpp ${AJI_HALF_SOURCES}
            $<TARGET_OBJECTS:aji_vk_core> $<TARGET_OBJECTS:aji_vk_shaders>
            $<TARGET_OBJECTS:aji_native_host>)
        target_include_directories(aji_vk
            PUBLIC include
            PRIVATE src "${AJI_ORT_ROOT}/build/native/include" "${AJI_DML_ROOT}/include")
        target_include_directories(aji_vk SYSTEM PRIVATE "${AJI_VK_INCLUDE}")
        target_compile_definitions(aji_vk PRIVATE AJI_VULKAN=1)
        target_link_libraries(aji_vk PRIVATE d3d12 d3d11 dxgi d3dcompiler)
        set_target_properties(aji_vk PROPERTIES CXX_VISIBILITY_PRESET hidden)
        add_dependencies(aji_harness_dml aji_vk)
    else()
        # libaji_vk.so (Linux, Android): the image ABI on Vulkan (aji_img_vk.cpp) and the video ABI
        # for Vulkan frames (aji_vk_frames.cpp: aji_create_vk, vk/video.h).
        find_package(Threads REQUIRED)
        add_library(aji_vk SHARED src/aji_img_vk.cpp src/aji_vk_frames.cpp src/vk/video.cpp src/aji_conf.cpp src/onnx_shape.cpp
            src/core/tiling.cpp ${AJI_HALF_SOURCES}
            $<TARGET_OBJECTS:aji_vk_core> $<TARGET_OBJECTS:aji_vk_shaders> $<TARGET_OBJECTS:aji_native_host>)
        target_include_directories(aji_vk PUBLIC include PRIVATE src)
        target_include_directories(aji_vk SYSTEM PRIVATE "${AJI_VK_INCLUDE}")
        target_link_libraries(aji_vk PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
        set_target_properties(aji_vk PROPERTIES CXX_VISIBILITY_PRESET hidden)
        target_link_libraries(aji_vk_model_test PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    endif()
    add_dependencies(aji_image aji_vk)
    # aji_encode --backend vulkan: libav's Vulkan hwcontext needs the Vulkan headers
    if(TARGET aji_encode)
        target_compile_definitions(aji_encode PRIVATE AJI_ENCODE_VULKAN=1)
        target_include_directories(aji_encode SYSTEM PRIVATE "${AJI_VK_INCLUDE}")
        add_dependencies(aji_encode aji_vk)
    endif()
endif()

# Self-contained tests for ctest. CI runs `ctest -L cpu` (no GPU on hosted
# runners); `ctest -L gpu` needs an NVIDIA GPU. aji_kernel_test, the harnesses
# and the sanity/*.py drivers take models or fixtures and are run by hand.
enable_testing()
add_executable(aji_frame_graph_test sanity/frame_graph_test.cpp)
target_include_directories(aji_frame_graph_test PRIVATE src include)
add_executable(aji_rife_conf_test sanity/rife_conf_test.cpp src/aji_conf.cpp)
target_include_directories(aji_rife_conf_test PRIVATE src)
add_executable(aji_span_profile_test sanity/span_profile_test.cpp src/aji_conf.cpp)
target_include_directories(aji_span_profile_test PRIVATE src)
set(_cpu_tests aji_frame_graph_test aji_half_test aji_tiling_test aji_runtime_packs_test aji_rife_conf_test aji_span_profile_test
               aji_held_test aji_native_pack_test aji_native_lower_test aji_native_layout_test aji_native_forms_test)
if(AJI_CUDA)
    # CPU-only, but compiled with nvcc
    list(APPEND _cpu_tests aji_native_onnx_test)
endif()
if(AJI_VULKAN)
    list(APPEND _cpu_tests aji_vk_spirv_test)
endif()
foreach(test IN LISTS _cpu_tests)
    add_test(NAME ${test} COMMAND ${test})
    set_tests_properties(${test} PROPERTIES LABELS cpu)
endforeach()
if(AJI_CUDA)
    foreach(test IN ITEMS aji_pre_fusion_test aji_msb444_test aji_frame_diff_test aji_native_conv_test aji_native_gconv_test aji_native_tfm_test)
        add_test(NAME ${test} COMMAND ${test})
        set_tests_properties(${test} PROPERTIES LABELS gpu)
    endforeach()
    if(WIN32)
        # any DirectML GPU; the CUDA cross-check is skipped without an NVIDIA one
        add_test(NAME aji_frame_diff_dml_test COMMAND aji_frame_diff_dml_test)
        set_tests_properties(aji_frame_diff_dml_test PROPERTIES LABELS gpu)
    endif()
endif()
if(AJI_VULKAN)
    # any Vulkan GPU with a kernel family (skips without one)
    add_test(NAME aji_vk_conv_test COMMAND aji_vk_conv_test)
    set_tests_properties(aji_vk_conv_test PROPERTIES LABELS gpu)
    add_test(NAME aji_vk_gconv_test COMMAND aji_vk_gconv_test)
    set_tests_properties(aji_vk_gconv_test PROPERTIES LABELS gpu)
endif()
