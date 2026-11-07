# Optional reproducible plugin/model bundle for Windows classic TensorRT 11.
option(AJI_BUILD_SPAN_PLUGINS "Build the experimental fixed-weight SPAN bundle" OFF)
set(AJI_SPAN_MODEL_DIR "" CACHE PATH "Directory containing the original V3.1 FP16 ONNX models")
if(NOT AJI_BUILD_SPAN_PLUGINS)
    return()
endif()
if(NOT WIN32)
    message(FATAL_ERROR "The SPAN plugin bundle currently builds on Windows only")
endif()
file(STRINGS "${AJI_TRT_INCLUDE}/NvInferVersion.h" _span_trt_major
    REGEX "^#define (NV_TENSORRT_MAJOR|TRT_MAJOR_ENTERPRISE)[ \t]+11([ \t]|$)")
if(NOT _span_trt_major)
    message(FATAL_ERROR "The SPAN plugin bundle requires TensorRT 11 headers")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(_span_cuda "${CUDAToolkit_BIN_DIR}" DIRECTORY)
set(_span_source "${CMAKE_CURRENT_SOURCE_DIR}/experimental/span-concat")
set(_span_extra)
if(CMAKE_CUDA_FLAGS MATCHES "allow-unsupported-compiler")
    list(APPEND _span_extra --allow-unsupported-compiler)
endif()
set(_span_dlls)
set(_span_models)
foreach(_kind balanced performance)
    if(_kind STREQUAL "balanced")
        set(_stock "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx")
        set(_export "2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Concat_Tiled3_v2.onnx")
        set(_dll "balanced-next-plugin.dll")
    else()
        set(_stock "2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx")
        set(_export "2x_AnimeJaNai_HD_V3.1_Performance_FP16_Concat_v1.onnx")
        set(_dll "performance-plugin.dll")
    endif()
    set(_input "${AJI_SPAN_MODEL_DIR}/${_stock}")
    if(NOT EXISTS "${_input}")
        message(FATAL_ERROR "Missing SPAN source model: ${_input}")
    endif()
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/span-plugins/${_kind}")
    add_custom_command(OUTPUT "${_out}/${_dll}" "${_out}/${_export}"
        COMMAND "${Python3_EXECUTABLE}" "${_span_source}/build.py"
            --kind "${_kind}" --model "${_input}" --cuda "${_span_cuda}"
            --trt-include "${AJI_TRT_INCLUDE}" --output "${_out}" ${_span_extra}
        DEPENDS "${_input}" "${_span_source}/build.py" "${_span_source}/prepare.py"
            "${_span_source}/${_kind}/plugin.cpp" "${_span_source}/${_kind}/kernels.cu"
        VERBATIM)
    list(APPEND _span_dlls "${_out}/${_dll}")
    list(APPEND _span_models "${_out}/${_export}" "${_input}")
endforeach()
add_custom_target(aji_span_plugins ALL
    DEPENDS ${_span_dlls} ${_span_models}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:aji_trt>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${_span_dlls} "$<TARGET_FILE_DIR:aji_trt>"
    VERBATIM)
add_dependencies(aji_trt aji_span_plugins)
install(TARGETS aji_trt RUNTIME DESTINATION inference COMPONENT experimental-span)
install(FILES ${_span_dlls} DESTINATION inference COMPONENT experimental-span)
# Include originals so unsupported devices/shapes/backends can use stock.
install(FILES ${_span_models} DESTINATION onnx COMPONENT experimental-span)
