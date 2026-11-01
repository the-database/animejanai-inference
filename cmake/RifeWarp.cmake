# Optional reproducible RIFE warp bundle. Stock RIFE remains the default.
option(AJI_BUILD_RIFE_WARP "Build the experimental RIFE warp bundle" OFF)
set(AJI_RIFE_MODEL_DIR "" CACHE PATH "Directory containing set-2 rife_v4.22.onnx")
if(NOT AJI_BUILD_RIFE_WARP)
    return()
endif()
if(NOT WIN32)
    message(FATAL_ERROR "The RIFE warp bundle currently builds on Windows only")
endif()
file(STRINGS "${AJI_TRT_INCLUDE}/NvInferVersion.h" _rife_trt_major
    REGEX "^#define (NV_TENSORRT_MAJOR|TRT_MAJOR_ENTERPRISE)[ \t]+11([ \t]|$)")
if(NOT _rife_trt_major)
    message(FATAL_ERROR "The RIFE warp bundle requires TensorRT 11 headers")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(_rife_cuda "${CUDAToolkit_BIN_DIR}" DIRECTORY)
set(_rife_source "${CMAKE_CURRENT_SOURCE_DIR}/experimental/rife-warp")
set(_rife_input "${AJI_RIFE_MODEL_DIR}/rife_v4.22.onnx")
if(NOT EXISTS "${_rife_input}")
    message(FATAL_ERROR "Missing set-2 RIFE source model: ${_rife_input}")
endif()
set(_rife_extra)
if(CMAKE_CUDA_FLAGS MATCHES "allow-unsupported-compiler")
    list(APPEND _rife_extra --allow-unsupported-compiler)
endif()
set(_rife_out "${CMAKE_CURRENT_BINARY_DIR}/rife-warp")
set(_rife_dll "${_rife_out}/rife-warp-plugin.dll")
set(_rife_model "${_rife_out}/rife_v4.22_fp16_warp_v2.onnx")
add_custom_command(OUTPUT "${_rife_dll}" "${_rife_model}"
    COMMAND "${Python3_EXECUTABLE}" "${_rife_source}/prepare.py"
        --model "${_rife_input}" --output "${_rife_out}"
    COMMAND "${Python3_EXECUTABLE}" "${_rife_source}/build.py"
        --cuda "${_rife_cuda}" --trt-include "${AJI_TRT_INCLUDE}"
        --cccl-include "${_rife_cuda}/include/cccl" --output "${_rife_out}" ${_rife_extra}
    DEPENDS "${_rife_input}" "${_rife_source}/prepare.py"
        "${_rife_source}/build.py" "${_rife_source}/plugin.cu"
    VERBATIM)
add_custom_target(aji_rife_warp ALL DEPENDS "${_rife_dll}" "${_rife_model}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:aji_trt>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_rife_dll}" "$<TARGET_FILE_DIR:aji_trt>"
    VERBATIM)
add_dependencies(aji_trt aji_rife_warp)
install(TARGETS aji_trt RUNTIME DESTINATION inference COMPONENT experimental-rife-warp)
install(FILES "${_rife_dll}" DESTINATION inference COMPONENT experimental-rife-warp)
install(FILES "${_rife_model}" "${_rife_input}" DESTINATION rife COMPONENT experimental-rife-warp)
