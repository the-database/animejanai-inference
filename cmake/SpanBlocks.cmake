option(AJI_BUILD_SPAN_BLOCKS "Build the optional GPU-specific SPAN recipes" OFF)
set(AJI_CUTLASS_INCLUDE "" CACHE PATH "CUTLASS include directory (tested e13f5cd4142c43efc37c2f6339923eb4a57afb99)")
if(NOT AJI_BUILD_SPAN_BLOCKS)
    return()
endif()
if(NOT AJI_BUILD_SPAN_PLUGINS)
    message(FATAL_ERROR "AJI_BUILD_SPAN_BLOCKS requires AJI_BUILD_SPAN_PLUGINS")
endif()
if(NOT EXISTS "${AJI_CUTLASS_INCLUDE}/cutlass/cutlass.h")
    message(FATAL_ERROR "Set AJI_CUTLASS_INCLUDE to the tested CUTLASS headers")
endif()
set(_blocks_source "${CMAKE_CURRENT_SOURCE_DIR}/experimental/span-blocks")
set(_blocks_out "${CMAKE_CURRENT_BINARY_DIR}/span-plugins/blocks")
set(_blocks_stock "${AJI_SPAN_MODEL_DIR}/2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx")
set(_blocks_performance "${AJI_SPAN_MODEL_DIR}/2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx")
set(_blocks_dll "${_blocks_out}/balanced-blocks-plugin.dll")
set(_blocks_concat "${_blocks_out}/2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Blocks_Concat_v1.onnx")
set(_blocks_native "${_blocks_out}/2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Blocks_NativeTail7_v1.onnx")
set(_blocks_cache "${_blocks_out}/2x_AnimeJaNai_HD_V3.1_Performance_FP16_Concat_Cache48_v1.onnx")
add_custom_command(OUTPUT "${_blocks_dll}" "${_blocks_concat}" "${_blocks_native}" "${_blocks_cache}"
    COMMAND "${Python3_EXECUTABLE}" "${_blocks_source}/build.py"
        --kind block-concat --model "${_blocks_stock}" --cuda "${_span_cuda}"
        --trt-include "${AJI_TRT_INCLUDE}" --cutlass-include "${AJI_CUTLASS_INCLUDE}"
        --cccl-include "${_span_cuda}/include/cccl" --output "${_blocks_out}" ${_span_extra}
    COMMAND "${Python3_EXECUTABLE}" "${_blocks_source}/prepare.py"
        --kind native-tail7 --model "${_blocks_stock}" --output "${_blocks_out}"
    COMMAND "${Python3_EXECUTABLE}" "${_blocks_source}/prepare.py"
        --kind performance-cache48 --model "${_blocks_performance}" --output "${_blocks_out}"
    DEPENDS "${_blocks_stock}" "${_blocks_performance}"
        "${_blocks_source}/build.py" "${_blocks_source}/prepare.py" "${_blocks_source}/plugin.cu"
        "${_span_source}/prepare.py" "${_span_source}/balanced/kernels.cu"
        "${AJI_CUTLASS_INCLUDE}/cutlass/cutlass.h"
    VERBATIM)
add_custom_target(aji_span_blocks ALL
    DEPENDS "${_blocks_dll}" "${_blocks_concat}" "${_blocks_native}" "${_blocks_cache}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:aji_trt>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_blocks_dll}" "$<TARGET_FILE_DIR:aji_trt>"
    VERBATIM)
add_dependencies(aji_trt aji_span_blocks)
install(FILES "${_blocks_dll}" DESTINATION inference COMPONENT experimental-span)
install(FILES "${_blocks_concat}" "${_blocks_native}" "${_blocks_cache}"
    DESTINATION onnx COMPONENT experimental-span)
