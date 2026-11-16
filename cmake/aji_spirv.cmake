# aji_spirv_embed(<name> <manifest>): compile the Vulkan compute shaders a manifest lists with
# glslc (AJI_GLSLC) and embed them in one generated C++ source, whose path is returned in
# AJI_VK_SHADERS_SOURCE. The source defines ajivk::kSpirvModules / kSpirvModuleCount
# (src/vk/spirv.h).
#
# Manifest: one module per line, `<module name> <shader file> [NAME=VALUE ...]`, the shader file
# relative to the manifest's directory and the NAME=VALUE pairs passed to glslc as -D defines.
# Blank lines and lines starting with # are skipped. One shader file can make many modules (its
# structural variants: channel counts, tile shapes, accumulation modes).
function(aji_spirv_embed name manifest)
    get_filename_component(manifest "${manifest}" ABSOLUTE)
    get_filename_component(shader_dir "${manifest}" DIRECTORY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${manifest}")
    set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    file(MAKE_DIRECTORY "${out_dir}")
    file(STRINGS "${manifest}" lines)
    set(spvs "")
    set(names "")
    foreach(line IN LISTS lines)
        string(STRIP "${line}" line)
        if(line STREQUAL "" OR line MATCHES "^#")
            continue()
        endif()
        string(REGEX REPLACE "[ \t]+" ";" fields "${line}")
        list(LENGTH fields n)
        if(n LESS 2)
            message(FATAL_ERROR "${manifest}: '${line}' needs a module name and a shader file")
        endif()
        list(GET fields 0 module)
        list(GET fields 1 shader)
        if(NOT module MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
            message(FATAL_ERROR "${manifest}: module name '${module}' is not an identifier")
        endif()
        if(module IN_LIST names)
            message(FATAL_ERROR "${manifest}: module '${module}' listed twice")
        endif()
        list(SUBLIST fields 2 -1 defs)
        set(dflags "")
        foreach(d IN LISTS defs)
            list(APPEND dflags "-D${d}")
        endforeach()
        set(src "${shader_dir}/${shader}")
        set(spv "${out_dir}/${module}.spv")
        add_custom_command(
            OUTPUT "${spv}"
            COMMAND "${AJI_GLSLC}" --target-env=vulkan1.3 -O ${dflags}
                    -MD -MF "${spv}.d" "${src}" -o "${spv}"
            DEPENDS "${src}"
            DEPFILE "${spv}.d"
            COMMENT "glslc ${module}"
            VERBATIM)
        list(APPEND spvs "${spv}")
        list(APPEND names "${module}")
    endforeach()
    set(gen "${out_dir}/${name}.cpp")
    add_custom_command(
        OUTPUT "${gen}"
        COMMAND "${CMAKE_COMMAND}" "-DOUT=${gen}" "-DNAMES=${names}" "-DSPVS=${spvs}"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/aji_spirv_gen.cmake"
        DEPENDS ${spvs} "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/aji_spirv_gen.cmake"
        COMMENT "embedding ${name} (SPIR-V)"
        VERBATIM)
    set(AJI_VK_SHADERS_SOURCE "${gen}" PARENT_SCOPE)
endfunction()
