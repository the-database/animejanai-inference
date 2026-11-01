// Fixed experimental exports and their ordinary-model fallbacks.
#pragma once
#include <cctype>
#include <string>

inline constexpr const char *BALANCED_CONCAT_MODEL =
    "2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Concat_Tiled3_v2";
inline constexpr const char *PERFORMANCE_CONCAT_MODEL =
    "2x_AnimeJaNai_HD_V3.1_Performance_FP16_Concat_v1";
inline constexpr const char *BALANCED_BLOCKS_MODEL =
    "2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Blocks_Concat_v1";
inline constexpr const char *BALANCED_NATIVE7_MODEL =
    "2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Blocks_NativeTail7_v1";
inline constexpr const char *PERFORMANCE_CACHE48_MODEL =
    "2x_AnimeJaNai_HD_V3.1_Performance_FP16_Concat_Cache48_v1";

inline int span_plugin_kind(const std::string &name)
{
    return name == BALANCED_CONCAT_MODEL ? 1 :
           name == PERFORMANCE_CONCAT_MODEL ? 2 :
           name == BALANCED_BLOCKS_MODEL ? 3 :
           name == BALANCED_NATIVE7_MODEL ? 4 :
           name == PERFORMANCE_CACHE48_MODEL ? 5 : 0;
}

inline const char *span_stock_model(const std::string &name)
{
    switch (span_plugin_kind(name)) {
    case 1: case 3: case 4:
        return "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16";
    case 2: case 5:
        return "2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16";
    default: return nullptr;
    }
}

inline const char *span_unsupported_reason(int kind, int w, int h, int sm)
{
    if (sm != 89 && sm != 120)
        return "plugin kernels support only sm_89 and sm_120";
    if ((kind == 3 || kind == 5) && (sm != 89 || w != 1920 || h != 1080))
        return "this Ada recipe is limited to sm_89 at 1920x1080";
    if (kind == 4 && (sm != 120 || w != 960 || h != 540))
        return "NativeTail7 is limited to sm_120 at 960x540";
    if (!((w == 480 && h == 360) || (w == 640 && h == 480) ||
          (w == 768 && h == 576) || (w == 960 && h == 540) ||
          (w == 1280 && h == 720) || (w == 1920 && h == 1080)))
        return "unsupported experimental input resolution";
    return nullptr;
}

inline bool span_has_option(const std::string &settings, const std::string &option)
{
    size_t p = 0;
    while ((p = settings.find(option, p)) != std::string::npos) {
        const size_t end = p + option.size();
        if ((!p || std::isspace(static_cast<unsigned char>(settings[p - 1]))) &&
            (end == settings.size() || settings[end] == '=' ||
             std::isspace(static_cast<unsigned char>(settings[end]))))
            return true;
        p = end;
    }
    return false;
}

inline std::string span_build_settings(std::string settings, int kind, int w, int h)
{
    // Presets supply defaults; explicit user options always win.
    if (kind == 3) {
        if (!span_has_option(settings, "--maxAuxStreams"))
            settings += " --maxAuxStreams=0";
        return settings; // measured 4090 recipe: no added tiling options
    }
    if ((kind == 1 || kind == 4 || (w == 1920 && h == 1080)) &&
        !span_has_option(settings, "--tilingOptimizationLevel"))
        settings += " --tilingOptimizationLevel=3";
    if (w == 1920 && h == 1080 && !span_has_option(settings, "--l2LimitForTiling"))
        settings += kind == 5 ? " --l2LimitForTiling=50331648" : " --l2LimitForTiling=67108864";
    return settings;
}
