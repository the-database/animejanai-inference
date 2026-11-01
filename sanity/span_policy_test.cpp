#include "span_models.h"
#include "core/plugin_cache.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void check(bool ok, const char *why)
{
    if (!ok) throw std::runtime_error(why);
}

int main()
{
    check(span_stock_model(BALANCED_CONCAT_MODEL) != nullptr, "missing Balanced fallback");
    check(span_stock_model(PERFORMANCE_CONCAT_MODEL) != nullptr, "missing Performance fallback");
    check(span_stock_model("user-custom-model") == nullptr, "ordinary model was remapped");
    check(span_unsupported_reason(1, 1920, 1080, 86) != nullptr, "unsupported cubin selected");
    check(span_unsupported_reason(1, 1918, 1078, 89) != nullptr, "unsupported shape selected");
    check(span_unsupported_reason(2, 1920, 1080, 120) == nullptr, "supported export rejected");
    const std::string user = "--builderOptimizationLevel=2 --optShapes=input:1x3x1080x1920 "
        "--tilingOptimizationLevel=1 --l2LimitForTiling=123 --skipInference";
    check(span_build_settings(user, 1, 1920, 1080) == user, "explicit builder options replaced");
    const std::string defaults = span_build_settings("--builderOptimizationLevel=2", 1, 1920, 1080);
    check(defaults.find("--builderOptimizationLevel=2") != std::string::npos &&
          defaults.find("--tilingOptimizationLevel=3") != std::string::npos &&
          defaults.find("--l2LimitForTiling=67108864") != std::string::npos,
          "preset defaults did not preserve user settings");
    check(!span_has_option("--tilingOptimizationLevelExtra=1", "--tilingOptimizationLevel"),
          "option prefix mistaken for an explicit setting");
    const auto path = std::filesystem::temp_directory_path() /
        ("aji-plugin-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    { std::ofstream file(path, std::ios::binary); file << "first"; }
    const std::string first = ajicore::plugin_binary_id(path.u8string());
    check(!first.empty() && first == ajicore::plugin_binary_id(path.u8string()), "unstable binary identity");
    { std::ofstream file(path, std::ios::binary); file << "other"; }
    check(first != ajicore::plugin_binary_id(path.u8string()), "same-size plugin change reused identity");
    std::filesystem::remove(path);
    std::cout << "PASS: SPAN eligibility, stock aliases, user settings and binary identity\n";
}
