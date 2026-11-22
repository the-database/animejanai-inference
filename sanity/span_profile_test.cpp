#include "aji_conf.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>

int main()
{
    using ajinative::SpanPrecision;
    AjiConf conf;
    std::string error;
    if (!aji_conf_load(nullptr, &conf, &error)) throw std::runtime_error(error);
    for (const auto &[slot, entry] : conf.slots)
        if (entry.span_precision != SpanPrecision::Default)
            throw std::runtime_error("Existing built-in profile opted into SPAN precision");

    const char *path = "span-profile-test.ini";
    std::ofstream(path) << "[slot_3]\nprofile_name=My own slot\n"
                           "[slot_1]\nnative_span_precision=FP16\n"
                           "[slot_2]\nnative_span_precision=fp8\n";
    const bool ok = aji_conf_load(path, &conf, &error);
    std::remove(path);
    if (!ok) throw std::runtime_error(error);
    if (conf.slots.at(3).span_precision != SpanPrecision::Default ||
        conf.slots.at(1).span_precision != SpanPrecision::FP16 || conf.slots.at(2).span_precision != SpanPrecision::FP8)
        throw std::runtime_error("Precision must follow each slot, including user overrides");
}
