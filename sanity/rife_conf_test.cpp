#include "aji_conf.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

int main()
{
    if (aji_rife_model_name(426, false) != "rife_v4.26" ||
        aji_rife_model_name(426, true) != "rife_v4.26_ensemble" ||
        aji_rife_model_name(4261, false) != "rife_v4.26_lite" ||
        aji_rife_model_name(4121, true) != "rife_v4.12_lite_ensemble" ||
        aji_rife_model_name(4262, false) != "rife_v4.26_heavy" ||
        !aji_rife_model_name(4262, true).empty()) {
        std::fputs("RIFE selector resolved to the wrong model\n", stderr);
        return 1;
    }
    std::puts("PASS RIFE standard, Lite, Heavy and ensemble selectors");

    const char *path = "rife-conf-test.ini";
    std::ofstream(path) <<
        "[slot_1]\nchain_1_rife=yes\n"
        "[slot_2]\nchain_1_rife=yes\nchain_1_rife_scene_detect_enhanced=yes\n"
        "[slot_3]\nchain_1_rife=yes\nchain_1_rife_scene_detect_enhanced=no\n";
    AjiConf conf;
    std::string error;
    bool loaded = aji_conf_load(path, &conf, &error);
    std::remove(path);
    if (!loaded) throw std::runtime_error(error);
    for (const auto &[slot, profile] : conf.slots) {
        for (const auto &chain : profile.chains) {
            if (chain.rife_scene_detect_enhanced != (slot == 2))
                throw std::runtime_error("Experimental detection must be opt-in per chain");
            if (chain.rife_scd_threshold != 0.150)
                throw std::runtime_error("Default RIFE threshold changed");
        }
    }
    if (!conf.reuse_held_frames)
        throw std::runtime_error("Held-frame reuse must default to on");
    std::puts("PASS RIFE enhanced setting: explicit opt-in, omitted/no and built-in defaults");

    // [global] reuse_held_frames: on unless no/off/false/0
    for (const auto &[value, want] : {std::pair<const char *, bool>{"no", false}, {"OFF", false},
                                      {"false", false}, {"0", false}, {"yes", true}, {"", true}}) {
        std::ofstream(path) << "[global]\nreuse_held_frames=" << value << "\n";
        AjiConf held;
        const bool ok = aji_conf_load(path, &held, &error);
        std::remove(path);
        if (!ok) throw std::runtime_error(error);
        if (held.reuse_held_frames != want)
            throw std::runtime_error(std::string("reuse_held_frames=") + value + " parsed wrong");
    }
    std::puts("PASS [global] reuse_held_frames: on by default, no/off/false/0 turn it off");
}
