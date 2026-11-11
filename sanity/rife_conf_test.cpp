#include "aji_conf.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>

int main()
{
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
    std::puts("PASS RIFE enhanced setting: explicit opt-in, omitted/no and built-in defaults");
}
