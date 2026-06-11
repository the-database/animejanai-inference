/* Internal: animejanai.conf model, ported from animejanai_config.py. */
#pragma once

#include <map>
#include <string>
#include <vector>

struct AjiModelConf {
    std::string name;                          // onnx base name; empty = none
    double resize_factor_before_upscale = 100.0;
    double resize_height_before_upscale = 0.0;
};

struct AjiChainConf {
    int index = 0;                             // chain_N, for logging
    double min_px = 0.0, max_px = 1e300;
    double min_fps = 0.0, max_fps = 1e300;
    std::string min_resolution = "0x0";
    std::string max_resolution = "infxinf";
    std::vector<AjiModelConf> models;
    bool rife = false;
    int rife_factor_num = 1;                   // output fps multiplier
    int rife_factor_den = 1;
    int rife_model = 414;                      // e.g. 414 = rife_v4.14
    bool rife_ensemble = false;
    double rife_scd_threshold = 0.150;         // misc.SCDetect threshold
};

struct AjiSlotConf {
    std::string profile_name = "New Profile";
    std::vector<AjiChainConf> chains;          // ordered by chain number
};

struct AjiConf {
    std::string backend = "TensorRT";
    bool logging = true;
    std::string trt_engine_settings;           // empty = code default
    std::map<int, AjiSlotConf> slots;
};

// Loads path (missing file = built-in defaults only). Returns false only on
// hard errors; *err gets the message.
bool aji_conf_load(const char *path, AjiConf *out, std::string *err);
