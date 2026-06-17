/*
 * animejanai.conf parsing — C++ port of animejanai_config.py.
 *
 * Semantics preserved:
 *  - INI with [global] and [slot_N]; keys are case-insensitive (lowered).
 *  - chain_{c}_min/max_resolution "WxH"; "" -> 0x0 / infxinf; a 0 component
 *    in max_resolution means inf for that axis. min_px = w*h of min, etc.
 *  - chain_{c}_max_fps: missing -> inf, 0 -> inf. min_fps: missing -> 0.
 *  - models are chain_{c}_model_{m}_*, m = 1..count(distinct m).
 *  - built-in default profiles in slots 1001-1003 (Quality/Balanced/
 *    Performance) and 1010/1011 (benchmarks); a [slot_X] section in the
 *    file replaces the built-in for that slot entirely.
 *  - [global] quality/balanced/performance_preset=sharp swaps _HD_V3.1_
 *    models to _HD_V3.1Sharp1_ in the corresponding built-in slot.
 *  - config_version < 2: a verbatim legacy default trt_engine_settings is
 *    dropped so the code default (static engines) governs.
 *  - final_resize_height/factor are not parsed into chains in the Python
 *    parser (dead keys) and are intentionally not ported.
 */

#include "aji_conf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

namespace {

const char *LEGACY_DEFAULT_TRT_ENGINE_SETTINGS =
    "--stronglyTyped --optShapes=input:%video_resolution% --inputIOFormats=fp16:chw "
    "--outputIOFormats=fp16:chw --builderOptimizationLevel=5 "
    "--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --skipInference";

std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char ch) { return std::tolower(ch); });
    return s;
}

std::string strip(const std::string &s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos)
        return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

bool parse_yes(const std::string &v)
{
    return lower(strip(v)) == "yes";
}

double to_double(const std::string &v, double dflt)
{
    std::string s = strip(v);
    if (s.empty())
        return dflt;
    if (lower(s) == "inf")
        return 1e300;
    char *end = nullptr;
    double d = std::strtod(s.c_str(), &end);
    return end == s.c_str() ? dflt : d;
}

// "WxH" -> (w, h); inf components allowed.
bool parse_resolution(const std::string &s, double *w, double *h)
{
    size_t x = s.find('x');
    if (x == std::string::npos)
        return false;
    *w = to_double(s.substr(0, x), 0);
    *h = to_double(s.substr(x + 1), 0);
    return true;
}

using Section = std::map<std::string, std::string>;

struct IniFile {
    std::map<std::string, Section> sections;  // section names lowered
};

bool read_ini(const char *path, IniFile *out)
{
    std::ifstream f(path);
    if (!f)
        return false;
    std::string line, cur;
    while (std::getline(f, line)) {
        std::string t = strip(line);
        if (t.empty() || t[0] == ';' || t[0] == '#')
            continue;
        if (t.front() == '[' && t.back() == ']') {
            cur = lower(strip(t.substr(1, t.size() - 2)));
            out->sections[cur];
            continue;
        }
        size_t eq = t.find('=');
        if (eq == std::string::npos || cur.empty())
            continue;
        std::string key = lower(strip(t.substr(0, eq)));
        out->sections[cur][key] = strip(t.substr(eq + 1));
    }
    return true;
}

AjiChainConf parse_chain(const Section &sec, int chain)
{
    AjiChainConf c;
    c.index = chain;
    std::string pre = "chain_" + std::to_string(chain) + "_";

    auto get = [&](const std::string &k, const char *dflt) -> std::string {
        auto it = sec.find(pre + k);
        return it == sec.end() ? dflt : it->second;
    };

    c.min_resolution = get("min_resolution", "0x0");
    c.max_resolution = get("max_resolution", "0x0");
    if (c.min_resolution.empty())
        c.min_resolution = "0x0";
    if (c.max_resolution.empty())
        c.max_resolution = "infxinf";

    double min_w = 0, min_h = 0, max_w = 0, max_h = 0;
    parse_resolution(c.min_resolution, &min_w, &min_h);
    parse_resolution(c.max_resolution, &max_w, &max_h);
    if (max_w == 0)
        max_w = 1e300;
    if (max_h == 0)
        max_h = 1e300;
    c.min_px = min_w * min_h;
    c.max_px = max_w * max_h;

    c.min_fps = to_double(get("min_fps", "0"), 0);
    c.max_fps = to_double(get("max_fps", "inf"), 1e300);
    if (c.max_fps == 0)
        c.max_fps = 1e300;

    // distinct model indices for this chain
    std::set<int> model_ids;
    std::string mpre = pre + "model_";
    for (const auto &kv : sec) {
        if (kv.first.rfind(mpre, 0) != 0)
            continue;
        int id = std::atoi(kv.first.c_str() + mpre.size());
        if (id > 0)
            model_ids.insert(id);
    }
    for (int m = 1; m <= (int)model_ids.size(); m++) {
        AjiModelConf mc;
        std::string mp = mpre + std::to_string(m) + "_";
        auto mget = [&](const std::string &k, const char *dflt) -> std::string {
            auto it = sec.find(mp + k);
            return it == sec.end() ? dflt : it->second;
        };
        mc.name = mget("name", "");
        mc.resize_factor_before_upscale =
            to_double(mget("resize_factor_before_upscale", "100"), 100);
        mc.resize_height_before_upscale =
            to_double(mget("resize_height_before_upscale", "0"), 0);
        c.models.push_back(std::move(mc));
    }

    // RIFE keys mirror animejanai_config.py: ints go through float parsing,
    // model code 414 = rife_v4.14 (trailing 1 on 4-digit codes = lite).
    c.rife = parse_yes(get("rife", "no"));
    c.rife_factor_num = (int)to_double(get("rife_factor_numerator", "1"), 1);
    c.rife_factor_den = (int)to_double(get("rife_factor_denominator", "1"), 1);
    c.rife_model = (int)to_double(get("rife_model", "414"), 414);
    c.rife_ensemble = parse_yes(get("rife_ensemble", "no"));
    c.rife_scd_threshold =
        to_double(get("rife_scene_detect_threshold", "0.150"), 0.150);
    c.rife_before_upscale = parse_yes(get("rife_before_upscale", "yes"));
    return c;
}

AjiSlotConf parse_slot(const Section &sec)
{
    AjiSlotConf s;
    auto it = sec.find("profile_name");
    if (it != sec.end())
        s.profile_name = it->second;

    std::set<int> chain_ids;
    for (const auto &kv : sec) {
        if (kv.first.rfind("chain_", 0) != 0)
            continue;
        int id = std::atoi(kv.first.c_str() + 6);
        if (id > 0)
            chain_ids.insert(id);
    }
    for (int c : chain_ids)
        s.chains.push_back(parse_chain(sec, c));
    return s;
}

AjiChainConf builtin_chain(int index, double min_px, double max_px,
                           const char *min_res, const char *max_res,
                           double max_fps, const char *model)
{
    AjiChainConf c;
    c.index = index;
    c.min_px = min_px;
    c.max_px = max_px;
    c.min_resolution = min_res;
    c.max_resolution = max_res;
    c.min_fps = 0;
    c.max_fps = max_fps;
    AjiModelConf m;
    m.name = model;
    c.models.push_back(m);
    return c;
}

const char *HD_BAL = "2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16";
const char *HD_PERF = "2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16";
const char *SD = "2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op23_dynamo";

void add_builtin_slots(AjiConf *conf)
{
    const double PX1080 = 1920.0 * 1080.0, PX720 = 1280.0 * 720.0;
    {
        AjiSlotConf s;
        s.profile_name = "Quality";
        s.chains.push_back(builtin_chain(1, PX720, PX1080, "1280x720", "1920x1080", 31, HD_BAL));
        s.chains.push_back(builtin_chain(2, 0, PX720, "0x0", "1280x720", 31, SD));
        s.chains.push_back(builtin_chain(3, 0, PX1080, "0x0", "1920x1080", 61, HD_BAL));
        conf->slots[1001] = std::move(s);
    }
    {
        AjiSlotConf s;
        s.profile_name = "Balanced";
        s.chains.push_back(builtin_chain(1, PX720, PX1080, "1280x720", "1920x1080", 31, HD_BAL));
        s.chains.push_back(builtin_chain(2, 0, PX720, "0x0", "1280x720", 31, SD));
        conf->slots[1002] = std::move(s);
    }
    {
        AjiSlotConf s;
        s.profile_name = "Performance";
        s.chains.push_back(builtin_chain(1, PX720, PX1080, "1280x720", "1920x1080", 31, HD_PERF));
        s.chains.push_back(builtin_chain(2, 0, PX720, "0x0", "1280x720", 31, SD));
        conf->slots[1003] = std::move(s);
    }
    {
        AjiSlotConf s;
        s.chains.push_back(builtin_chain(1, 0, 1e300, "0x0", "infxinf", 1e300, HD_BAL));
        conf->slots[1010] = std::move(s);
    }
    {
        AjiSlotConf s;
        s.chains.push_back(builtin_chain(1, 0, 1e300, "0x0", "infxinf", 1e300, HD_PERF));
        conf->slots[1011] = std::move(s);
    }
    // Benchmark slots for comparing the RIFE/upscale order: identical HD
    // Balanced model and 2x interpolation, differing only in rife_before_upscale.
    // 1012 = upscale then interpolate; 1013 = interpolate then upscale (default).
    // Exercised headlessly by aji_harness --rife-chain.
    {
        AjiSlotConf s;
        s.profile_name = "Benchmark Balanced RIFE 2x (upscale then RIFE)";
        AjiChainConf ch = builtin_chain(1, 0, 1e300, "0x0", "infxinf", 1e300, HD_BAL);
        ch.rife = true;
        ch.rife_factor_num = 2;
        ch.rife_factor_den = 1;
        ch.rife_before_upscale = false;
        s.chains.push_back(ch);
        conf->slots[1012] = std::move(s);
    }
    {
        AjiSlotConf s;
        s.profile_name = "Benchmark Balanced RIFE 2x (RIFE then upscale)";
        AjiChainConf ch = builtin_chain(1, 0, 1e300, "0x0", "infxinf", 1e300, HD_BAL);
        ch.rife = true;
        ch.rife_factor_num = 2;
        ch.rife_factor_den = 1;
        ch.rife_before_upscale = true;
        s.chains.push_back(ch);
        conf->slots[1013] = std::move(s);
    }
}

void apply_sharp_presets(AjiConf *conf, const Section &global)
{
    const std::pair<int, const char *> keys[] = {
        {1001, "quality_preset"},
        {1002, "balanced_preset"},
        {1003, "performance_preset"},
    };
    for (auto &k : keys) {
        auto it = global.find(k.second);
        if (it == global.end() || lower(strip(it->second)) != "sharp")
            continue;
        auto sit = conf->slots.find(k.first);
        if (sit == conf->slots.end())
            continue;
        for (auto &chain : sit->second.chains) {
            for (auto &m : chain.models) {
                size_t pos = m.name.find("_HD_V3.1_");
                if (pos != std::string::npos)
                    m.name.replace(pos, 9, "_HD_V3.1Sharp1_");
            }
        }
    }
}

} // namespace

bool aji_conf_load(const char *path, AjiConf *out, std::string *err)
{
    *out = AjiConf();
    add_builtin_slots(out);

    IniFile ini;
    if (!path || !read_ini(path, &ini))
        return true;  // missing conf = built-ins only (matches lenient py read)

    auto git = ini.sections.find("global");
    if (git != ini.sections.end()) {
        const Section &g = git->second;
        int version = 1;
        auto vit = g.find("config_version");
        if (vit != g.end())
            version = std::max(1, std::atoi(vit->second.c_str()));

        auto bit = g.find("backend");
        if (bit != g.end())
            out->backend = bit->second;
        auto lit = g.find("logging");
        if (lit != g.end())
            out->logging = parse_yes(lit->second);
        auto tit = g.find("trt_engine_settings");
        if (tit != g.end()) {
            std::string v = strip(tit->second);
            // v1 -> v2 migration: drop the legacy materialized default.
            if (!(version < 2 && v == LEGACY_DEFAULT_TRT_ENGINE_SETTINGS))
                out->trt_engine_settings = v;
        }
        apply_sharp_presets(out, g);
    }

    for (const auto &sec : ini.sections) {
        if (sec.first.rfind("slot_", 0) != 0)
            continue;
        int slot = std::atoi(sec.first.c_str() + 5);
        if (slot <= 0)
            continue;
        out->slots[slot] = parse_slot(sec.second);  // file replaces built-in
    }

    (void)err;
    return true;
}
