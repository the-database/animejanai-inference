// Cache fingerprint for the actual fixed-weight plugin binary, not just its name.
#pragma once
#include "fileio.h"
#include <cstdint>
#include <cstdio>

namespace ajicore {
inline std::string plugin_binary_id(const std::string &path)
{
    std::vector<char> bytes;
    if (!read_file_bytes(path, &bytes) || bytes.empty())
        return {};
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned char byte : bytes) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    char id[17];
    std::snprintf(id, sizeof(id), "%016llx", static_cast<unsigned long long>(hash));
    return id; // FNV-1a content identity; not an authenticity/security check.
}
} // namespace ajicore
