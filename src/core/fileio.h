/*
 * core/fileio.h — path and file helpers shared by every aji library.
 *
 * Part of the internal core (no ABI of its own): compiled into each
 * library that uses it, with hidden visibility, so nothing here is
 * exported. Paths are UTF-8; on Windows they are widened and given the
 * \\?\ long-path prefix before touching the filesystem.
 */

#ifndef AJI_CORE_FILEIO_H
#define AJI_CORE_FILEIO_H

#include <string>
#include <vector>

namespace ajicore {

#ifdef _WIN32
std::wstring widen_utf8(const std::string &s);
std::wstring win_long_path(const std::string &path);
#endif

bool file_exists(const std::string &path);
void remove_file(const std::string &path);
// Size in bytes, or -1 if the file can't be stat'ed.
long long file_size(const std::string &path);
// Rename over an existing file.
bool move_file(const std::string &from, const std::string &to);
bool read_file_bytes(const std::string &path, std::vector<char> *out);
void append_file_text(const std::string &path, const std::string &text);

// Directory of the library this code is compiled into (aji.dll,
// aji_trt.dll, ...): backends resolve their runtime dependencies there.
std::string own_dir();

} // namespace ajicore

#endif
