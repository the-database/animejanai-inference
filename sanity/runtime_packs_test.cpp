/*
 * runtime_packs_test — CPU-only checks for core/runtime_packs'
 * aji-runtime.json reader (no GPU, no files).
 */

#include <cstdio>
#include <string>

#include "core/runtime_packs.h"

using namespace ajicore;

namespace {

int failures = 0;

void check(bool ok, const char *what)
{
    if (!ok) {
        printf("FAIL: %s\n", what);
        failures++;
    }
}

#ifdef _WIN32
const std::string DIR = "C:\\app";
const std::string ABS = "C:\\\\store\\\\trt";  // as JSON text
const std::string ABS_VAL = "C:\\store\\trt";
#else
const std::string DIR = "/app";
const std::string ABS = "/store/trt";
const std::string ABS_VAL = "/store/trt";
#endif

} // namespace

int main()
{
    {
        RuntimePacks r = parse_runtime_packs(
            "{ \"format\": 1, \"trt\": \"" + ABS + "\", \"trt_builder\": [\"b1\", \"b2/\"],"
            " \"trtrtx\": \"rtx\" }", DIR);
        check(r.present && r.error.empty(), "valid file parses");
        check(r.trt == ABS_VAL, "absolute path kept");
#ifdef _WIN32
        check(r.trt_builder.size() == 2 && r.trt_builder[0] == "C:\\app\\b1" &&
                  r.trt_builder[1] == "C:\\app\\b2",
              "relative builder paths resolve against the file's folder");
        check(r.trtrtx == "C:\\app\\rtx", "relative trtrtx path");
#else
        check(r.trt_builder.size() == 2 && r.trt_builder[0] == "/app/b1" &&
                  r.trt_builder[1] == "/app/b2",
              "relative builder paths resolve against the file's folder");
        check(r.trtrtx == "/app/rtx", "relative trtrtx path");
#endif
    }
    {
        RuntimePacks r = parse_runtime_packs("\xEF\xBB\xBF{\"format\":1}", DIR);
        check(r.error.empty() && r.trt.empty() && r.trt_builder.empty() &&
                  r.trtrtx.empty(),
              "BOM and an empty pack set");
    }
    {
        RuntimePacks r = parse_runtime_packs(
            "{\"later\": {\"a\": [1, 2.5e3, true, null, \"x\"]}, \"format\": 1,"
            " \"trt\": \"\\u00e9\\ud83d\\ude00/x\"}", DIR);
        check(r.error.empty(), "unknown keys are skipped");
        check(r.trt.find("\xC3\xA9\xF0\x9F\x98\x80") != std::string::npos,
              "\\u escapes become UTF-8 (incl. surrogate pairs)");
    }
    check(!parse_runtime_packs("{\"trt\": \"x\"}", DIR).error.empty(),
          "missing format is an error");
    check(!parse_runtime_packs("{\"format\": 2}", DIR).error.empty(),
          "a newer format is an error");
    check(!parse_runtime_packs("{\"format\": 1,}", DIR).error.empty(),
          "trailing comma is an error");
    check(!parse_runtime_packs("{\"format\": 1} x", DIR).error.empty(),
          "trailing data is an error");
    check(!parse_runtime_packs("{\"format\": 1, \"trt_builder\": \"x\"}", DIR).error.empty(),
          "trt_builder must be an array");
    check(!parse_runtime_packs("{\"format\": 1, \"trt\": \"x", DIR).error.empty(),
          "unterminated string is an error");
    check(!parse_runtime_packs("", DIR).error.empty(), "empty file is an error");
    check(parse_runtime_packs("", DIR).present, "a broken file still counts as present");

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("runtime packs tests passed\n");
    return 0;
}
