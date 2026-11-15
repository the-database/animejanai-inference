// native/guard.h — runs a native model build (parse, lower, create) so that no C++ exception from
// an untrusted model file (an allocation failure, say) crosses libaji's C API: it becomes a
// refusal with a reason, and the caller uses TensorRT. Internal, host only.
#ifndef AJI_NATIVE_GUARD_H
#define AJI_NATIVE_GUARD_H

#include <exception>
#include <string>

namespace ajinative {

template <class F>
bool guarded(std::string *why, F &&f)
{
    try {
        return f();
    } catch (const std::exception &e) {
        if (why) *why = std::string("native build failed: ") + e.what();
    } catch (...) {
        if (why) *why = "native build failed";
    }
    return false;
}

}  // namespace ajinative

#endif
