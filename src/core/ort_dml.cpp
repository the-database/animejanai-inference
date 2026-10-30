/*
 * core/ort_dml.cpp — see ort_dml.h.
 */

#include "ort_dml.h"
#include "fileio.h"

namespace ajicore {

const OrtApi *g_ort = nullptr;
const OrtDmlApi *g_dml_api = nullptr;
HRESULT(WINAPI *g_dml_create_device)(ID3D12Device *, DML_CREATE_DEVICE_FLAGS,
                                     REFIID, void **) = nullptr;

bool load_ort(std::string *err)
{
    if (g_ort && g_dml_api && g_dml_create_device)
        return true;
    const std::string dir = own_dir();
    const DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    HMODULE dml = LoadLibraryExA((dir + "\\DirectML.dll").c_str(), NULL, flags);
    if (!dml) {
        *err = "DirectML.dll not found next to aji_dml.dll (error " +
               std::to_string(GetLastError()) + ")";
        return false;
    }
    HMODULE ort = LoadLibraryExA((dir + "\\onnxruntime.dll").c_str(), NULL,
                                 flags);
    if (!ort) {
        *err = "onnxruntime.dll not found next to aji_dml.dll (error " +
               std::to_string(GetLastError()) + ")";
        return false;
    }
    auto get_base = (const OrtApiBase *(ORT_API_CALL *)(void))
        GetProcAddress(ort, "OrtGetApiBase");
    if (!get_base) {
        *err = "onnxruntime.dll has no OrtGetApiBase";
        return false;
    }
    g_ort = get_base()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        *err = "onnxruntime.dll is older than the ORT headers we built "
               "against (API " + std::to_string(ORT_API_VERSION) + ")";
        return false;
    }
    if (g_ort->GetExecutionProviderApi("DML", ORT_API_VERSION,
                                       (const void **)&g_dml_api)) {
        *err = "this onnxruntime.dll has no DirectML execution provider";
        return false;
    }
    *(FARPROC *)&g_dml_create_device = GetProcAddress(dml, "DMLCreateDevice");
    if (!g_dml_create_device) {
        *err = "DirectML.dll has no DMLCreateDevice";
        return false;
    }
    return true;
}

} // namespace ajicore
