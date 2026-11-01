/*
 * core/ort_dml.h — ONNX Runtime + DirectML loading shared by the video
 * and image front-ends of aji_dml (Windows only).
 */

#ifndef AJI_CORE_ORT_DML_H
#define AJI_CORE_ORT_DML_H

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>

#include <DirectML.h>

#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>

#include <string>

namespace ajicore {

extern const OrtApi *g_ort;
extern const OrtDmlApi *g_dml_api;
extern HRESULT(WINAPI *g_dml_create_device)(ID3D12Device *, DML_CREATE_DEVICE_FLAGS,
                                            REFIID, void **);

// Loads DirectML.dll then onnxruntime.dll from our own directory (the
// order matters: onnxruntime delay-loads DirectML, and preloading ours
// keeps the System32 copy out of the picture). Idempotent.
bool load_ort(std::string *err);

} // namespace ajicore

#endif
