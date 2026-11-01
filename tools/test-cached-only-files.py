"""GPU regression: cached-only checks must preserve stale and empty engines.

Usage: python tools/test-cached-only-files.py <runtime-dir> <small-model.onnx> <new-output-dir>
No inference or engine builds are performed. Only files created under output
are changed. Uses a real TensorRT backend on the first NVIDIA GPU.
"""
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

class Params(c.Structure):
    _fields_ = [("api_version", c.c_uint32), ("cuda_context", c.c_void_p),
        ("conf_path", c.c_char_p), ("model_dir", c.c_char_p), ("trtexec", c.c_char_p),
        ("trtexec_env", c.c_char_p), ("slot", c.c_int), ("rife_model_dir", c.c_char_p),
        ("async_build", c.c_int), ("d3d11_device", c.c_void_p), ("engine_path", c.c_char_p),
        ("max_width", c.c_int), ("max_height", c.c_int), ("log", c.c_void_p), ("log_opaque", c.c_void_p)]

runtime, model, output = [Path(v).resolve() for v in sys.argv[1:]]
output.mkdir(exist_ok=False, parents=True)
shutil.copyfile(model, output / model.name)
conf = output / "test.conf"
conf.write_text(f"[global]\nconfig_version=3\nbackend=TensorRT\n[slot_1]\nchain_1_max_fps=61\nchain_1_model_1_name={model.stem}\n")
stale = output / (model.stem + "_trt-11.0.0_gpu-test.engine")
analysis = output / (model.stem + ".model-test.analysisengine")
for path in [stale, analysis]: path.write_bytes(b"preserve")
with os.add_dll_directory(str(runtime)):
    dll = c.CDLL(str(runtime / "aji.dll"))
    dll.aji_create.argtypes = [c.POINTER(Params)]; dll.aji_create.restype = c.c_void_p
    dll.aji_configure.argtypes = [c.c_void_p, c.c_int, c.c_int, c.c_double, c.POINTER(c.c_int), c.POINTER(c.c_int)]
    dll.aji_last_error.argtypes = [c.c_void_p]; dll.aji_last_error.restype = c.c_char_p
    dll.aji_destroy.argtypes = [c.POINTER(c.c_void_p)]
    params = Params(api_version=9, conf_path=str(conf).encode(), model_dir=str(output).encode(), slot=1, async_build=2)
    def check():
        ctx = c.c_void_p(dll.aji_create(c.byref(params)))
        assert ctx.value, "Could not initialize TensorRT test context"
        try:
            w, h = c.c_int(), c.c_int()
            assert dll.aji_configure(ctx, 640, 360, 24.0, c.byref(w), c.byref(h)) < 0
            return dll.aji_last_error(ctx).decode()
        finally: dll.aji_destroy(c.byref(ctx))
    missing = check()
    assert missing.startswith("AJN_ENGINE_MISSING:") and "model=" + model.stem in missing, missing
    expected = Path(missing.split("; expected=", 1)[1])
    assert expected.parent == output and stale.read_bytes() == b"preserve" and analysis.read_bytes() == b"preserve"
    expected.touch()
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    empty = check()
    assert empty.startswith("AJN_ENGINE_INCOMPATIBLE:") and str(expected) in empty, empty
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    assert before == after, "Cached-only mode modified files"
    (output / "results.json").write_text(json.dumps({"passed": True, "missing": missing, "empty": empty, "filesUnchanged": True}, indent=2))
    print("PASS cached-only missing/empty engines preserve every file and identify model/path")
