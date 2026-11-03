"""GPU regression: cached-only checks must preserve stale and empty engines.

Usage: python tools/test-cached-only-files.py <runtime-dir> <small-model.onnx> <new-output-dir>
No inference or engine builds are performed. Only files created under output
are changed. Uses a real TensorRT backend on the first NVIDIA GPU.
"""
import ctypes as c
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

class Params(c.Structure):
    _fields_ = [("api_version", c.c_uint32), ("cuda_context", c.c_void_p),
        ("conf_path", c.c_char_p), ("model_dir", c.c_char_p), ("trtexec", c.c_char_p),
        ("trtexec_env", c.c_char_p), ("slot", c.c_int), ("rife_model_dir", c.c_char_p),
        ("async_build", c.c_int), ("d3d11_device", c.c_void_p), ("engine_path", c.c_char_p),
        ("max_width", c.c_int), ("max_height", c.c_int), ("log", c.c_void_p), ("log_opaque", c.c_void_p)]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runtime", type=Path)
parser.add_argument("model", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--valid-engine", type=Path, help="Optional matching engine to test invalid legacy cache + valid short cache")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=360)
args = parser.parse_args()
runtime, model, output = [p.resolve() for p in (args.runtime, args.model, args.output)]
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
    def check(expect_success=False):
        ctx = c.c_void_p(dll.aji_create(c.byref(params)))
        assert ctx.value, "Could not initialize TensorRT test context"
        try:
            w, h = c.c_int(), c.c_int()
            result = dll.aji_configure(ctx, args.width, args.height, 24.0, c.byref(w), c.byref(h))
            assert (result >= 0) == expect_success, dll.aji_last_error(ctx).decode()
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
    if args.valid_engine:
        shutil.copyfile(args.valid_engine, expected)
        legacy = expected.with_name(model.stem + "." + expected.name.split(".", 1)[1])
        legacy.touch()
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
        check(expect_success=True)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
        assert before == after, "Valid short cache fallback modified files"
        print("PASS empty legacy engine does not mask a valid short-name engine")
    (output / "results.json").write_text(json.dumps({"passed": True, "missing": missing, "empty": empty,
        "filesUnchanged": True, "validShortCacheFallback": bool(args.valid_engine)}, indent=2))
    print("PASS cached-only missing/empty engines preserve every file and identify model/path")
