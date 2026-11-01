"""Windows, no GPU needed: cached-only requests must not change backends.

Usage: python tools/test-stream-policy-dispatch.py build/Release/aji.dll
The temporary directory intentionally contains no backend DLLs. Ordinary
TensorRT-RTX startup should attempt upstream's fallback; async_build=2 must not.
"""

import ctypes as c
from pathlib import Path
import shutil
import sys
import tempfile


Log = c.CFUNCTYPE(None, c.c_void_p, c.c_int, c.c_char_p)


class Params(c.Structure):
    _fields_ = [
        ("api_version", c.c_uint32), ("cuda_context", c.c_void_p),
        ("conf_path", c.c_char_p), ("model_dir", c.c_char_p),
        ("trtexec", c.c_char_p), ("trtexec_env", c.c_char_p),
        ("slot", c.c_int), ("rife_model_dir", c.c_char_p),
        ("async_build", c.c_int), ("d3d11_device", c.c_void_p),
        ("engine_path", c.c_char_p), ("max_width", c.c_int),
        ("max_height", c.c_int), ("log", Log), ("log_opaque", c.c_void_p),
    ]


with tempfile.TemporaryDirectory(prefix="ajn-stream-policy-") as temporary:
    root = Path(temporary)
    library = root / "aji.dll"
    shutil.copy2(Path(sys.argv[1]).resolve(), library)
    conf = root / "animejanai.conf"
    conf.write_text("[global]\nconfig_version=3\nbackend=TensorRT-RTX\n", encoding="utf-8")
    dll = c.CDLL(str(library))
    dll.aji_create.argtypes = [c.POINTER(Params)]
    dll.aji_create.restype = c.c_void_p
    lines = []

    @Log
    def log(_opaque, _level, message):
        lines.append(message.decode("utf-8", errors="replace"))

    params = Params(api_version=9, conf_path=str(conf).encode("utf-8"),
                    model_dir=str(root).encode("utf-8"), slot=1002, log=log)
    try:
        for policy in (0, 2):
            lines.clear()
            params.async_build = policy
            assert not dll.aji_create(c.byref(params)), "Missing backend unexpectedly opened"
            assert any("aji_trtrtx" in line for line in lines), "RTX selection was not tested"
            fallback = any("using TensorRT instead" in line for line in lines)
            assert fallback == (policy == 0), "Unexpected fallback policy: " + repr(lines)
            print(f"PASS async_build={policy}: fallback attempted={fallback}")
    finally:
        # Windows cannot remove the temporary DLL while it remains loaded.
        kernel = c.WinDLL("kernel32", use_last_error=True)
        kernel.FreeLibrary.argtypes = [c.c_void_p]
        kernel.FreeLibrary.restype = c.c_int
        if not kernel.FreeLibrary(dll._handle):
            raise c.WinError(c.get_last_error())
