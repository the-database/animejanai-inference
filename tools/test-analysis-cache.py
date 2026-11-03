"""GPU regression for an empty analysis engine and a delete-pending cache lock.

Usage: python tools/test-analysis-cache.py <runtime> <model.onnx> <matching-cache> <new-output> [--lock-only]
The cache filename must match this model, GPU and driver. Only the new output
directory is modified. Rebuilding the empty engine can take several minutes.
"""
import ctypes as c
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

runtime, model, template, output = map(lambda p: Path(p).resolve(), sys.argv[1:5])
assert sys.argv[5:] in ([], ["--lock-only"])
lock_only = bool(sys.argv[5:])
output.mkdir(parents=True, exist_ok=False)
shutil.copyfile(model, output / model.name)
cached = output / template.name
if lock_only:
    shutil.copyfile(template, cached)
else:
    cached.touch()

with os.add_dll_directory(str(runtime)):
    cuda = c.WinDLL("nvcuda.dll")
    cuda.cuDevicePrimaryCtxRetain.argtypes = [c.POINTER(c.c_void_p), c.c_int]
    cuda.cuDevicePrimaryCtxRelease.argtypes = [c.c_int]
    context = c.c_void_p()
    assert cuda.cuInit(0) == 0
    assert cuda.cuDevicePrimaryCtxRetain(c.byref(context), 0) == 0
    dll = c.CDLL(str(runtime / "aji_analysis.dll"))
    create = dll.aji_observer_create
    create.restype = c.c_void_p
    create.argtypes = [c.c_void_p, c.c_int, c.c_int, c.c_char_p,
                      c.c_void_p, c.c_void_p, c.c_void_p, c.c_char_p, c.c_size_t]
    dll.aji_observer_destroy.argtypes = [c.c_void_p]

    def load():
        error = c.create_string_buffer(2048)
        observer = create(context, 64, 36, str(output / model.name).encode(),
                          None, None, None, error, len(error))
        assert observer, error.value.decode(errors="replace")
        dll.aji_observer_destroy(observer)

    try:
        load()
        assert cached.stat().st_size > 0, "Cache was not loaded at the expected key"
        print("PASS " + ("existing engine loaded" if lock_only else "empty analysis engine rebuilt and loaded"), flush=True)
        kernel = c.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [c.c_wchar_p, c.c_uint32, c.c_uint32, c.c_void_p,
                                      c.c_uint32, c.c_uint32, c.c_void_p]
        kernel.CreateFileW.restype = c.c_void_p
        kernel.CloseHandle.argtypes = [c.c_void_p]
        lock = str(cached) + ".lock"
        # Closing a delete-on-close handle while another handle remains open
        # leaves the name delete-pending. DeleteFileW can immediately unlink
        # the name on newer Windows, which would not reproduce this race.
        deleting = kernel.CreateFileW(lock, 0xC0000000, 7, None, 2, 0x04000100, None)
        assert deleting != c.c_void_p(-1).value, c.get_last_error()
        handle = kernel.CreateFileW(lock, 0x80000000, 7, None, 3, 0x100, None)
        kernel.CloseHandle(deleting)
        assert handle != c.c_void_p(-1).value, c.get_last_error()
        # Verify the fixture is delete-pending, not merely sharing-locked.
        probe = kernel.CreateFileW(lock, 0xC0000000, 0, None, 4, 0x100, None)
        error_code = c.get_last_error()
        if probe != c.c_void_p(-1).value:
            kernel.CloseHandle(probe)
        assert probe == c.c_void_p(-1).value and error_code == 5, error_code
        release = threading.Timer(0.6, lambda: kernel.CloseHandle(handle))
        release.start()
        started = time.monotonic()
        try:
            load()
        finally:
            release.join()
        elapsed = time.monotonic() - started
        assert not Path(lock).exists(), "Observer left its cache lock behind"
        result = {"passed": True, "emptyCacheRebuilt": not lock_only,
                  "deletePendingLockRecovered": True, "reloadSeconds": elapsed,
                  "engineBytes": cached.stat().st_size}
        (output / "results.json").write_text(json.dumps(result, indent=2))
        print("PASS delete-pending cache lock recovered and engine reloaded", flush=True)
    finally:
        cuda.cuDevicePrimaryCtxRelease(0)
