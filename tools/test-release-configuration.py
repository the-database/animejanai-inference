"""Run the release batch parser and reconfigure the actual TensorRT CMake block."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

source = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="aji-release-") as temporary:
    root = Path(temporary)
    pins = json.loads((source / "ci/windows/dependencies.json").read_text())
    shutil.copy2(source / "ci/windows/dependencies.json", root)
    line = next(line for line in (source / "ci/windows/build-aji-release.bat").read_text().splitlines()
                if line.startswith('for /f "tokens=1,2'))
    batch = root / "parse.bat"
    batch.write_text("@echo off\nsetlocal\n" + line + "\n" + "\n".join(
        f'if not "%{variable}%"=="{pins[key]}" exit /b 1'
        for variable, key in (("CUDA_DIR", "cudaDirectory"), ("TRT_DIR", "tensorRTDirectory"), ("TRT_LIB", "tensorRTLibrary"))) + "\n")
    env = {key: value for key, value in os.environ.items() if key not in ("CUDA_DIR", "TRT_DIR", "TRT_LIB")}
    subprocess.run(["cmd.exe", "/d", "/c", str(batch)], env=env, check=True)
    print("PASS release dependency parser")
    cmake = (source / "CMakeLists.txt").read_text()
    block = cmake[cmake.index('if(WIN32)'):cmake.index('find_package(CUDAToolkit')]
    (root / "CMakeLists.txt").write_text('cmake_minimum_required(VERSION 3.24)\nproject(Check NONE)\nset(WIN32 TRUE)\n' +
        block + '\nfile(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${AJI_TRT_LIB}")\n')
    for sdk in ("old-sdk", "new-sdk"):
        subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build"), "-G", "Ninja",
                        "-DAJI_TRT_ROOT=" + sdk], check=True, stdout=subprocess.DEVNULL)
        assert (root / "build/selected.txt").read_text() == sdk
    print("PASS TensorRT library follows reconfigured SDK root")
