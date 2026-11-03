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
    release_lines = (source / "ci/windows/build-aji-release.bat").read_text().splitlines()
    line = next(line for line in release_lines
                if line.startswith('for /f "tokens=1,2'))
    guards = "\n".join(line for line in release_lines
                       if line.startswith(("if not defined CUDA_DIR", "if not defined TRT_DIR", "if not defined TRT_LIB")))
    batch = root / "parse.bat"
    batch.write_text("@echo off\nsetlocal\n" + line + "\n" + guards + "\n" + "\n".join(
        f'if not "%{variable}%"=="{pins[key]}" exit /b 1'
        for variable, key in (("CUDA_DIR", "cudaDirectory"), ("TRT_DIR", "tensorRTDirectory"), ("TRT_LIB", "tensorRTLibrary"))) + "\n")
    env = {key: value for key, value in os.environ.items() if key not in ("CUDA_DIR", "TRT_DIR", "TRT_LIB")}
    subprocess.run(["cmd.exe", "/d", "/c", str(batch)], env=env, check=True)
    print("PASS release dependency parser")
    missing_library = dict(pins)
    del missing_library["tensorRTLibrary"]
    (root / "dependencies.json").write_text(json.dumps(missing_library))
    # Use the real parser/guards without the fixture's value comparisons.
    batch.write_text("@echo off\nsetlocal\n" + line + "\n" + guards + "\nexit /b 0\n")
    assert subprocess.run(["cmd.exe", "/d", "/c", str(batch)], env=env).returncode == 1
    print("PASS missing TensorRT library rejects release configuration")
    cmake = (source / "CMakeLists.txt").read_text()
    block = cmake[cmake.index('if(WIN32)'):cmake.index('find_package(CUDAToolkit')]
    (root / "CMakeLists.txt").write_text('cmake_minimum_required(VERSION 3.24)\nproject(Check NONE)\nset(WIN32 TRUE)\n' +
        block + '\nfile(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${AJI_TRT_LIB}")\n')
    for sdk in ("old-sdk", "new-sdk"):
        directory = root / sdk
        directory.mkdir()
        library = directory.as_posix()
        subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build"), "-G", "Ninja",
                        "-DAJI_TRT_ROOT=" + library], check=True, stdout=subprocess.DEVNULL)
        selected = (root / "build/selected.txt").read_text()
        # CMake may expand a runner's Windows short temp path or normalize its
        # drive-letter case. Compare the selected directories, not spellings.
        assert Path(selected).samefile(directory), (selected, library)
    print("PASS TensorRT library follows reconfigured SDK root")
