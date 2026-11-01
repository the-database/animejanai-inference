@echo off
REM The release build of the aji engine: every architecture, clean build.
REM Dependency layout and rationale: docs\BUILD-WINDOWS.md.
REM   AJI_WIN  the SDK working area (default %USERPROFILE%\aji-win)
REM   AJI_TRTRTX_ROOT  TensorRT for RTX SDK for aji_trtrtx (default %AJI_WIN%\trtrtx171)
REM Builds this checkout into build-win-release\ and prints AJI-RELEASE-BUILD-OK.
setlocal
if not defined AJI_WIN set "AJI_WIN=%USERPROFILE%\aji-win"
if not defined AJI_TRTRTX_ROOT set "AJI_TRTRTX_ROOT=%AJI_WIN%\trtrtx171"
for %%i in ("%~dp0..\..") do set "SRC=%%~fi"
set "BLD=%SRC%\build-win-release"
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [18.0^,19.0^) -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (echo Visual Studio 18 with the C++ tools not found & exit /b 1)
call "%VSDIR%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -no_logo || exit /b 1
set "CUDA_PATH=%AJI_WIN%\cuda134"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "AJI_SPAN_BUILD=OFF"
if defined AJI_SPAN_MODEL_DIR set "AJI_SPAN_BUILD=ON"
set "AJI_BLOCKS_BUILD=OFF"
if defined AJI_CUTLASS_INCLUDE set "AJI_BLOCKS_BUILD=ON"
cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual" ^
  "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%/bin/nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_PATH%" ^
  -DAJI_TRT_ROOT=%AJI_WIN%/trt114 -DAJI_NVINFER=nvinfer_11 ^
  -DAJI_BUILD_SPAN_PLUGINS=%AJI_SPAN_BUILD% "-DAJI_SPAN_MODEL_DIR=%AJI_SPAN_MODEL_DIR%" ^
  -DAJI_BUILD_SPAN_BLOCKS=%AJI_BLOCKS_BUILD% "-DAJI_CUTLASS_INCLUDE=%AJI_CUTLASS_INCLUDE%" ^
  "-DAJI_TRTRTX_ROOT:PATH=%AJI_TRTRTX_ROOT%" ^
  -DAJI_ORT_ROOT=%AJI_WIN%/ort/Microsoft.ML.OnnxRuntime.DirectML ^
  -DAJI_DML_ROOT=%AJI_WIN%/ort/Microsoft.AI.DirectML ^
  -DAJI_FFMPEG_ROOT=%AJI_WIN%/ffmpeg-shared/ffmpeg-9.0.2-full_build-shared ^
  "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler" || exit /b 1
cmake --build "%BLD%" --clean-first || exit /b 1
if "%AJI_SPAN_BUILD%"=="ON" (
    cmake --install "%BLD%" --component experimental-span --prefix "%BLD%\span-package\animejanai" || exit /b 1
)
echo AJI-RELEASE-BUILD-OK
