@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -O3 -std=c++17 -arch=sm_120 -lineinfo -Xptxas -v
%NVCC% -o spanf3_native.exe spanf3_native.cu || exit /b 1
%NVCC% -o compact_native.exe compact_native.cu || exit /b 1
