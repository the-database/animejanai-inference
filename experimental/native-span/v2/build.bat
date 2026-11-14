@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -O3 -std=c++17 -arch=sm_120 -lineinfo -Xptxas -v -o spanf3_native.exe spanf3_native.cu
