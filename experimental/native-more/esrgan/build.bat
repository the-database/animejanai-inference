@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set SRC=%~dp0..\wt-native\src
cd /d "%~dp0"
nvcc -O3 -std=c++17 -arch=sm_120 -I"%SRC%" convbench.cu "%SRC%\native\conv.cu" "%SRC%\native\pack.cpp" -o convbench.exe
