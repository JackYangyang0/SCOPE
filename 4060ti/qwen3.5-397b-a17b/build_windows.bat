@echo off
setlocal

set "ARCH_NUM=%~1"
if "%ARCH_NUM%"=="" set "ARCH_NUM=87"
set "TARGET=%~2"
if "%TARGET%"=="" set "TARGET=scope_gemm"
set "MAXRREGCOUNT=%~3"
if "%MAXRREGCOUNT%"=="" set "MAXRREGCOUNT=128"

if defined VCVARS64 if exist "%VCVARS64%" goto setup_msvc
if exist "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"& goto setup_msvc
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"& goto setup_msvc
if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"& goto setup_msvc
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"& goto setup_msvc
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"& goto setup_msvc

echo Could not find vcvars64.bat. Set VCVARS64 to its full path.
exit /b 2

:setup_msvc
call "%VCVARS64%"
if errorlevel 1 exit /b %errorlevel%

if not exist build mkdir build
nvcc -O3 -std=c++17 ^
  -gencode arch=compute_%ARCH_NUM%,code=sm_%ARCH_NUM% ^
  -gencode arch=compute_%ARCH_NUM%,code=compute_%ARCH_NUM% ^
  -lineinfo --maxrregcount=%MAXRREGCOUNT% --ptxas-options=-v -x cu ^
  main.cpp kernel_dispatch.cu ^
  -o "build\%TARGET%.exe" -lcublas
exit /b %errorlevel%
