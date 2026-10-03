@echo off
setlocal

set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=sm_89"
set "TARGET=%~2"
if "%TARGET%"=="" set "TARGET=scope_gemm"

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
nvcc -O3 -std=c++17 -arch=%ARCH% -lineinfo --maxrregcount=128 --ptxas-options=-v -x cu ^
  main.cpp kernel_dispatch.cu ^
  -o "build\%TARGET%.exe" -lcublas
exit /b %errorlevel%
