@echo off
setlocal

where cl >nul 2>nul
if not errorlevel 1 goto compiler_ready
if defined VCVARS64 if exist "%VCVARS64%" goto setup_msvc
if exist "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"& goto setup_msvc
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"& goto setup_msvc
if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"& goto setup_msvc
echo Could not find cl.exe or vcvars64.bat. Set VCVARS64 to its full path.
exit /b 2

:setup_msvc
call "%VCVARS64%"
if errorlevel 1 exit /b %errorlevel%

:compiler_ready
if not defined OPENBLAS_ROOT set "OPENBLAS_ROOT=C:\OpenBLAS"
if not exist "%OPENBLAS_ROOT%\include\cblas.h" (
  echo OpenBLAS cblas.h not found under "%OPENBLAS_ROOT%\include".
  echo Set OPENBLAS_ROOT to the OpenBLAS installation root.
  exit /b 2
)
if not exist "%OPENBLAS_ROOT%\lib\libopenblas.lib" (
  echo OpenBLAS import library not found: "%OPENBLAS_ROOT%\lib\libopenblas.lib"
  exit /b 2
)

if not exist build mkdir build
cl /O2 /TC /I"%OPENBLAS_ROOT%\include" openblas_sgemm_fp32.c ^
  /Fe:build\openblas_sgemm_fp32.exe /link ^
  /LIBPATH:"%OPENBLAS_ROOT%\lib" libopenblas.lib
if errorlevel 1 exit /b %errorlevel%

echo Built build\openblas_sgemm_fp32.exe
