@echo off
nvcc -o build/child child.cu src/base/globals.cu -ftz=true --restrict -arch=sm_80 -lcuda

if %errorlevel% equ 0 (
  REM Compilation successful, run the executable
  echo.
  echo ----Compilation successful. Running----
  echo.
  node server.js
) else (
  REM Compilation failed
  echo Compilation failed. 
)

