@echo off
setlocal
set PYTHONUTF8=1
if not defined WEEK4_PYTHON set "WEEK4_PYTHON=python"
"%WEEK4_PYTHON%" -B -u "%~dp0code\run_demo.py" %*
if errorlevel 1 (
  echo Activate a compatible Python environment or set WEEK4_PYTHON to its python.exe.
  pause
)
endlocal
