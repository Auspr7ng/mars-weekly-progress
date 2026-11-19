@echo off
setlocal
set PYTHONUTF8=1
if not defined WEEK6_PYTHON set "WEEK6_PYTHON=D:\robotics\SundayA1_mjlab\.venv\Scripts\python.exe"
if not exist "%WEEK6_PYTHON%" (
  echo Set WEEK6_PYTHON to an existing mjlab Python environment.
  pause
  exit /b 1
)
"%WEEK6_PYTHON%" -B -u "%~dp0code\run_demo.py" %*
if errorlevel 1 pause
endlocal
