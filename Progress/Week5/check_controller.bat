@echo off
setlocal
set PYTHONUTF8=1
if not defined WEEK5_PYTHON set "WEEK5_PYTHON=D:\robotics\SundayA1_mjlab\.venv\Scripts\python.exe"
if not exist "%WEEK5_PYTHON%" (
  echo Set WEEK5_PYTHON to an existing Python environment.
  pause
  exit /b 1
)
"%WEEK5_PYTHON%" -B -u "%~dp0code\check_controller.py" %*
pause
endlocal
