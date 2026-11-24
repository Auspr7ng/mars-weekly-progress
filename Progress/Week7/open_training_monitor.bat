@echo off
setlocal
if not defined WEEK7_PYTHON set "WEEK7_PYTHON=D:\robotics\SundayA1_mjlab\.venv\Scripts\python.exe"
if not exist "%WEEK7_PYTHON%" (
  echo Set WEEK7_PYTHON to your compatible Python environment.
  pause
  exit /b 1
)
"%WEEK7_PYTHON%" -B "%~dp0code\training_monitor.py" %*
endlocal
