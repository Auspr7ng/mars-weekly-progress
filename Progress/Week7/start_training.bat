@echo off
setlocal
set PYTHONUTF8=1
if not defined WEEK7_PYTHON set "WEEK7_PYTHON=D:\robotics\SundayA1_mjlab\.venv\Scripts\python.exe"
if not exist "%WEEK7_PYTHON%" (
  echo Set WEEK7_PYTHON to a compatible Week4 Python environment.
  exit /b 1
)
"%WEEK7_PYTHON%" -B -u "%~dp0code\train.py" %*
endlocal
