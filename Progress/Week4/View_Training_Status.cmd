@echo off
setlocal
set PYTHONUTF8=1
if not defined WEEK4_PYTHON set "WEEK4_PYTHON=python"
"%WEEK4_PYTHON%" -B "%~dp0code\scripts\show_status.py"
pause
endlocal
