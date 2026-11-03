@echo off
setlocal
set PYTHONUTF8=1
if not defined WEEK4_PYTHON set "WEEK4_PYTHON=python"
echo Open http://127.0.0.1:6006 after TensorBoard starts.
"%WEEK4_PYTHON%" -m tensorboard.main --logdir "%~dp0results\training" --host 127.0.0.1 --port 6006
pause
endlocal
