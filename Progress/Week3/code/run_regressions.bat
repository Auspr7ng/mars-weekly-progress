@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -Mode regressions
if errorlevel 1 echo The demo exited with an error. Read the message above.
pause
