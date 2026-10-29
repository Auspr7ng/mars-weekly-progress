@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0code\launch.ps1" -Mode demo
if errorlevel 1 echo The demo exited with an error. Read the message above.
pause
