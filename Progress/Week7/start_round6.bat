@echo off
setlocal
cd /d "%~dp0"
call start_training.bat --round 6 --init-from policies/r26_round4_parent400/model_400.pt --envs 512 --iterations 400 --ramp-iterations 200 --seed 747 --run-dir runs/r26_round6_vertical_20261124_06 --name "R26 Round 6 - Vertical Gait Freedom"
endlocal
