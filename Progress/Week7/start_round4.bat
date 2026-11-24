@echo off
setlocal
cd /d "%~dp0"
call start_training.bat --round 4 --init-from policies/r26_round2/model_1199.pt --envs 512 --iterations 800 --ramp-iterations 400 --seed 745 --run-dir runs/r26_round4_raise_only_20261124_04 --name "R26 Round 4 - Bilateral Lift and Lateral Control"
endlocal
