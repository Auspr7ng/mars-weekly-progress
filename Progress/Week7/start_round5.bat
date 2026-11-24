@echo off
setlocal
cd /d "%~dp0"
call start_training.bat --round 5 --init-from policies/r26_round4_parent200/model_200.pt --envs 512 --iterations 600 --ramp-iterations 300 --seed 746 --run-dir runs/r26_round5_balanced35_20261124_05 --name "R26 Round 5 - Bilateral 35 mm Reference"
endlocal
