@echo off
setlocal
cd /d "%~dp0"
call start_training.bat --round 2 --init-from runs/r26_clearance_speed_arms_20261119_01/model_799.pt --envs 256 --iterations 1200 --ramp-iterations 800 --seed 743 --run-dir runs/r26_round2_toe_heading_45mm_20261119_02 --name "R26 Round 2 - Toe Heading and 45 mm Foot Lift"
endlocal
