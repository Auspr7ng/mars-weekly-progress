@echo off
setlocal
cd /d "%~dp0"
call start_training.bat --round 3 --init-from runs/r26_round2_toe_heading_45mm_20261119_02/model_1199.pt --envs 256 --iterations 1200 --ramp-iterations 800 --seed 744 --run-dir runs/r26_round3_phase_symmetry_20261120_03 --name "R26 Round 3 - Bilateral Gait Symmetry"
endlocal
