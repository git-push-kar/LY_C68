@echo off
setlocal
cd /d "%~dp0"
call conda activate veritas

set DATASET_ROOT=%~dp0datasets\hydrafake
set JSON_ROOT=%~dp0datasets\hydrafake\jsons
set MODEL_PATH=./models/InternVL3-2B

rem ep001-ep003 already evaluated (logs complete) - only eval remaining ep004, ep005 + best
for %%E in (4 5) do (
    echo ============================================
    echo Running evaluation for epoch %%E - exp3
    echo ============================================
    if not exist ".\runs\intern_exp3_corrective\checkpoints\ep00%%E.pth" (
        echo SKIP: checkpoint not found for ep00%%E - skipping
    ) else (
        python eval.py --dataset_root "%DATASET_ROOT%" --json_root "%JSON_ROOT%" --model_path "%MODEL_PATH%" --checkpoint "./runs/intern_exp3_corrective/checkpoints/ep00%%E.pth" --batch_size 8 --num_workers 8
        if errorlevel 1 (
            echo ERROR: eval failed for ep00%%E - continuing to next epoch
        )
    )
)

echo ============================================
echo Running evaluation for best.pth - exp3
echo ============================================
if not exist ".\runs\intern_exp3_corrective\checkpoints\best.pth" (
    echo SKIP: best.pth not found - skipping
) else (
    python eval.py --dataset_root "%DATASET_ROOT%" --json_root "%JSON_ROOT%" --model_path "%MODEL_PATH%" --checkpoint "./runs/intern_exp3_corrective/checkpoints/best.pth" --batch_size 8 --num_workers 8
    if errorlevel 1 (
        echo ERROR: eval failed for best.pth
    )
)

echo.
echo All evaluations completed - exp3 ep4, ep5, best (ep1-3 skipped, logs complete).
echo ============================================
echo Resuming training for ep6 - config.yaml resume ep005 - epochs_joint 6
echo ============================================

python train.py
if errorlevel 1 (
    echo ERROR: training resume failed
    exit /b 1
)

echo.
echo Done. Training resumed for ep6.
endlocal
