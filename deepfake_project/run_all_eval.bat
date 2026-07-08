@echo off

call conda activate veritas

set DATASET_ROOT=C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake
set JSON_ROOT=C:\Users\Admin\Desktop\ly project c 68\deepfake_project\datasets\hydrafake\jsons
set MODEL_PATH=./models/InternVL3-2B

for %%E in (8 7 6 5) do (

    echo ============================================
    echo Running evaluation for epoch %%E
    echo ============================================

    python eval.py ^
        --dataset_root "%DATASET_ROOT%" ^
        --json_root "%JSON_ROOT%" ^
        --model_path "%MODEL_PATH%" ^
        --checkpoint "./runs/intern_exp2/checkpoints/ep00%%E.pth" ^
        --batch_size 8 ^
        --num_workers 8
)

echo.
echo All evaluations completed.
pause