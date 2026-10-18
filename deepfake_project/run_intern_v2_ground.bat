@echo off
REM ============================================================================
REM  Stage 4 (Grounded "verdict-with-evidence") on the FRESH VANILLA base
REM  ./models/InternVL3-2B  --  continues from epoch 6 of runs/intern_v2
REM
REM  Mirrors the fused-base recipe (runs/intern_fused_ground):
REM    init        : runs/intern_v2/checkpoints/ep006.pth
REM    epochs_cls  : 0     (classifier already trained)
REM    epochs_joint: 3
REM    freeze      : --freeze_cls_head
REM    grounding   : ENABLED (default; do NOT pass --no_grounding)
REM    output      : ./runs/intern_v2_ground
REM
REM  Expected wall time: ~26-34 h (~8.5-14 h per grounded epoch on RTX A5000).
REM  DO NOT run automatically -- this file only documents/prepares the command.
REM ============================================================================
setlocal
cd /d "%~dp0"

python -u train.py ^
  --config config_intern_v2_ground.yaml ^
  --model_path ./models/InternVL3-2B ^
  --tokenizer_path ./models/InternVL3-2B ^
  --weights_from ./runs/intern_v2/checkpoints/ep006.pth ^
  --output_dir ./runs/intern_v2_ground ^
  --epochs_cls 0 ^
  --epochs_joint 3 ^
  --freeze_cls_head

if errorlevel 1 (
    echo ERROR: grounded fine-tune failed
    exit /b 1
)
echo Done. Grounded vanilla-base run written to ./runs/intern_v2_ground
endlocal
