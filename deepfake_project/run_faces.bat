@echo off
REM ============================================================================
REM  Multi-face inference: detect faces -> context crops -> the same adapter
REM
REM  Single-face images keep the validated whole-image path (bitwise identical
REM  to inference.py); images with several faces are classified one crop per
REM  face. See INTEGRATION_GUIDE.md section 11.
REM
REM  Usage:
REM    run_faces.bat                                   (2 sample images)
REM    run_faces.bat test_images\3-people.jpg          (one image)
REM    run_faces.bat path\to\a.jpg path\to\b.jpg       (images only, no flags)
REM
REM  Output: per-image JSON at runs\face_results.json, plus a full run log
REM  under runs\inference_logs\. Override the weights with ADAPTER_DIR, and
REM  the reasoning mode with REASON_FACES (none ^| top1 ^| all).
REM ============================================================================
setlocal
cd /d "%~dp0"

set PY=python
if exist "%~dp0venv\Scripts\python.exe" set PY=%~dp0venv\Scripts\python.exe

if "%ADAPTER_DIR%"=="" set ADAPTER_DIR=.\runs\intern_fused_ground\adapters
if "%REASON_FACES%"=="" set REASON_FACES=top1

if not exist "%ADAPTER_DIR%" (
    echo ERROR: adapter package not found at %ADAPTER_DIR%
    echo        Build it with extract_adapters.py, or set ADAPTER_DIR.
    exit /b 1
)

if not exist ".\models\yolov8n-face.pt" (
    echo Detector weights missing - fetching the pinned blob ...
    "%PY%" fetch_detector_weights.py
    if errorlevel 1 (
        echo ERROR: could not fetch models\yolov8n-face.pt
        exit /b 1
    )
)

if "%~1"=="" (
    set IMAGES=test_images\3-people.jpg test_images\multiple-faces.jpg
) else (
    set IMAGES=%*
)

echo Running multi-face inference with reason_faces=%REASON_FACES%
"%PY%" -u inference_faces.py ^
  --adapter_dir "%ADAPTER_DIR%" ^
  --images %IMAGES% ^
  --reason_faces %REASON_FACES% ^
  --json_out runs\face_results.json

if errorlevel 1 (
    echo ERROR: multi-face inference failed
    exit /b 1
)

echo.
echo Done. Per-image JSON written to runs\face_results.json
endlocal
