@echo off
REM setup_venv.bat - Create and initialize Python venv on Windows (RTX A5000 / CUDA)
echo ================================================================
echo Setting up Python virtual environment (venv) for Deepfake v2
echo ================================================================

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found on PATH. Please install Python 3.10 or 3.11.
    pause
    exit /b 1
)

REM Create virtual environment if it does not exist
if not exist "venv" (
    echo Creating virtual environment in .\venv ...
    python -m venv venv
) else (
    echo Existing virtual environment found in .\venv.
)

REM Activate virtual environment
echo Activating virtual environment ...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip ...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing requirements from requirements.txt ...
pip install -r requirements.txt

echo ================================================================
echo Virtual environment setup complete!
echo To activate in the future:
echo    .\venv\Scripts\activate
echo ================================================================
