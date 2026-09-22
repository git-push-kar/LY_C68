#!/usr/bin/env bash
# setup_venv.sh - Create and initialize Python venv on Linux (RTX A5000 / CUDA)
set -e

echo "================================================================"
echo "Setting up Python virtual environment (venv) for Deepfake v2"
echo "================================================================"

# Verify python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10 or 3.11."
    exit 1
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment in ./venv ..."
    python3 -m venv venv
else
    echo "Existing virtual environment found in ./venv."
fi

# Activate
echo "Activating virtual environment ..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip ..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements from requirements.txt ..."
pip install -r requirements.txt

echo "================================================================"
echo "Virtual environment setup complete!"
echo "To activate in the future:"
echo "    source venv/bin/activate"
echo "================================================================"
