#!/bin/bash
# Setup the venv w/ req

# Exit on error
set -e

# Check python
if ! command -v python3 &>/dev/null; then
  echo "Python3 does not exist."
  exit 1
fi

# Ensure required system packages exist
echo "Installing required system packages..."
sudo apt update
sudo apt install -y swig build-essential python3-dev pkg-config

# Create venv if nonexistent 
if [ ! -d "venv" ]; then
  echo "Creating virtual environment."
  python3 -m venv venv
else
  echo "Virtual environment already exists."
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Pi 5 compatible PaddlePaddle (ARM64)
echo "Installing PaddlePaddle for Pi 5..."
pip install paddlepaddle==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/
pip install paddleocr==2.7.0.3

# Install other requirements if file exists
if [ -f "requirements.txt" ]; then
  echo "Installing other dependencies from requirements.txt..."
  pip install -r requirements.txt
fi

echo "Setup complete."
echo "Start venv w/ 'source venv/bin/activate'."
