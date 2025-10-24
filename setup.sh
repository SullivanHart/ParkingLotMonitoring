#!/bin/bash
# Setup the venv w/ req

# Exit on error
set -e

# Check python
if ! command -v python &>/dev/null; then
  echo "Python does not exist."
  exit 1
fi

# Create venv if nonexistent 
if [ ! -d "venv" ]; then
  echo "Creating virtual environment."
  python3 -m venv venv
else
  echo "Virtual environment already exists."
fi

# Activate venv
source venv/bin/activate

# Upgrade pip and install requirements
echo "Installing dependencies."
pip install --upgrade pip
pip install -r requirements.txt

echo "Start venv w/ 'source pi./venv/bin/activate'."
