#!/bin/bash
# Setup environment using uv and pyproject.toml

############################################################
# Set exit on error
############################################################
set -e 

############################################################
# Ensure system dependencies
############################################################
echo "Installing system packages..."
sudo apt update
sudo apt install -y swig build-essential python3-dev pkg-config curl

############################################################
# Install uv if missing (Raspberry Pi OS compatible)
############################################################
if ! command -v uv &>/dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

############################################################
# Create uv environment
############################################################
if [ ! -d ".venv" ]; then
    echo "Creating uv virtual environment..."
    uv venv
else
    echo ".venv already exists."
fi

# Activate environment
source .venv/bin/activate

############################################################
# Install project dependencies from pyproject.toml
############################################################
echo "Installing dependencies from pyproject.toml..."
uv sync

echo "Setup complete!"
echo "Activate with: source .venv/bin/activate"
