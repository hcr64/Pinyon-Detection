#!/usr/bin/env bash

# Stop on error
set -e

# Name of the virtual environment
VENV_NAME="open3d_env"

echo "Creating virtual environment: $VENV_NAME"

# make sure its python 3.10
module load python/3.10

# Create venv
python -m venv $VENV_NAME

# Activate venv
source $VENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install specific Open3D version
pip install --no-cache-dir open3d
pip install numpy
pip install laspy

