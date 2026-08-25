#!/usr/bin/env bash
set -e

ENV_NAME="open3d_env"

if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: an environment is already active ($VIRTUAL_ENV$CONDA_DEFAULT_ENV). Deactivate first."
    exit 1
fi

echo "Loading miniforge3..."
module load miniforge3/26.3.2

echo "Creating conda environment: $ENV_NAME (python 3.10)"
# explicitly include pip in the env so it doesn't fall back to a user-level pip
conda create -n "$ENV_NAME" python=3.10 pip -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# sanity check — confirm python/pip actually resolve INSIDE the conda env
echo "python: $(which python)"
echo "pip:    $(which pip)"

# use `python -m pip` rather than bare `pip` — guarantees the pip tied to
# the currently active python, sidestepping PATH ambiguity entirely
python -m pip install --upgrade pip
python -m pip install --no-cache-dir open3d
python -m pip install numpy laspy pandas scipy scikit-learn matplotlib pyproj rasterio scikit-image imbalanced-learn xgboost lightgbm

# install torch too
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo
echo "Verifying numpy resolves inside the env:"
python -c "import numpy; print(numpy.__file__)"

echo "Done. Activate with: module load miniforge3/26.3.2 && conda activate $ENV_NAME"