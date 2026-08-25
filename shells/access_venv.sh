  #!/usr/bin/env bash

  # re-enter the venv
  module load miniforge3/26.3.2
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate open3d_env