#!/bin/bash
module load miniforge3/26.3.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate open3d_env

python -m pip install ipykernel pyzmq jupyter_client --break-system-packages
python -c "import zmq"   # should exit silently with no error

python -m ipykernel install --user --name open3d_env --display-name "Python (open3d_env)"


#python -m pip install ipykernel pyzmq jupyter_client
