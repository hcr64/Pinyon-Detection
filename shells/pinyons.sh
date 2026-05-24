#!/bin/bash
#SBATCH --job-name=pinyon_main                        # the name of your job
#SBATCH --output=/scratch/hcr64/pinyon_output.txt      # this is the file your $
#SBATCH --error=/scratch/hcr64/pinyon_output.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection                  # your work directory
#SBATCH --time=40:00                            # (max time) 40 min (shorte$
#SBATCH --mem=100G                              # (total mem) 100GB of memory

# get in the open3d environment
# source open3d_env/bin/activate

# Run your application: precede the application command with 'srun'
# A couple example applications...
# srun date
srun python --version

# the trial name, for file locations
TRIAL_NAME="Sunset_sfm_trial"

# variables
# right now, they have all been optimized since 5/21/2026

# larger EPS means larger clusters
EPS=2.0

# green threshold, as it increases more points allowed
# the higher it is, the more exclusive.
# It is the minimum gap between greenness and other colors in a point.
GREEN=0.025

# for splitting large clusters, clusters with radii larger get split
RADIUS=4.0

# maximum distance for assigning labels to clusters
MAX_DISTANCE=2.85

# minimum points per cluster
MIN_POINTS=40

# voxel size when downsizing
VOXEL_SIZE=0.08

# run the main function
python -u main.py \
    --eps $EPS \
    --green_threshold $GREEN \
    --max_radius $RADIUS \
    --max_distance $MAX_DISTANCE \
    --min_points $MIN_POINTS \
    --voxel_size $VOXEL_SIZE \
    --trial_name $TRIAL_NAME > /scratch/hcr64/pinyon_output.txt
