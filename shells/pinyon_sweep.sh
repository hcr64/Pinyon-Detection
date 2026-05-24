#!/bin/bash
#SBATCH --job-name=pinyon_sweep
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=1-4        # match number of lines in params.txt
#SBATCH --time=02:00:00
#SBATCH --mem=256G
#SBATCH --partition=core

# activate your environment
source /home/hcr64/Pinyon-Detection/open3d_env/bin/activate

# clear out previous logs
rm -rf /home/hcr64/Pinyon-Detection/logs/*

# dont change this unless changing dataset
TRIAL_NAME="Sunset_sfm_trial"

# read the parameters for this job
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" parameters/params.txt)
EPS=$(echo $PARAMS | awk '{print $1}')
GREEN=$(echo $PARAMS | awk '{print $2}')
RADIUS=$(echo $PARAMS | awk '{print $3}')
MAX_DISTANCE=$(echo $PARAMS | awk '{print $4}')
MIN_POINTS=$(echo $PARAMS | awk '{print $5}')
VOXEL_SIZE=$(echo $PARAMS | awk '{print $6}')


echo "Running: eps=$EPS green=$GREEN radius=$RADIUS min_points=$MIN_POINTS"

python -u main.py \
    --eps $EPS \
    --green_threshold $GREEN \
    --max_radius $RADIUS \
    --max_distance $MAX_DISTANCE \
    --min_points $MIN_POINTS \
    --voxel_size $VOXEL_SIZE \
    --trial_name $TRIAL_NAME