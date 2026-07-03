#!/bin/bash
#SBATCH --job-name=pinyon_sweep
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=1-16
#SBATCH --time=02:00:00
#SBATCH --mem=92G
#SBATCH --partition=core

source /home/hcr64/Pinyon-Detection/open3d_env/bin/activate

TRIAL_NAME="Sunset_sfm_trial"

# clear out logs and images folder before beginning
find trial_data/$TRIAL_NAME/logs/* -mmin +15 -type f -delete
find trial_data/$TRIAL_NAME/images/* -mmin +15 -type f -delete

# skip comment and blank lines, then get row SLURM_ARRAY_TASK_ID
PARAMS=$(grep -v '^\s*#' parameters/params.txt | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")

EPS=$(             echo $PARAMS | awk '{print $1}')
GREEN=$(            echo $PARAMS | awk '{print $2}')
RADIUS=$(           echo $PARAMS | awk '{print $3}')
MAX_DISTANCE=$(     echo $PARAMS | awk '{print $4}')
MIN_POINTS=$(       echo $PARAMS | awk '{print $5}')
VOXEL_SIZE=$(       echo $PARAMS | awk '{print $6}')
MIN_PEAK_DISTANCE=$(echo $PARAMS | awk '{print $7}')
K=$(                echo $PARAMS | awk '{print $8}')
MIN_HEIGHT=$(       echo $PARAMS | awk '{print $9}')
SEARCH_RADIUS_M=$(  echo $PARAMS | awk '{print $10}')

echo "Task ${SLURM_ARRAY_TASK_ID}: eps=$EPS green=$GREEN crown=$RADIUS max_dist=$MAX_DISTANCE min_pts=$MIN_POINTS voxel=$VOXEL_SIZE mpd=$MIN_PEAK_DISTANCE k=$K min_h=$MIN_HEIGHT sr=$SEARCH_RADIUS_M"

# sweeps only ever needed clustering + GPS matching score — never the
# classifier comparison, which used to run (and clutter these logs) on
# every single array task via the old main.py
python -u run_clustering.py \
    --eps              $EPS \
    --green_threshold  $GREEN \
    --max_radius       $RADIUS \
    --max_distance     $MAX_DISTANCE \
    --min_points       $MIN_POINTS \
    --voxel_size       $VOXEL_SIZE \
    --min_peak_distance $MIN_PEAK_DISTANCE \
    --k                $K \
    --min_height       $MIN_HEIGHT \
    --search_radius_m  $SEARCH_RADIUS_M \
    --job_id           $SLURM_JOB_ID \
    --trial_name       $TRIAL_NAME