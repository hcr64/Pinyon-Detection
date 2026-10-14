#!/bin/bash
#SBATCH --job-name=pinyon_main
#SBATCH --output=clustering/logs/job_%j.out
#SBATCH --error=clustering/logs/job_%j.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection/
#SBATCH --time=1:00:00
#SBATCH --mem=48G

# access the venv
module load miniforge3/26.3.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate open3d_env

# set the trial name to field site name
TRIAL_NAME="Sunset_sfm_trial"

EPS=2.0                 # unused with CHM method
GREEN=0.025             # unused (Clean_Pointcloud is off)
RADIUS=4.25             # max_radius for split_large_clusters
MAX_DISTANCE=4.0
MIN_POINTS=300
VOXEL_SIZE=0.08
MIN_PEAK_DISTANCE=3.5
K=400
MIN_HEIGHT=0.4
SEARCH_RADIUS_M=2.5
GPS_SIGMA=4.0
SMOOTH_SIGMA=3

# clear out the image & log folders before it start
# find trial_data/$TRIAL_NAME/logs/* -mmin +15 -type f -delete
# find trial_data/$TRIAL_NAME/images/* -mmin +15 -type f -delete

# touch everything in the data folder so it doesn't get thrown away
touch /scratch/hcr64/Pinyon-Detection/data/$TRIAL_NAME/point_cloud/*

echo "Task ${SLURM_ARRAY_TASK_ID}: eps=$EPS green=$GREEN crown=$RADIUS max_dist=$MAX_DISTANCE min_pts=$MIN_POINTS voxel=$VOXEL_SIZE mpd=$MIN_PEAK_DISTANCE k=$K min_h=$MIN_HEIGHT sr=$SEARCH_RADIUS_M"

# keep false for sweeps
SAVE=True

# notes for reference in the output file
NOTES="GPS Loss Diagnostics"

# clustering + GPS labeling only. Run train_model.sh (or `python
# train_model.py --trial_name $TRIAL_NAME`) separately once this finishes.
python -u clustering/run_clustering.py \
    --eps               $EPS \
    --green_threshold   $GREEN \
    --max_radius        $RADIUS \
    --max_distance      $MAX_DISTANCE \
    --min_points        $MIN_POINTS \
    --voxel_size        $VOXEL_SIZE \
    --min_peak_distance $MIN_PEAK_DISTANCE \
    --k                 $K \
    --min_height        $MIN_HEIGHT \
    --search_radius_m   $SEARCH_RADIUS_M \
    --gps_sigma         $GPS_SIGMA \
    --smooth_sigma      $SMOOTH_SIGMA \
    --job_id            $SLURM_JOB_ID \
    --notes             "$NOTES" \
    --save              $SAVE \
    --trial_name        $TRIAL_NAME