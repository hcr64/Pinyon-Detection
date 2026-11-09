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
VOXEL_SIZE=0.08
GPS_SIGMA=4.0

# ── params.txt row 5 ──
RADIUS=4.25             # max_radius
CROWN_RADIUS=8.0
MAX_DISTANCE=4.0
MIN_POINTS=250
MIN_PEAK_DISTANCE=3.0
K=400
MIN_DENSITY_RATIO=1.5
MIN_HEIGHT=0.4
SEARCH_RADIUS_M=2.5
SMOOTH_SIGMA=3.0
USE_COLOR=False
COLOR_WEIGHT=10.0
MIN_EXG=0.05
MIN_CLUSTER_HEIGHT=1.5

# get a user inputted message after the -m command
while getopts "m:" opt; do
  case ${opt} in
    m ) MESSAGE="$OPTARG" ;;
    \? ) echo "Usage: $0 -m \"your message\"" >&2; exit 1 ;;
  esac
done

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

# the path to save results to
RESULTS_PATH="trial_data/$TRIAL_NAME/GPS_results/job_${SLURM_JOB_ID}.csv"

# print header message
mkdir -p "$(dirname "$RESULTS_PATH")"
{
    echo "# ==== sweep started $(date '+%Y-%m-%d %H:%M:%S') (SLURM array job ${SLURM_JOB_ID}) ===="
    echo "# ==== Notes: ${MESSAGE} ===="
} >> "$RESULTS_PATH"

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
    --crown_radius       $CROWN_RADIUS \
    --min_density_ratio  $MIN_DENSITY_RATIO \
    --use_color          $USE_COLOR \
    --color_weight       $COLOR_WEIGHT \
    --min_exg            $MIN_EXG \
    --min_cluster_height $MIN_CLUSTER_HEIGHT \
    --job_id            $SLURM_JOB_ID \
    --notes             "$NOTES" \
    --results_path      $RESULTS_PATH \
    --save              $SAVE \
    --trial_name        $TRIAL_NAME