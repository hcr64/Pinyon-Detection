#!/bin/bash
#SBATCH --job-name=pinyon_sweep
#SBATCH --output=clustering/logs/job_%A_%a.out
#SBATCH --error=clustering/logs/job_%A_%a.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection/
#SBATCH --array=1-30
#SBATCH --time=02:00:00
#SBATCH --mem=92G
#SBATCH --partition=core

source open3d_env/bin/activate

TRIAL_NAME="Sunset_sfm_trial"

# files older than this get deleted
MINS=45

# clear out logs and images folder before beginning
find clustering/logs/* -mmin +$MINS -type f -delete
find clustering/trial_data/$TRIAL_NAME/images/* -mmin +$MINS -type f -delete

# print a divider in the results file 
RESULTS_PATH=clustering/trial_data/$TRIAL_NAME/results/results_Jul18.csv
if [ "${SLURM_ARRAY_TASK_ID}" == "${SLURM_ARRAY_TASK_MIN}" ]; then
    mkdir -p "$(dirname "$RESULTS_PATH")"
    {
        echo ""
        echo "# ==== sweep started $(date '+%Y-%m-%d %H:%M:%S') (SLURM array job ${SLURM_ARRAY_JOB_ID}) ===="
    } >> "$RESULTS_PATH"
fi

# skip comment and blank lines, then get row SLURM_ARRAY_TASK_ID
PARAMS=$(grep -v '^\s*#' clustering/params.txt | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")

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
GPS_SIGMA=$(        echo $PARAMS | awk '{print $11}')
SMOOTH_SIGMA=$(     echo $PARAMS | awk '{print $12}')

# keep false for sweeps
SAVE=False

echo "Task ${SLURM_ARRAY_TASK_ID}: eps=$EPS green=$GREEN crown=$RADIUS max_dist=$MAX_DISTANCE min_pts=$MIN_POINTS voxel=$VOXEL_SIZE mpd=$MIN_PEAK_DISTANCE k=$K min_h=$MIN_HEIGHT sr=$SEARCH_RADIUS_M gps_sigma=$GPS_SIGMA smooth_sigma=$SMOOTH_SIGMA"

# sweeps only ever needed clustering + GPS matching score — never the
# classifier comparison, which used to run (and clutter these logs) on
# every single array task via the old main.py
python -u clustering/run_clustering.py \
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
    --gps_sigma        $GPS_SIGMA \
    --smooth_sigma     $SMOOTH_SIGMA \
    --job_id           $SLURM_JOB_ID \
    --save             $SAVE \
    --trial_name       $TRIAL_NAME