#!/bin/bash
#SBATCH --job-name=pinyon_sweep
#SBATCH --output=clustering/logs/sweep_%A_%a.out
#SBATCH --error=clustering/logs/sweep_%A_%a.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection/
#SBATCH --array=1-23
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --partition=core

module load miniforge3/26.3.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate open3d_env

TRIAL_NAME="Sunset_sfm_trial"
MESSAGE="N/A"

# get a user inputted message after the -m command
while getopts "m:" opt; do
  case ${opt} in
    m ) MESSAGE="$OPTARG" ;;
    \? ) echo "Usage: $0 -m \"your message\"" >&2; exit 1 ;;
  esac
done

# delete log files that are not running
bash shells/clear_garbage_files.sh

# clear old images (MINS was previously undefined here, which made find error out)
# MINS=15
# find clustering/trial_data/$TRIAL_NAME/images/* -mmin +$MINS -type f -delete 2>/dev/null || true

# MUST match PATHS['GPS_results'] in constants.py. New file because the
# column set changed — the old results_Jul18.csv header no longer matches.
RESULTS_PATH=trial_data/$TRIAL_NAME/results/
if [ "${SLURM_ARRAY_TASK_ID}" == "${SLURM_ARRAY_TASK_MIN}" ]; then
    mkdir -p "$(dirname "$RESULTS_PATH")"
    {
        echo ""
        echo "# ==== sweep started $(date '+%Y-%m-%d %H:%M:%S') (SLURM array job ${SLURM_ARRAY_JOB_ID}) ===="
        echo "# ==== Notes: ${MESSAGE} ===="
        echo
    } >> "$RESULTS_PATH"
fi

# skip comment and blank lines, then get row SLURM_ARRAY_TASK_ID
PARAMS=$(grep -v '^\s*#' clustering/params.txt | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$PARAMS" ]; then
    echo "No params row for task ${SLURM_ARRAY_TASK_ID}. Check --array against params.txt."
    exit 1
fi

MAX_RADIUS=$(          echo $PARAMS | awk '{print $1}')
CROWN_RADIUS=$(        echo $PARAMS | awk '{print $2}')
MAX_DISTANCE=$(        echo $PARAMS | awk '{print $3}')
MIN_POINTS=$(          echo $PARAMS | awk '{print $4}')
MIN_PEAK_DISTANCE=$(   echo $PARAMS | awk '{print $5}')
K=$(                   echo $PARAMS | awk '{print $6}')
MIN_DENSITY_RATIO=$(   echo $PARAMS | awk '{print $7}')
MIN_HEIGHT=$(          echo $PARAMS | awk '{print $8}')
SEARCH_RADIUS_M=$(     echo $PARAMS | awk '{print $9}')
SMOOTH_SIGMA=$(        echo $PARAMS | awk '{print $10}')
USE_COLOR=$(           echo $PARAMS | awk '{print $11}')
COLOR_WEIGHT=$(        echo $PARAMS | awk '{print $12}')
MIN_EXG=$(             echo $PARAMS | awk '{print $13}')
MIN_CLUSTER_HEIGHT=$(  echo $PARAMS | awk '{print $14}')

# convert 1/0 to True/False for str2bool
if [ "$USE_COLOR" == "1" ] || [ "$USE_COLOR" == "True" ]; then USE_COLOR=True; else USE_COLOR=False; fi

# keep false for sweeps
SAVE=False

echo "Task ${SLURM_ARRAY_TASK_ID}: max_r=$MAX_RADIUS crown_r=$CROWN_RADIUS max_dist=$MAX_DISTANCE min_pts=$MIN_POINTS mpd=$MIN_PEAK_DISTANCE k=$K density_ratio=$MIN_DENSITY_RATIO min_h=$MIN_HEIGHT sr=$SEARCH_RADIUS_M smooth=$SMOOTH_SIGMA color=$USE_COLOR cw=$COLOR_WEIGHT min_exg=$MIN_EXG min_clust_h=$MIN_CLUSTER_HEIGHT"

python -u clustering/run_clustering.py \
    --max_radius         $MAX_RADIUS \
    --crown_radius       $CROWN_RADIUS \
    --max_distance       $MAX_DISTANCE \
    --min_points         $MIN_POINTS \
    --min_peak_distance  $MIN_PEAK_DISTANCE \
    --k                  $K \
    --min_density_ratio  $MIN_DENSITY_RATIO \
    --min_height         $MIN_HEIGHT \
    --search_radius_m    $SEARCH_RADIUS_M \
    --smooth_sigma       $SMOOTH_SIGMA \
    --use_color          $USE_COLOR \
    --color_weight       $COLOR_WEIGHT \
    --min_exg            $MIN_EXG \
    --min_cluster_height $MIN_CLUSTER_HEIGHT \
    --job_id             $SLURM_JOB_ID \
    --notes              "$MESSAGE" \
    --save               $SAVE \
    --trial_name         $TRIAL_NAME