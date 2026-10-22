#!/bin/bash
#SBATCH --job-name=pinyon_full_pipeline
#SBATCH --output=/scratch/hcr64/full_run_%j.txt
#SBATCH --error=/scratch/hcr64/full_run_%j.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection/
#SBATCH --time=4:00:00
#SBATCH --mem=64G

# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: run_clustering.py -> train_model.py -> collect plots.
#
# Output folder (created under OUT_ROOT):
#   <trial>_<jobid>_<timestamp>/
#       prediction_map/       prediction_map.png
#       confusion_matrices/   confusion_matrix.png, drought_..., combined_...,
#                             cm_<model>.png (from --advanced)
#       gps_overlap/          coordinate_overlap.png, GPS_Clusters.png
#       run_info.txt          parameters used
#
# Only files written DURING this run are copied (checked via a timestamp
# marker), so stale plots from earlier runs never end up in the folder.
# ─────────────────────────────────────────────────────────────────────────────

# access the venv
module load miniforge3/26.3.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate open3d_env

# exit immediately if either stage fails (don't train on stale dataframes)
set -e

# ── settings ─────────────────────────────────────────────────────────────────
TRIAL_NAME="Sunset_sfm_trial"
REPO=/home/hcr64/Pinyon-Detection
OUT_ROOT=$REPO/run_outputs

# clustering parameters (same as clustering/shells/pinyons.sh)
EPS=2.0
GREEN=0.025
RADIUS=4.25
MAX_DISTANCE=4.0
MIN_POINTS=40
VOXEL_SIZE=0.08
MIN_PEAK_DISTANCE=3.0
K=40
MIN_HEIGHT=0.5
SEARCH_RADIUS_M=2.5
GPS_SIGMA=4.0
SMOOTH_SIGMA=3

# modelling flags (same as modelling/shells/train_model.sh)
MODEL_FLAGS="--embeddings --advanced --drought"

# SAVE must be True: train_model.py reads the clusters + dataframes that
# run_clustering.py writes to disk.
SAVE=True

# ── paths ────────────────────────────────────────────────────────────────────
IMG_DIR=$REPO/trial_data/$TRIAL_NAME/images
RUN_ID="${SLURM_JOB_ID:-manual}"
OUT_DIR=$OUT_ROOT/${TRIAL_NAME}_${RUN_ID}_$(date +%Y%m%d_%H%M)

# marker file: anything in the images folder newer than this was made by this run
MARKER=$(mktemp)
touch "$MARKER"

# make sure the data folder doesn't get purged from scratch
touch /scratch/hcr64/Pinyon-Detection/data/$TRIAL_NAME/point_cloud/* 2>/dev/null || true

echo "Run $RUN_ID  trial=$TRIAL_NAME  start=$(date '+%H:%M:%S')"
echo "Output folder: $OUT_DIR"
echo

# ── stage 1: clustering + GPS labeling ───────────────────────────────────────
echo "========== STAGE 1: clustering =========="
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
    --job_id            $RUN_ID \
    --save              $SAVE \
    --trial_name        $TRIAL_NAME

echo
echo "Clustering finished at $(date '+%H:%M:%S')"
echo

# ── stage 2: modelling ───────────────────────────────────────────────────────
echo "========== STAGE 2: modelling =========="
python -u modelling/train_model.py \
    --trial_name $TRIAL_NAME \
    $MODEL_FLAGS

echo
echo "Modelling finished at $(date '+%H:%M:%S')"
echo

# ── stage 3: collect outputs ─────────────────────────────────────────────────
echo "========== STAGE 3: collecting outputs =========="

# usage: collect <subfolder> <filename pattern> [more patterns...]
collect() {
    local dest="$OUT_DIR/$1"; shift
    mkdir -p "$dest"
    for pat in "$@"; do
        find "$IMG_DIR" -maxdepth 1 -type f -name "$pat" -newer "$MARKER" \
            -exec cp {} "$dest"/ \;
    done
    local n
    n=$(find "$dest" -type f | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "  WARNING: nothing found for $(basename "$dest") (patterns: $*)"
    else
        echo "  $(basename "$dest"): $n file(s)"
    fi
}

mkdir -p "$OUT_DIR"

collect prediction_map      "prediction_map.png"
collect confusion_matrices  "confusion_matrix.png" \
                            "drought_confusion_matrix.png" \
                            "combined_confusion_matrix.png" \
                            "cm_*.png"
collect gps_overlap         "coordinate_overlap.png" \
                            "GPS_Clusters*.png"

# record what produced this folder
{
    echo "trial:        $TRIAL_NAME"
    echo "job id:       $RUN_ID"
    echo "finished:     $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "clustering:   eps=$EPS green=$GREEN max_radius=$RADIUS max_distance=$MAX_DISTANCE"
    echo "              min_points=$MIN_POINTS voxel=$VOXEL_SIZE min_peak_distance=$MIN_PEAK_DISTANCE k=$K"
    echo "              min_height=$MIN_HEIGHT search_radius_m=$SEARCH_RADIUS_M gps_sigma=$GPS_SIGMA smooth_sigma=$SMOOTH_SIGMA"
    echo "modelling:    $MODEL_FLAGS"
} > "$OUT_DIR/run_info.txt"

rm -f "$MARKER"

echo
echo "Done. Results in: $OUT_DIR"
echo "Time at completion: $(date '+%H:%M:%S')"