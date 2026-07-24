#!/bin/bash
#SBATCH --job-name=pinyon_model_sweep
#SBATCH --output=modelling/logs/model_sweep_%A_%a.txt
#SBATCH --error=modelling/logs/model_sweep_%A_%a.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection
#SBATCH --array=1-11
#SBATCH --time=00:45:00
#SBATCH --mem=16G

source /home/hcr64/Pinyon-Detection/open3d_env/bin/activate

TRIAL_NAME="Sunset_sfm_trial"
RESULTS_PATH=/home/hcr64/Pinyon-Detection/modelling/trial_data/$TRIAL_NAME/results/model_sweep_results.csv

MINS=5
find modelling/logs/* -mmin +$MINS -type f -delete 


# path is relative to --chdir above -- model_params.txt lives under modelling/,
# not the repo root
PARAMS_FILE="modelling/model_params.txt"

# exit loudly instead of silently proceeding with empty params
set -e

if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: $PARAMS_FILE not found (cwd: $(pwd)). Exiting."
    exit 1
fi

PARAMS=$(grep -v '^\s*#' "$PARAMS_FILE" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$PARAMS" ]; then
    echo "ERROR: no params row found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in $PARAMS_FILE. Exiting."
    exit 1
fi

MODE=$(                    echo "$PARAMS" | awk '{print $1}')
ALPHA=$(                   echo "$PARAMS" | awk '{print $2}')
GAMMA=$(                   echo "$PARAMS" | awk '{print $3}')
CONFIDENCE_THRESHOLD=$(    echo "$PARAMS" | awk '{print $4}')
EVAL_RANDOM_STATE=$(       echo "$PARAMS" | awk '{print $5}')
SMOTE=$(                   echo "$PARAMS" | awk '{print $6}')
WEIGHTING=$(               echo "$PARAMS" | awk '{print $7}')
SELECTION=$(               echo "$PARAMS" | awk '{print $8}')
TUNING=$(                  echo "$PARAMS" | awk '{print $9}')

echo "Task ${SLURM_ARRAY_TASK_ID}: mode=$MODE alpha=$ALPHA gamma=$GAMMA conf=$CONFIDENCE_THRESHOLD seed=$EVAL_RANDOM_STATE smote=$SMOTE weighting=$WEIGHTING selection=$SELECTION tuning=$TUNING"

if [ "$MODE" == "advanced" ]; then
    python -u modelling/train_model.py \
        --trial_name          "$TRIAL_NAME" \
        --advanced \
        --alpha                "$ALPHA" \
        --gamma                "$GAMMA" \
        --confidence_threshold "$CONFIDENCE_THRESHOLD" \
        --eval_random_state    "$EVAL_RANDOM_STATE" \
        --job_id               "$SLURM_ARRAY_TASK_ID" \
        --results_path         "$RESULTS_PATH"
else
    python -u modelling/train_model.py \
        --trial_name  "$TRIAL_NAME" \
        --smote        "$SMOTE" \
        --weighting    "$WEIGHTING" \
        --selection    "$SELECTION" \
        --tuning       "$TUNING" \
        --job_id        "$SLURM_ARRAY_TASK_ID" \
        --results_path   "$RESULTS_PATH"
fi