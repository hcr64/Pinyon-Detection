#!/bin/bash
#SBATCH --job-name=pinyon_train
#SBATCH --output=/scratch/hcr64/%j.txt
#SBATCH --error=/scratch/hcr64/%j.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection
#SBATCH --time=00:30:00
#SBATCH --mem=16G

module load miniforge3/26.3.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate open3d_env

TRIAL_NAME="Sunset_sfm_trial"

python -u modelling/train_model.py \
    --trial_name $TRIAL_NAME \
    --embeddings