#!/bin/bash
#SBATCH --job-name=pinyon_train
#SBATCH --output=/scratch/hcr64/%j.txt
#SBATCH --error=/scratch/hcr64/%j.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection
#SBATCH --time=00:30:00
#SBATCH --mem=16G

source /home/hcr64/Pinyon-Detection/open3d_env/bin/activate

TRIAL_NAME="Sunset_sfm_trial"

# add --advanced to use run_advanced_classifiers() + label spreading
# instead of train_tree_classifier()
python -u train_model.py \
    --trial_name $TRIAL_NAME