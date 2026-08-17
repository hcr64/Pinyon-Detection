#!/bin/bash
#SBATCH --job-name=saving_labeled_clusters
#SBATCH --output=/scratch/hcr64/saving_labeled_clusters.txt
#SBATCH --error=/scratch/hcr64/saving_labeled_clusters.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection
#SBATCH --time=60:00
#SBATCH --mem=1000

TRIAL_NAME=Sunset_sfm_trial

LOCAL_PATH=/scratch/hcr64/Pinyon-Detection/data/$TRIAL_NAME/labeled_clusters/
DRIVE_PATH=gdrive:Sunset_Crater_trial/labeled_clusters/

set -e

echo "Deleting $(rclone size $DRIVE_PATH) folder..."
rclone delete $DRIVE_PATH --tpslimit 5

echo "Saving $(du -sh "$LOCAL_PATH" | cut -f1) to $DRIVE_PATH..."
rclone copy $LOCAL_PATH $DRIVE_PATH --tpslimit 5