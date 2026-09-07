#!/bin/bash
#SBATCH --job-name=saving_labeled_clusters
#SBATCH --output=/scratch/hcr64/saving_labeled_clusters.txt
#SBATCH --error=/scratch/hcr64/saving_labeled_clusters.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection
#SBATCH --time=60:00
#SBATCH --mem=1000

set -e

TRIAL_NAME=Sunset_sfm_trial

# a function to save a dir to google drive dir
# arg 1: drive path, arg 2: local path
save_clusters() {
    echo "Deleting $(rclone size $1) folder..."
    rclone delete $1 --tpslimit 5

    echo "Saving $(du -sh "$2" | cut -f1) to $1..."
    rclone copy $2 $1 --tpslimit 5

    echo 
}

# labeled clusters
LOCAL_PATH=/scratch/hcr64/Pinyon-Detection/data/$TRIAL_NAME/labeled_clusters/
DRIVE_PATH=gdrive:Sunset_Crater_trial/labeled_clusters/

# call the function on labeled clusters
save_clusters $DRIVE_PATH $LOCAL_PATH

# all clusters
LOCAL_PATH=/scratch/hcr64/Pinyon-Detection/data/$TRIAL_NAME/clusters/
DRIVE_PATH=gdrive:Sunset_Crater_trial/clusters/

# call the function on all clusters
echo "Deleting $(rclone size $DRIVE_PATH) folder..."
rclone delete $DRIVE_PATH --tpslimit 5

echo "Saving $(du -sh "$LOCAL_PATH" | cut -f1) to $DRIVE_PATH..."
rclone copy $LOCAL_PATH $DRIVE_PATH --tpslimit 5