#!/bin/bash
#SBATCH --job-name=saving_clusters                        # the name of your job
#SBATCH --output=/scratch/hcr64/saving_clusters.txt      # this is the file your $
#SBATCH --error=/scratch/hcr64/saving_clusters.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection                  # your work directory
#SBATCH --time=40:00                            # (max time) 40 min (shorte$
#SBATCH --mem=1000

# file paths
LOCAL_CLUSTERS=/home/hcr64/Pinyon-Detection/clusters/
DRIVE_PATH=gdrive:Sunset_Crater_trial/clusters/

# exit if any command fails
set -e

# say how many files are being deleted
echo "Deleting $(rclone size $DRIVE_PATH) folder..."

# clear the folder before cloning
rclone delete $DRIVE_PATH

echo "Saving $(du -sh "$LOCAL_CLUSTERS" | cut -f1) clusters to $FILE_PATH..."

# copy the folder to the drive
rclone copy $LOCAL_CLUSTERS $DRIVE_PATH