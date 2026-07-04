#!/bin/bash
#SBATCH --job-name=saving_clusters                        # the name of your job
#SBATCH --output=/scratch/hcr64/saving_clusters.txt      # this is the file your $
#SBATCH --error=/scratch/hcr64/saving_clusters.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection                  # your work directory
#SBATCH --time=40:00                            # (max time) 40 min (shorte$
#SBATCH --mem=1000

# the trial name
TRIAL_NAME=Sunset_sfm_trial

# file paths
LOCAL_WD=/home/hcr64/Pinyon-Detection/
LOCAL_PATHS=( "pre_split_clusters/" "trial_data/$TRIAL_NAME/labeled_clusters/" )

# Drive locations
DRIVE_WD=gdrive:Sunset_Crater_trial/
DRIVE_PATHS=( "pre_split_clusters/" "clusters/" "labeled_clusters/" )

# exit if any command fails
set -e

# iterate over all the paths
for index in "${!LOCAL_PATHS[@]}"; do

    # get the local and drive paths
    LOCAL_PATH="$LOCAL_WD${LOCAL_PATHS[$index]}"
    DRIVE_PATH="$DRIVE_WD${DRIVE_PATHS[$index]}"

    # clear the drive folder
    echo "Deleting $(rclone size $DRIVE_PATH) folder..."
    rclone delete $DRIVE_PATH

    # save the local folder to drive
    echo "Saving $(du -sh "$LOCAL_PATH" | cut -f1) to $DRIVE_PATH..."
    rclone copy $LOCAL_PATH $DRIVE_PATH

done
