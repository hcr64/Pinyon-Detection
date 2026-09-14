#!/bin/bash
#SBATCH --job-name=saving_multimatch                           # the name of your job
#SBATCH --output=/scratch/hcr64/saving_multimatch.txt          # stdout
#SBATCH --error=/scratch/hcr64/saving_multimatch.err           # stderr
#SBATCH --chdir=/home/hcr64/Pinyon-Detection                   # your work directory
#SBATCH --time=40:00                                            # (max time) 40 min
#SBATCH --mem=1000

# the trial name
TRIAL_NAME=Sunset_sfm_trial

# file paths
LOCAL_WD=/home/hcr64/Pinyon-Detection/
LOCAL_PATH="${LOCAL_WD}trial_data/$TRIAL_NAME/multi_match_clusters/"

# Drive location
DRIVE_WD=gdrive:Sunset_Crater_trial/
DRIVE_PATH="${DRIVE_WD}multi_match_clusters/"

# exit if any command fails
set -e

# make sure the local folder actually exists before trying to sync it
if [ ! -d "$LOCAL_PATH" ]; then
    echo "Local path $LOCAL_PATH does not exist. Did main.py run with multi_match_save_path set? Exiting."
    exit 1
fi

# clear the drive folder first, same pattern as save_clusters_to_drive.sh
echo "Deleting $(rclone size $DRIVE_PATH 2>/dev/null || echo 'empty/new') folder..."
rclone delete "$DRIVE_PATH" || true

# save the local folder to drive
echo "Saving $(du -sh "$LOCAL_PATH" | cut -f1) to $DRIVE_PATH..."
rclone copy "$LOCAL_PATH" "$DRIVE_PATH"

echo "Done. Multi-match clusters synced to $DRIVE_PATH"