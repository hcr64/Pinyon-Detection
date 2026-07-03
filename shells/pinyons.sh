#!/bin/bash
#SBATCH --job-name=pinyon_main
#SBATCH --output=/scratch/hcr64/%j.txt
#SBATCH --error=/scratch/hcr64/%j.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection
#SBATCH --time=3:00:00
#SBATCH --mem=96G

source /home/hcr64/Pinyon-Detection/open3d_env/bin/activate

TRIAL_NAME="Sunset_sfm_trial"

# DBSCAN clustering (unused with CHM method, keep at 2.0)
EPS=2.0

# green filter — 0.025 is the sweet spot, wider range made no difference
GREEN=0.025

# maximum crown radius in metres for cluster_by_chm_peaks
RADIUS=3.0

# maximum GPS-to-cluster distance for label matching
MAX_DISTANCE=4.0

# minimum points per cluster — may need to increase now that clusters include all points, not just green
MIN_POINTS=200

# voxel downsampling size — 0.05/0.06/0.08 all scored the same, 0.08 is fastest
VOXEL_SIZE=0.08

# minimum distance between density peaks when splitting large clusters
MIN_PEAK_DISTANCE=3.0

# k neighbours used for density-weighted center and peak detection
K=40

# minimum canopy height to count as a tree peak — 1.0 beats 1.5
MIN_HEIGHT=1.0

# local max window radius for find_chm_peaks — 2.5 gave best score (0.771)
SEARCH_RADIUS_M=3.0

# clear out the image & log folders before it start
# find trial_data/$TRIAL_NAME/logs/* -mmin +15 -type f -delete
# find trial_data/$TRIAL_NAME/images/* -mmin +15 -type f -delete

# touch everything in the data folder so it doesn't get thrown away
touch /scratch/hcr64/Pinyon-Detection/data/$TRIAL_NAME/point_cloud/*

echo "Task ${SLURM_ARRAY_TASK_ID}: eps=$EPS green=$GREEN crown=$RADIUS max_dist=$MAX_DISTANCE min_pts=$MIN_POINTS voxel=$VOXEL_SIZE mpd=$MIN_PEAK_DISTANCE k=$K min_h=$MIN_HEIGHT sr=$SEARCH_RADIUS_M"

# clustering + GPS labeling only. Run train_model.sh (or `python
# train_model.py --trial_name $TRIAL_NAME`) separately once this finishes.
python -u run_clustering.py \
    --eps               $EPS \
    --green_threshold   $GREEN \
    --max_radius        $RADIUS \
    --max_distance       $MAX_DISTANCE \
    --min_points        $MIN_POINTS \
    --voxel_size        $VOXEL_SIZE \
    --min_peak_distance $MIN_PEAK_DISTANCE \
    --k                 $K \
    --min_height        $MIN_HEIGHT \
    --search_radius_m   $SEARCH_RADIUS_M \
    --job_id            $SLURM_JOB_ID \
    --trial_name        $TRIAL_NAME