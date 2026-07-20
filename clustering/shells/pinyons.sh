#!/bin/bash
#SBATCH --job-name=pinyon_main
#SBATCH --output=/scratch/hcr64/%j.txt
#SBATCH --error=/scratch/hcr64/%j.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection/
#SBATCH --time=3:00:00
#SBATCH --mem=64G

source /home/hcr64/Pinyon-Detection/open3d_env/bin/activate

TRIAL_NAME="Sunset_sfm_trial"

# DBSCAN clustering (unused with CHM method, keep at 2.0)
EPS=2.0

# green filter — currently DEAD (STEPS['Clean_Pointcloud']=False means
# clean_up_pointcloud() never runs), value kept for when that's re-enabled
GREEN=0.025

# maximum crown radius for cluster_by_chm_peaks — swept and confirmed best
# among {3.0, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0} at 0.8347 (197/236)
RADIUS=4.25

# maximum GPS-to-cluster distance for label matching — plateaus at 4.0-4.5,
# but max_distance=4.25 combined with max_radius=4.25 measurably HURT
# (0.8136 vs 0.8347), so stick with 4.0 rather than the wider plateau edge
MAX_DISTANCE=4.0

# minimum points per cluster
MIN_POINTS=40

# voxel downsampling size
VOXEL_SIZE=0.08

# minimum distance between density peaks when splitting large clusters —
# confirmed flat/insensitive across 2.75-3.25, kept at original default
MIN_PEAK_DISTANCE=3.0

# k neighbours — DEAD (K = MIN_POINTS forced unconditionally in
# run_clustering.py), value has no independent effect
K=40

# minimum canopy height to count as a tree peak — plateaus 0.25-0.5,
# 0.5 chosen as the safer edge of that plateau (further from ground-noise
# floor); min_height=0.3-0.4 results still pending re-run confirmation
MIN_HEIGHT=0.5

# local max window radius for find_chm_peaks — swept optimum at 2.5
SEARCH_RADIUS_M=2.5

# assumed 1-sigma GPS horizontal error for the Gaussian label-matching cost
# — confirmed to NOT affect matching_score, kept at prior value
GPS_SIGMA=4.0

# Gaussian smoothing (px) applied to CHM before peak detection — swept
# optimum at 3, combined with search_radius_m=2.5 and max_radius=4.25
SMOOTH_SIGMA=3

# clear out the image & log folders before it start
# find trial_data/$TRIAL_NAME/logs/* -mmin +15 -type f -delete
# find trial_data/$TRIAL_NAME/images/* -mmin +15 -type f -delete

# touch everything in the data folder so it doesn't get thrown away
touch /scratch/hcr64/Pinyon-Detection/data/$TRIAL_NAME/point_cloud/*

echo "Task ${SLURM_ARRAY_TASK_ID}: eps=$EPS green=$GREEN crown=$RADIUS max_dist=$MAX_DISTANCE min_pts=$MIN_POINTS voxel=$VOXEL_SIZE mpd=$MIN_PEAK_DISTANCE k=$K min_h=$MIN_HEIGHT sr=$SEARCH_RADIUS_M"

# keep false for sweeps
SAVE=True

# clustering + GPS labeling only. Run train_model.sh (or `python
# train_model.py --trial_name $TRIAL_NAME`) separately once this finishes.
python -u clustering/run_clustering.py \
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
    --gps_sigma         $GPS_SIGMA \
    --smooth_sigma      $SMOOTH_SIGMA \
    --job_id            $SLURM_JOB_ID \
    --save              $SAVE \
    --trial_name        $TRIAL_NAME