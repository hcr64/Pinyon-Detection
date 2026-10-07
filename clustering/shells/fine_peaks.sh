#!/bin/bash
#SBATCH --job-name=fine_peaks
#SBATCH --output=/scratch/hcr64/%j.txt
#SBATCH --error=/scratch/hcr64/%j.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection/
#SBATCH --time=04:00:00
#SBATCH --mem=32G

module load miniforge3/26.3.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate open3d_env

# smooth_sigma is in pixels at the native 3.67 cm grid: 3 = production, 1.5 = lighter
for SIGMA in 3 1.5 0.75; do
    python -u clustering/compute_fine_peaks.py \
        --trial_name Sunset_sfm_trial \
        --smooth_sigma $SIGMA \
        --search_radius_m 1.5 \
        --min_height 0.5 \
        --tag s${SIGMA}_r1.5
done