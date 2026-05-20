#!/bin/bash
#SBATCH --job-name=pinyon_main                        # the name of your job
#SBATCH --output=/scratch/hcr64/pinyon_output.txt      # this is the file your $
#SBATCH --error=/scratch/hcr64/pinyon_output.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection                  # your work directory
#SBATCH --time=40:00                            # (max time) 40 min (shorte$
#SBATCH --mem=100G                              # (total mem) 100GB of memory

# get in the open3d environment
# source open3d_env/bin/activate

# Run your application: precede the application command with 'srun'
# A couple example applications...
# srun date
srun python --version

# the trial name
TRIAL_NAME="Sunset_sfm_trial"

# variables
EPS=2.0
GREEN=0.012
RADIUS=1.5
MIN_POINTS=70

# run the main function
python -u main.py \
    --eps $EPS \
    --green_threshold $GREEN \
    --max_radius $RADIUS \
    --min_points $MIN_POINTS \
    --trial_name $TRIAL_NAME > /scratch/hcr64/pinyon_output.txt
