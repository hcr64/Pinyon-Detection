#!/bin/bash
#SBATCH --job-name=pinyon_main                        # the name of your job
#SBATCH --output=/scratch/hcr64/pinyon_output.txt      # this is the file your $
#SBATCH --error=/scratch/hcr64/pinyon_output.err
#SBATCH --chdir=/home/hcr64/Pinyon-Detection                  # your work directory
#SBATCH --time=40:00                            # (max time) 40 min (shorte$
#SBATCH --mem=100000                              # (total mem) 100GB of memory

# get in the open3d environment
# source open3d_env/bin/activate

# Run your application: precede the application command with 'srun'
# A couple example applications...
# srun date
srun python --version

# run the main function
srun python -u main.py > /scratch/hcr64/pinyon_output.txt
