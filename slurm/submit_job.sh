#!/bin/bash
#SBATCH --job-name=sc_vae_train    # Job name
#SBATCH --output=%x-%j.out         # Output file (jobname-jobid.out)
#SBATCH --error=%x-%j.err          # Error file (jobname-jobid.err)
#SBATCH --time=03:00:00            # Time limit (HH:MM:SS) - Adjust as needed
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=16         # CPUs (set higher than --num-workers)
#SBATCH --gpus=1                   # Number of GPUs requested
#SBATCH --constraint=gpu           # (Specific to NERSC/Perlmutter)
#SBATCH --qos=regular              # debug limits to 30 minutes
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# --- JOB ARRAY ---
#SBATCH --array=0-7
# Scanning latent dimensions
LATENT_DIMS=(8 12 16 24 32 64 128 256)
CURRENT_LATENT=${LATENT_DIMS[$SLURM_ARRAY_TASK_ID]}

cd /pscratch/sd/n/ndwang/vae
ml load conda
conda activate vae

python scripts/train.py model.latent_dim=$CURRENT_LATENT
