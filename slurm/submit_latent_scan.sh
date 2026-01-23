#!/bin/bash
#SBATCH --job-name=latent_scan    # Job name
#SBATCH --output=%x-%j.out         # Output file (jobname-jobid.out)
#SBATCH --error=%x-%j.err          # Error file (jobname-jobid.err)
#SBATCH --time=03:00:00            # Time limit (HH:MM:SS) - Adjust as needed
#SBATCH --constraint=gpu           # (Specific to NERSC/Perlmutter)
#SBATCH --qos=regular              # debug limits to 30 minutes
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# --- NODE CONFIGURATION ---
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32         # CPUs (set higher than --num-workers)
#SBATCH --gpus=4                   # Number of GPUs requested

cd /pscratch/sd/n/ndwang/vae
ml load conda
conda activate vae
mkdir -p logs

# srun flags:
#   --exact:         Don't let processes share resources
#   --ntasks 1:      Run 1 task
#   --gpus 1:        Give it 1 GPU
#   --cpus-per-task: Give it 32 CPUs
export SRUN_ARGS="--exact --ntasks 1 --gpus 1 --cpus-per-task 32"

parallel -j 4 --delay 0.2 \
    "srun $SRUN_ARGS python scripts/train.py model.latent_dim={} > logs/latent_{}.log 2>&1" \
    :::: latent_scan.txt
