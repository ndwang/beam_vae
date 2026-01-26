#!/bin/bash
#SBATCH --job-name=vae_single
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# SINGLE TRAINING RUN
# ============================================
# Usage: sbatch submit_single.sh
# Modify the OVERRIDES variable below to change hyperparameters
# ============================================

# --- CONFIGURATION ---
RUN_PREFIX="baseline"                    # Descriptive prefix for this run
OVERRIDES="model.latent_dim=256 training.beta=0"

cd /pscratch/sd/n/ndwang/vae
ml load conda
conda activate vae

# Generate unique run name with timestamp
RUN_NAME="${RUN_PREFIX}_$(date +%y%m%d_%H%M)"

# Run training with W&B in offline mode
python scripts/train.py $OVERRIDES run_name=${RUN_NAME} training.wandb.enabled=true

# Upload W&B logs after training completes
echo "Syncing W&B logs..."
wandb sync runs/${RUN_NAME}/wandb/offline-run-* --sync-all
