#!/bin/bash
#SBATCH --job-name=vae_2d_grid
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5089
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nw285@cornell.edu

# ============================================
# 2D GRID SEARCH
# ============================================
# Usage: sbatch submit_2d_grid.sh
# Runs a grid of param1 x param2 combinations
# ============================================

# --- CONFIGURATION ---
PARAM1_NAME="model.latent_dim"
PARAM1_VALUES=(16 32 64 128)

PARAM2_NAME="training.beta"
PARAM2_VALUES=(1e-7 1e-6 1e-5 1e-4)

FIXED_OVERRIDES=""  # Additional fixed overrides (optional)

cd /pscratch/sd/n/ndwang/vae
ml load conda
conda activate vae

export SRUN_ARGS="--exact --ntasks 1 --gpus 1 --cpus-per-task 32"

# Generate all combinations and run
run_combo() {
    local val1=$1
    local val2=$2
    local p1_short=$(echo $PARAM1_NAME | sed 's/.*\.//')
    local p2_short=$(echo $PARAM2_NAME | sed 's/.*\.//')
    local ts=$(date +%y%m%d_%H%M)
    local run_name="${p1_short}_${val1}_${p2_short}_${val2}_${ts}"
    srun $SRUN_ARGS python scripts/train.py \
        ${PARAM1_NAME}=${val1} \
        ${PARAM2_NAME}=${val2} \
        run_name=${run_name} \
        ${FIXED_OVERRIDES} \
        training.wandb.enabled=true \
        > logs/${p1_short}_${val1}_${p2_short}_${val2}.log 2>&1
}
export -f run_combo
export PARAM1_NAME PARAM2_NAME FIXED_OVERRIDES SRUN_ARGS

parallel -j 4 --delay 0.2 run_combo ::: "${PARAM1_VALUES[@]}" ::: "${PARAM2_VALUES[@]}"

echo "Grid search complete: ${#PARAM1_VALUES[@]} x ${#PARAM2_VALUES[@]} = $((${#PARAM1_VALUES[@]} * ${#PARAM2_VALUES[@]})) configs"

# Upload all W&B logs after training completes
echo "Syncing W&B logs..."
for dir in runs/*/wandb/offline-run-*; do
    [ -d "$dir" ] && wandb sync "$dir"
done
echo "W&B sync complete."
