# VAE Training Pipeline

Variational Autoencoder training pipeline for frequency map data on NERSC Perlmutter.

## Project Structure

```
├── configs/           # YAML configuration files
│   ├── model/        # Model architectures (vae2d, residual_vae2d)
│   ├── training/     # Training hyperparameters
│   └── data/         # Dataset paths
├── scripts/          # Entry point scripts
│   └── train.py      # Main training script
├── slurm/            # NERSC job submission scripts
└── src/              # Source code
    ├── models/       # VAE2D, ResidualVAE2D
    ├── data/         # FrequencyMapDataset
    ├── training/     # Trainer, losses
    └── utils/        # Config loading, activations
```

## Quick Commands

```bash
# Train with defaults
python scripts/train.py

# Override hyperparameters
python scripts/train.py model.latent_dim=128 training.beta=1e-5

# Use residual model
python scripts/train.py model=model/residual_vae2d.yaml

# Submit to SLURM
sbatch slurm/submit_job.sh
```

## Key Files

- `src/models/vae2d.py` - Standard VAE architecture
- `src/models/residual_vae2d.py` - VAE with residual blocks
- `src/training/trainer.py` - Training loop and checkpointing
- `src/utils/config.py` - YAML config loading with CLI overrides

## Data

- Input: 15-channel 64x64 frequency maps stored as `.npy` files
- Location: `/pscratch/sd/n/ndwang/frequency_maps/`
- Preprocessing: Min-max normalized, optionally log-transformed

## Conventions

- Config overrides use dot notation: `model.latent_dim=64`
- Run outputs saved to `runs/<run_name>/` with config.yaml for reproducibility
- SLURM scripts use GNU parallel for multi-GPU sweeps

## Environment

```bash
ml load conda
conda activate vae
```
