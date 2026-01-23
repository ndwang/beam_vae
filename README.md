# VAE Training Pipeline

A Variational Autoencoder (VAE) training pipeline for frequency map reconstruction, designed for NERSC Perlmutter.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [SLURM Jobs](#slurm-jobs)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Models](#models)

## Installation

### On NERSC Perlmutter

```bash
# Clone the repository
git clone <repo-url> /pscratch/sd/$USER/vae
cd /pscratch/sd/$USER/vae

# Load conda and create environment
ml load conda
conda env create -f environment.yml
conda activate vae
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Quick Start

```bash
# Train with default settings
python scripts/train.py

# Train with custom latent dimension
python scripts/train.py model.latent_dim=128

# Train residual VAE with custom beta
python scripts/train.py model=model/residual_vae2d.yaml training.beta=1e-5
```

## Configuration

The pipeline uses a YAML-based configuration system with composable configs and CLI overrides.

### Config Structure

```
configs/
├── default.yaml              # Main config (references sub-configs)
├── model/
│   ├── vae2d.yaml           # Standard VAE
│   └── residual_vae2d.yaml  # Residual VAE
├── training/
│   └── default.yaml         # Training hyperparameters
└── data/
    ├── frequency_maps.yaml      # Min-max normalized data
    └── frequency_maps_log.yaml  # Log-transformed data
```

### Default Configuration

**Model** (`configs/model/vae2d.yaml`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_channels` | 15 | Number of input channels |
| `hidden_channels` | [32, 64, 128, 256, 512] | Encoder/decoder channel sizes |
| `latent_dim` | 64 | Latent space dimension |
| `input_size` | 64 | Spatial size (64x64) |
| `activation` | relu | Activation function |
| `dropout_rate` | 0.0 | Dropout probability |

**Training** (`configs/training/default.yaml`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 300 | Number of training epochs |
| `batch_size` | 256 | Batch size |
| `lr` | 5e-4 | Learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `beta` | 0.0 | KL divergence weight (β-VAE) |
| `val_split` | 0.1 | Validation split ratio |
| `seed` | 42 | Random seed |

### CLI Overrides

Override any config value using dot notation:

```bash
# Single override
python scripts/train.py model.latent_dim=32

# Multiple overrides
python scripts/train.py model.latent_dim=32 training.lr=1e-4 training.epochs=500

# Switch sub-config
python scripts/train.py model=model/residual_vae2d.yaml

# Custom run name
python scripts/train.py run_name=my_experiment
```

## Training

### Basic Training

```bash
python scripts/train.py
```

This will:
1. Load config from `configs/default.yaml`
2. Create dataset from configured path
3. Train the model
4. Save outputs to `runs/<run_name>/`

### Output Files

Each training run creates a directory with:
```
runs/<run_name>/
├── config.yaml           # Full configuration (for reproducibility)
├── <run_name>.pth       # Model weights
└── <run_name>_history.csv  # Training/validation losses
```

### Monitoring Training

Training prints per-epoch metrics:
```
Epoch 1/300 | Train: 0.012345 | Val: 0.013456 | LR: 5.00e-04
Epoch 2/300 | Train: 0.011234 | Val: 0.012345 | LR: 5.00e-04
...
```

## SLURM Jobs

### Single Job with Array

Submit a latent dimension sweep:

```bash
sbatch slurm/submit_job.sh
```

This runs 8 jobs (latent_dim = 8, 12, 16, 24, 32, 64, 128, 256).

### Multi-GPU Parallel Sweep

Run multiple experiments on 4 GPUs simultaneously:

```bash
# Latent dimension sweep
sbatch slurm/submit_latent_scan.sh

# Beta sweep
sbatch slurm/submit_beta_scan.sh
```

### Custom SLURM Job

```bash
#!/bin/bash
#SBATCH --job-name=vae_train
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m5089

cd /pscratch/sd/$USER/vae
ml load conda
conda activate sc_surrogate

python scripts/train.py model.latent_dim=64 training.epochs=500
```

## Visualization

### Plot Loss Curves

```bash
python scripts/visualize_loss.py runs/<run_name>/<run_name>_history.csv --save
```

Creates a PNG with training/validation curves for total loss, reconstruction loss, and KL divergence.

### Visualize Reconstructions

```bash
python scripts/visualize_recon.py \
    runs/<run_name>/<run_name>.pth \
    /pscratch/sd/n/ndwang/frequency_maps/frequency_maps_minmax.npy \
    --sample-index 0 \
    --channels 0 1 2 3 4
```

Creates a comparison plot showing original, reconstruction, and error for selected channels.

## Project Structure

```
vae/
├── configs/                 # YAML configuration files
│   ├── default.yaml
│   ├── model/
│   ├── training/
│   └── data/
├── scripts/                 # Entry point scripts
│   ├── train.py            # Main training script
│   ├── visualize_loss.py   # Loss curve plotting
│   └── visualize_recon.py  # Reconstruction visualization
├── slurm/                   # NERSC job scripts
│   ├── submit_job.sh
│   ├── submit_latent_scan.sh
│   └── submit_beta_scan.sh
├── src/                     # Source code
│   ├── models/             # VAE architectures
│   │   ├── vae2d.py
│   │   └── residual_vae2d.py
│   ├── data/               # Dataset classes
│   │   └── dataset.py
│   ├── training/           # Training utilities
│   │   ├── trainer.py
│   │   └── losses.py
│   └── utils/              # Utilities
│       ├── config.py
│       └── activations.py
├── data/                    # Dataset files (not in git)
├── runs/                    # Training outputs (not in git)
├── CLAUDE.md
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Models

### VAE2D

Standard convolutional VAE with:
- 5 encoder blocks with strided convolution downsampling
- Bottleneck with FC layers for μ and log(σ²)
- 5 decoder blocks with bilinear upsampling
- Sigmoid output activation

### ResidualVAE2D

Enhanced VAE with residual connections:
- Residual blocks before each down/upsample operation
- Skip connections for better gradient flow
- ~2x more parameters than standard VAE

### Choosing a Model

| Use Case | Recommended Model |
|----------|-------------------|
| Fast iteration, baseline | `vae2d` |
| Higher quality reconstruction | `residual_vae2d` |
| Limited GPU memory | `vae2d` with smaller `hidden_channels` |

## Tips

### Hyperparameter Tuning

1. **Latent dimension**: Start with 64, try 32-256 range
2. **Beta (KL weight)**: Start with 0 (pure AE), increase to 1e-5 to 1e-3 for disentanglement
3. **Learning rate**: 5e-4 works well, reduce if training unstable

### Debugging

```bash
# Quick test run (2 epochs, small batch)
python scripts/train.py training.epochs=2 training.batch_size=32

# Check for NaN losses
# The trainer will raise an error if NaN is detected
```

### Reproducibility

Each run saves its full config. To reproduce:

```bash
python scripts/train.py --config runs/previous_run/config.yaml
```
