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
git clone https://github.com/ndwang/beam_vae.git $PSCRATCH/vae
cd $PSCRATCH/vae

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
| `checkpoint_freq` | 50 | Save checkpoint every N epochs |
| `wandb.enabled` | false | Enable Weights & Biases logging |
| `wandb.project` | beam-vae | W&B project name |
| `wandb.offline` | true | Offline mode (for NERSC) |

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
├── config.yaml              # Full configuration (for reproducibility)
├── <run_name>.pth          # Final model weights
├── <run_name>_best.pth     # Best model checkpoint (lowest val loss)
├── <run_name>_epoch{N}.pth  # Periodic checkpoints
├── <run_name>_history.csv  # Training/validation loss history
└── wandb/                   # W&B logs (if enabled)
```

**History CSV Contents** (`<run_name>_history.csv`):
| Column | Description |
|--------|-------------|
| `epoch` | Epoch number |
| `train_total` | Training total loss (recon + β×KL) |
| `train_recon` | Training reconstruction loss |
| `train_kl` | Training KL divergence |
| `val_total` | Validation total loss |
| `val_recon` | Validation reconstruction loss |
| `val_kl` | Validation KL divergence |

**Run Naming**: Auto-generated run names use a concise format:
```
# Format: latent{dim}_beta{beta}_{YYMMDD}_{HHMM}
latent64_beta1e-05_260126_1430
```
The timestamp ensures uniqueness while keeping names readable.

**Checkpoint Contents:**
| Key | Description |
|-----|-------------|
| `epoch` | Training epoch number |
| `model_state_dict` | Model weights |
| `optimizer_state_dict` | Optimizer state |
| `scheduler_state_dict` | Scheduler state |
| `train_loss` | Training total loss |
| `train_recon_loss` | Training reconstruction loss |
| `train_kl_loss` | Training KL divergence |
| `val_loss` | Validation total loss |
| `val_recon_loss` | Validation reconstruction loss |
| `val_kl_loss` | Validation KL divergence |
| `beta` | KL divergence weight (β-VAE) |

### Resuming Training

Resume training from a checkpoint if interrupted or to continue training:

```bash
# Resume from best checkpoint
python scripts/train.py --resume runs/my_run/vae_best.pth

# Resume from specific epoch checkpoint
python scripts/train.py --resume runs/my_run/vae_epoch100.pth

# Resume with modified config (e.g., more epochs)
python scripts/train.py --resume runs/my_run/vae_best.pth training.epochs=500
```

The resume functionality:
- Restores model weights, optimizer state, and scheduler state
- Continues from the saved epoch number
- Preserves the best validation loss for checkpointing
- Warns if beta differs between checkpoint and current config

### Monitoring Training

Training prints per-epoch metrics:
```
Epoch 1/300 | Train: 0.012345 | Val: 0.013456 | LR: 5.00e-04
Epoch 2/300 | Train: 0.011234 | Val: 0.012345 | LR: 5.00e-04
...
```

## Weights & Biases Integration

Track experiments, visualize metrics, and manage model artifacts with W&B.

### Installation

```bash
conda activate vae
pip install wandb
wandb login  # One-time setup (from login node with internet)
```

### Basic Usage

```bash
# Enable W&B logging
python scripts/train.py training.wandb.enabled=true

# Customize W&B settings
python scripts/train.py \
    training.wandb.enabled=true \
    training.wandb.project=my-vae-experiments \
    training.wandb.offline=false
```

### Configuration

W&B settings in `configs/training/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wandb.enabled` | false | Enable/disable W&B logging |
| `wandb.project` | beam-vae | W&B project name |
| `wandb.entity` | null | W&B team/username (null = default) |
| `wandb.offline` | true | Offline mode (sync later) |
| `wandb.tags` | [] | Optional tags for run organization |
| `wandb.notes` | null | Optional run description |

### NERSC Offline Workflow

Use offline mode (default) to avoid internet access during training:

**SLURM jobs:** W&B logs are automatically synced at the end of each job.

**Manual runs:** Sync logs afterwards from login node:
```bash
# Sync all offline runs
./slurm/sync_wandb.sh

# Or sync specific run
wandb sync runs/<run_name>/wandb/offline-run-*
```

**View on W&B dashboard:** Visit https://wandb.ai to see your synced runs.

### Logged Metrics

W&B tracks per-epoch metrics:
- `train/total_loss` - Total training loss
- `train/recon_loss` - Reconstruction loss
- `train/kl_loss` - KL divergence
- `val/total_loss` - Validation total loss
- `val/recon_loss` - Validation reconstruction loss
- `val/kl_loss` - Validation KL divergence
- `learning_rate` - Current learning rate

### Checkpoint Tracking

W&B logs checkpoint metadata (file paths and metrics) without uploading the actual checkpoint files:
- **Best model**: Path, epoch, and validation loss tracked in run summary
- **Periodic checkpoints**: Path logged at each checkpoint interval
- **Checkpoint files remain local** - easily accessible via file paths in W&B dashboard

This approach keeps W&B runs lightweight while maintaining full checkpoint traceability.

### Checkpointing

Independent of W&B, the trainer saves:
- **Best model**: `<run_name>_best.pth` (updated when validation loss improves)
- **Periodic**: `<run_name>_epoch{N}.pth` (every `checkpoint_freq` epochs, default 50)
- **Final model**: `<run_name>.pth` (end of training)

Adjust checkpoint frequency:
```bash
python scripts/train.py training.checkpoint_freq=25  # Save every 25 epochs
```

### Example: Multi-Run Sweep with W&B

Use the SLURM scan scripts for parallel sweeps:

```bash
# Edit slurm/submit_1d_scan.sh to configure your sweep, then:
sbatch slurm/submit_1d_scan.sh
# W&B logs are automatically synced at the end of the job
```

Then compare all runs on the W&B dashboard with interactive plots and parallel coordinates.

## SLURM Jobs

Submit SLURM jobs in the top directory. SLURM logs are written to `logs/` (create this directory before first submission).

### Single Run

```bash
sbatch slurm/submit_single.sh
```

Edit `RUN_PREFIX` and `OVERRIDES` in the script to configure the run.

### 1D Parameter Scan

Run a sweep over a single parameter using 4 GPUs (1 node) in parallel:

```bash
sbatch slurm/submit_1d_scan.sh
```

Edit `PARAM_NAME` and `PARAM_VALUES` in the script.

### 2D Grid Search

Run a grid search over two parameters:

```bash
sbatch slurm/submit_2d_grid.sh
```

Edit `PARAM1_*` and `PARAM2_*` variables in the script.

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
│   ├── submit_single.sh     # Single training run
│   ├── submit_1d_scan.sh    # 1D parameter sweep
│   ├── submit_2d_grid.sh    # 2D grid search
│   └── sync_wandb.sh        # Sync W&B for manual runs
├── logs/                    # SLURM output logs (not in git)
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
│       ├── config.py       # Config loading with CLI overrides
│       ├── validation.py   # Pydantic config schema validation
│       ├── activations.py
│       ├── logging.py      # W&B callback classes
│       └── wandb_init.py   # W&B initialization
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
