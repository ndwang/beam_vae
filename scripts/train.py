#!/usr/bin/env python
"""Main training script for VAE models.

Usage:
    # Use default config
    python scripts/train.py

    # Use specific config
    python scripts/train.py --config configs/default.yaml

    # Override specific values
    python scripts/train.py model.latent_dim=128 training.epochs=500

    # Use different model
    python scripts/train.py model=model/residual_vae2d.yaml

    # Combine config and overrides
    python scripts/train.py --config configs/default.yaml training.beta=1e-5

    # Resume from checkpoint
    python scripts/train.py --resume runs/my_run/vae_best.pth
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models import VAE2D, ResidualVAE2D
from src.data import FrequencyMapDataset
from src.training import Trainer
from src.utils import load_config, save_config, config_to_model_config, generate_run_name, init_wandb


def get_args():
    parser = argparse.ArgumentParser(
        description="VAE Training with YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Base directory for config files"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    # Collect remaining args as overrides
    args, overrides = parser.parse_known_args()
    args.overrides = overrides

    return args


def main():
    args = get_args()

    # Load configuration
    config = load_config(
        config_path=args.config,
        config_dir=args.config_dir,
        overrides=args.overrides,
    )

    # Extract sub-configs
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    data_cfg = config.get('data', {})

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    seed = training_cfg.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dataset setup
    dataset_path = data_cfg.get('path')
    if not dataset_path:
        raise ValueError("Dataset path not specified in config")

    full_dataset = FrequencyMapDataset(dataset_path)
    n_samples = len(full_dataset)
    val_split = training_cfg.get('val_split', 0.1)
    val_size = int(val_split * n_samples)
    train_size = n_samples - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    batch_size = training_cfg.get('batch_size', 256)
    num_workers = training_cfg.get('num_workers', 8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"Dataset: {n_samples} samples ({train_size} train, {val_size} val)")

    # Create model
    model_name = model_cfg.get('name', 'vae2d')
    model_config = config_to_model_config(config)

    if model_name == 'residual_vae2d':
        model = ResidualVAE2D(model_config)
    else:
        model = VAE2D(model_config)

    # Optimizer
    lr = training_cfg.get('lr', 5e-4)
    weight_decay = training_cfg.get('weight_decay', 1e-4)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Scheduler
    scheduler_cfg = training_cfg.get('scheduler', {})
    scheduler_name = scheduler_cfg.get('name', 'reduce_on_plateau')

    if scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.get('factor', 0.5),
            patience=scheduler_cfg.get('patience', 10)
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_cfg.get('epochs', 300)
        )
    else:
        scheduler = None

    # Output setup (before trainer so W&B can use the directory)
    run_name = config.get('run_name') or generate_run_name(config)
    output_dir = Path(config.get('output_dir', './runs')) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B (if enabled)
    wandb_run, logger_callback = init_wandb(config, run_name, output_dir)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        beta=training_cfg.get('beta', 0.0),
        loss_type=training_cfg.get('loss_type', 'mse'),
        grad_clip=training_cfg.get('grad_clip', 1.0),
        logger_callback=logger_callback,
    )

    # Load checkpoint if resuming
    if args.resume:
        resume_path = Path(args.resume)
        trainer.load_checkpoint(resume_path)

    # Save config to output directory
    save_config(config, output_dir / "config.yaml")

    # Train
    epochs = training_cfg.get('epochs', 300)
    checkpoint_freq = training_cfg.get('checkpoint_freq', 50)
    print(f"Starting training: {run_name}")
    print(f"Config saved to: {output_dir / 'config.yaml'}")

    try:
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            max_steps=training_cfg.get('max_steps'),
            save_dir=output_dir,
            model_name=run_name,
            checkpoint_freq=checkpoint_freq,
        )
        print("Training complete!")
    finally:
        logger_callback.finish()


if __name__ == "__main__":
    main()
