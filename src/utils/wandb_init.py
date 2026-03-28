"""W&B initialization utility for VAE training."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .logging import LoggingCallback, NoOpCallback, WandbCallback


def init_wandb(
    config: Dict[str, Any],
    run_name: str,
    output_dir: Path,
) -> Tuple[Optional[Any], LoggingCallback]:
    """Initialize Weights & Biases if enabled.

    Args:
        config: Full training configuration dictionary.
        run_name: Name for the run.
        output_dir: Directory for run outputs (used for W&B dir).

    Returns:
        Tuple of (wandb.Run or None, LoggingCallback).
        Returns NoOpCallback if W&B is disabled or not installed.
    """
    # Get wandb config section
    training_cfg = config.get('training', {})
    wandb_cfg = training_cfg.get('wandb', {})

    # Check if W&B is enabled
    if not wandb_cfg.get('enabled', False):
        return None, NoOpCallback()

    # Try to import wandb
    try:
        import wandb
    except ImportError:
        print("Warning: wandb not installed. Logging disabled.")
        return None, NoOpCallback()

    # Set offline mode if configured (default True for NERSC)
    if wandb_cfg.get('offline', True):
        os.environ['WANDB_MODE'] = 'offline'

    # Initialize W&B run
    try:
        run = wandb.init(
            project=wandb_cfg.get('project', 'vae-training'),
            entity=wandb_cfg.get('entity'),
            group=wandb_cfg.get('group'),
            name=run_name,
            config=config,
            dir=str(output_dir),
            tags=wandb_cfg.get('tags', []),
            notes=wandb_cfg.get('notes'),
            reinit=True,
        )

        callback = WandbCallback(run)

        mode = "offline" if wandb_cfg.get('offline', True) else "online"
        print(f"W&B initialized ({mode} mode): {run.name}")

        return run, callback

    except Exception as e:
        print(f"Warning: Failed to initialize W&B: {e}")
        return None, NoOpCallback()
