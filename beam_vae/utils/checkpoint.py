"""Shared checkpoint loading for evaluation / benchmark scripts."""

from pathlib import Path

import torch

from beam_vae.models import VAE2D, ResidualVAE2D
from beam_vae.utils.config import load_yaml


def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Load model and config from a checkpoint file.

    Returns:
        model        — loaded, eval-mode model on device
        model_name   — "vae2d" | "residual_vae2d"
        label        — short display label, e.g. "vae2d z256"
        config       — raw config dict from config.yaml
        ckpt         — raw checkpoint dict (for epoch/loss metadata)
    """
    ckpt_path = Path(ckpt_path)
    run_dir = ckpt_path.parent
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    config = load_yaml(config_path)

    model_name = config.get("model", {}).get("name", "vae2d")
    model_cls = ResidualVAE2D if model_name == "residual_vae2d" else VAE2D
    model = model_cls(config)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()

    latent_dim = config.get("model", {}).get("latent_dim", "?")
    label = f"{model_name} z{latent_dim}"
    return model, model_name, label, config, ckpt
