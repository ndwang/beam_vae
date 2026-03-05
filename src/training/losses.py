"""Loss functions for VAE training."""

import torch
import torch.nn.functional as F


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse"
) -> torch.Tensor:
    """Compute reconstruction loss between reconstructed and target tensors.

    Args:
        recon: Reconstructed tensor from decoder.
        target: Original input tensor.
        loss_type: Type of loss - 'mse' or 'bce'.

    Returns:
        Scalar loss tensor.
    """
    if loss_type == "mse":
        return F.mse_loss(recon, target)
    elif loss_type == "bce":
        return F.binary_cross_entropy(recon, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence for VAE latent space.

    KL(q(z|x) || p(z)) where q is the encoder distribution and p is N(0,1).

    Args:
        mu: Mean of the latent distribution.
        logvar: Log variance of the latent distribution.

    Returns:
        Scalar KL divergence loss.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))


def scale_loss(
    pred_scales: torch.Tensor,
    target_scales: torch.Tensor,
) -> torch.Tensor:
    """Compute scale reconstruction loss in log-space.

    MSE on log(sigma) handles the wide dynamic range of physical scales.

    Args:
        pred_scales: (B, n_scales) predicted scales from decoder.
        target_scales: (B, n_scales) ground-truth scales.

    Returns:
        Scalar loss tensor.
    """
    return F.mse_loss(pred_scales, torch.log(target_scales))


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    loss_type: str = "mse",
    pred_scales: torch.Tensor = None,
    target_scales: torch.Tensor = None,
    gamma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total VAE loss (reconstruction + KL divergence + scale loss).

    Args:
        recon: Reconstructed tensor from decoder.
        target: Original input tensor.
        mu: Mean of the latent distribution.
        logvar: Log variance of the latent distribution.
        beta: Weight for KL divergence term (beta-VAE).
        loss_type: Type of reconstruction loss - 'mse' or 'bce'.
        pred_scales: (B, n_scales) predicted scales from decoder.
        target_scales: (B, n_scales) ground-truth scales.
        gamma: Weight for scale reconstruction loss.

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss, scale_loss_val).
    """
    recon_loss = reconstruction_loss(recon, target, loss_type)
    kl_loss = kl_divergence(mu, logvar)

    s_loss = torch.tensor(0.0, device=recon.device)
    if pred_scales is not None and target_scales is not None and gamma > 0:
        s_loss = scale_loss(pred_scales, target_scales)

    total_loss = recon_loss + beta * kl_loss + gamma * s_loss

    return total_loss, recon_loss, kl_loss, s_loss
