"""Loss functions for VAE training."""

import torch
import torch.nn.functional as F


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    loss_config: dict = None,
) -> torch.Tensor:
    """Compute reconstruction loss between reconstructed and target tensors.

    Args:
        recon: Reconstructed tensor from decoder.
        target: Original input tensor.
        loss_type: Type of loss - 'mse', 'weighted_mse', or 'bce'.
        loss_config: Extra parameters for specific loss types.

    Returns:
        Scalar loss tensor.
    """
    if loss_config is None:
        loss_config = {}

    if loss_type == "mse":
        # Sum over pixels (H, W), mean over batch and channels.
        # Each channel is a normalized density — its summed squared error
        # is the natural per-channel reconstruction metric.
        return ((recon - target) ** 2).sum(dim=(2, 3)).mean()
    elif loss_type == "weighted_mse":
        # Weight each pixel by target intensity so signal regions dominate.
        # Floor prevents zero gradient on background (avoids hallucination).
        floor = loss_config.get("floor", 1e-6)
        weight = torch.clamp(target, min=floor)
        return (weight * (recon - target) ** 2).sum(dim=(2, 3)).mean()
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
    """Compute scale prediction loss (MSE).

    Both pred and target should be in the same space — normalized if the
    dataset applies normalization, raw log-space otherwise.

    Args:
        pred_scales: (B, n_scales) predicted scales from decoder.
        target_scales: (B, n_scales) target scales from dataset.

    Returns:
        Scalar loss tensor.
    """
    return F.mse_loss(pred_scales, target_scales)


def centroid_loss(
    pred_centroids: torch.Tensor,
    target_centroids: torch.Tensor,
) -> torch.Tensor:
    """Compute centroid prediction loss (MSE).

    Both pred and target should be in the same space — normalized if the
    dataset applies normalization, raw otherwise.

    Args:
        pred_centroids: (B, n_centroids) predicted centroids from decoder.
        target_centroids: (B, n_centroids) target centroids from dataset.

    Returns:
        Scalar loss tensor.
    """
    return F.mse_loss(pred_centroids, target_centroids)


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
    pred_centroids: torch.Tensor = None,
    target_centroids: torch.Tensor = None,
    delta: float = 0.0,
    loss_config: dict = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total VAE loss (reconstruction + KL + scale + centroid).

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
        pred_centroids: (B, n_centroids) predicted centroids from decoder.
        target_centroids: (B, n_centroids) ground-truth centroids.
        delta: Weight for centroid reconstruction loss.

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss, scale_loss_val, centroid_loss_val).
    """
    recon_loss = reconstruction_loss(recon, target, loss_type, loss_config)
    kl_loss = kl_divergence(mu, logvar)

    s_loss = torch.tensor(0.0, device=recon.device)
    if pred_scales is not None and target_scales is not None and gamma > 0:
        s_loss = scale_loss(pred_scales, target_scales)

    c_loss = torch.tensor(0.0, device=recon.device)
    if pred_centroids is not None and target_centroids is not None and delta > 0:
        c_loss = centroid_loss(pred_centroids, target_centroids)

    total_loss = recon_loss + beta * kl_loss + gamma * s_loss + delta * c_loss

    return total_loss, recon_loss, kl_loss, s_loss, c_loss
