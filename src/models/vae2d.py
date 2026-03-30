"""
2D Variational Autoencoder (VAE) for multi-channel images.

This module implements a flexible 2D VAE architecture with an encoder-decoder
structure. The encoder downsamples using strided Conv2d blocks; the decoder
upsamples using Upsample + Conv2d blocks. Designed for inputs like 15-channel
64x64 images.
"""

from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import logging
from src.utils.activations import get_activation

logger = logging.getLogger(__name__)


class EncoderBlock2D(nn.Module):
    """
    Encoder block for 2D inputs using Conv2d downsampling.

    Structure:
        [ReflectPad] -> Conv2d(stride=2 if downsample) -> [BN] -> Act -> [Dropout] -> Downsample(x2)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        padding: ReflectPad2d padding size (0 disables pad wrapper)
        activation: Activation function name
        batch_norm: Whether to use BatchNorm2d
        dropout_rate: Dropout probability (0 disables)
        downsample: If True, first conv uses stride=2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = 'relu',
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        
        self.activation = get_activation(activation)
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate and dropout_rate > 0 else nn.Identity()
        self.downsample_conv = (
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True)
            if downsample else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolve first, then downsample
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.downsample_conv(x)
        return x


class DecoderBlock2D(nn.Module):
    """
    Decoder block for 2D inputs using Upsample + Conv2d.

    Structure:
        Upsample(x2) -> [ReflectPad] -> Conv2d -> [BN] -> Act -> [Dropout]

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        padding: ReflectPad2d padding size (0 disables pad wrapper)
        activation: Activation function name
        batch_norm: Whether to use BatchNorm2d
        dropout_rate: Dropout probability (0 disables)
        upsample_mode: 'bilinear' (default) or 'nearest'
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = 'relu',
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
        upsample_mode: str = 'bilinear',
    ) -> None:
        super().__init__()

        
        self.activation = get_activation(activation)
        padding = kernel_size // 2

        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False if upsample_mode == 'bilinear' else None)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate and dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample first, then convolve
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class VAE2D(nn.Module):
    """
    2D Variational Autoencoder with Conv2d encoder and Upsample+Conv2d decoder.

    - Input: (batch, channels=15, H=64, W=64) by default
    - Encoder: series of EncoderBlock2D with strided Conv2d downsampling
    - Bottleneck: AdaptiveAvgPool2d -> Linear heads to mu and logvar
    - Decoder: Linear projection -> reshape -> series of DecoderBlock2D
    - Output: reconstructed input (same number of channels as input)

    Config fields (under config['model']):
        input_channels, hidden_channels, latent_dim, input_size, kernel_size,
        padding, activation, batch_norm, dropout_rate, weight_init,
        use_reparameterization
    """

    def __init__(self, config: Dict[str, Any], norm_stats: Optional[Dict[str, torch.Tensor]] = None) -> None:
        super().__init__()
        model_config = config.get('model', {})

        self.input_channels = int(model_config.get('input_channels', 15))
        hidden_channels: List[int] = list(model_config.get('hidden_channels', [32, 64]))
        self.latent_dim = int(model_config.get('latent_dim', 64))
        self.input_size = int(model_config.get('input_size', 64))  # assumes square input
        kernel_size = int(model_config.get('kernel_size', 3))
        activation = str(model_config.get('activation', 'relu'))
        batch_norm = bool(model_config.get('batch_norm', True))
        dropout_rate = float(model_config.get('dropout_rate', 0.0))
        self.weight_init = str(model_config.get('weight_init', 'kaiming_normal'))
        output_activation = str(model_config.get('output_activation', 'sigmoid'))
        self.use_reparameterization = bool(model_config.get('use_reparameterization', True))
        self.n_scales = int(model_config.get('n_scales', 6))
        self.n_centroids = int(model_config.get('n_centroids', 6))

        if self.input_size % (2 ** len(hidden_channels)) != 0:
            raise ValueError(
                f"input_size={self.input_size} is not divisible by 2**{len(hidden_channels)}; "
                f"downsampling would not land on integer spatial size."
            )

        # Encoder
        self.encoder_blocks: nn.ModuleList = nn.ModuleList()
        in_ch = self.input_channels
        for out_ch in hidden_channels:
            block = EncoderBlock2D(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                downsample=True,
            )
            self.encoder_blocks.append(block)
            in_ch = out_ch

        # Bottleneck heads
        self.decoder_start_hw: int = self.input_size // (2 ** len(hidden_channels))
        bottleneck_features = hidden_channels[-1] * self.decoder_start_hw * self.decoder_start_hw
        # Final projection to latent-dim space before parameter heads
        # Scales and centroids are concatenated to flattened conv features
        self.fc_bottleneck = nn.Linear(bottleneck_features + self.n_scales + self.n_centroids, self.latent_dim)
        self.bottleneck_activation = get_activation(activation)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        # Decoder projection
        self.fc_proj = nn.Linear(self.latent_dim, hidden_channels[-1] * self.decoder_start_hw * self.decoder_start_hw)

        # Scale prediction head: recovers physical scales from latent vector
        self.scale_head = nn.Linear(self.latent_dim, self.n_scales)

        # Centroid prediction head: recovers beam orbit from latent vector
        self.centroid_head = nn.Linear(self.latent_dim, self.n_centroids)

        # Decoder blocks
        self.decoder_blocks: nn.ModuleList = nn.ModuleList()
        rev_channels = list(reversed(hidden_channels))
        in_ch = rev_channels[0]
        for out_ch in rev_channels[1:]:
            block = DecoderBlock2D(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                upsample_mode='bilinear',
            )
            self.decoder_blocks.append(block)
            in_ch = out_ch

        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=self.input_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # Output normalization
        self.output_normalization = get_activation(output_activation)

        # Initialize weights
        self._initialize_weights()

        # Normalization stats for scale/centroid targets (z-score per dimension).
        # Heads predict in normalized space; use denormalize_*() for physical values.
        if norm_stats is not None:
            self.register_buffer('scale_mean', norm_stats['scale_mean'])
            self.register_buffer('scale_std', norm_stats['scale_std'])
            self.register_buffer('centroid_mean', norm_stats['centroid_mean'])
            self.register_buffer('centroid_std', norm_stats['centroid_std'])
        else:
            # Placeholders: mean=0, std=1 is identity normalization
            self.register_buffer('scale_mean', torch.zeros(self.n_scales))
            self.register_buffer('scale_std', torch.ones(self.n_scales))
            self.register_buffer('centroid_mean', torch.zeros(self.n_centroids))
            self.register_buffer('centroid_std', torch.ones(self.n_centroids))

        # Log model info
        summary = self.get_model_summary()
        logger.info(f"VAE2D initialized with {summary['total_parameters']:,} total parameters")
        logger.info(f"Trainable parameters: {summary['trainable_parameters']:,}")
        logger.info(f"Architecture: {self.input_channels} channels -> {' -> '.join(map(str, hidden_channels))} -> latent_dim={self.latent_dim} -> {self.input_channels} channels")
        logger.info(f"Input size: {self.input_size}x{self.input_size}, Decoder start size: {self.decoder_start_hw}x{self.decoder_start_hw}")

    def encode(self, x: torch.Tensor, scales: torch.Tensor, centroids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder path
        current = x
        for block in self.encoder_blocks:
            current = block(current)

        h = current.flatten(1)  # (B, bottleneck_features)
        h = torch.cat([h, scales, centroids], dim=1)  # (B, bottleneck_features + n_scales + n_centroids)
        h = self.fc_bottleneck(h)
        h = self.bottleneck_activation(h)
        mu = torch.clamp(self.fc_mu(h), min=-10, max=10)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_scales = self.scale_head(z)  # (B, n_scales)
        pred_centroids = self.centroid_head(z)  # (B, n_centroids)
        h = self.fc_proj(z)
        h = h.view(z.size(0), -1, self.decoder_start_hw, self.decoder_start_hw)
        current = h
        for block in self.decoder_blocks:
            current = block(current)

        current = self.final_upsample(current)
        current = self.final_conv(current)

        return self.output_normalization(current), pred_scales, pred_centroids

    def forward(self, x: torch.Tensor, scales: torch.Tensor, centroids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, scales, centroids)
        if self.use_reparameterization and self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        recon, pred_scales, pred_centroids = self.decode(z)
        return recon, pred_scales, pred_centroids, mu, logvar

    def normalize_scales(self, scales: torch.Tensor) -> torch.Tensor:
        """Normalize log(scales) targets to zero mean, unit variance."""
        return (torch.log(scales) - self.scale_mean) / self.scale_std

    def normalize_centroids(self, centroids: torch.Tensor) -> torch.Tensor:
        """Normalize centroid targets to zero mean, unit variance."""
        return (centroids - self.centroid_mean) / self.centroid_std

    def denormalize_scales(self, pred: torch.Tensor) -> torch.Tensor:
        """Convert normalized scale predictions back to physical scales."""
        return torch.exp(pred * self.scale_std + self.scale_mean)

    def denormalize_centroids(self, pred: torch.Tensor) -> torch.Tensor:
        """Convert normalized centroid predictions back to physical centroids."""
        return pred * self.centroid_std + self.centroid_mean

    def get_model_summary(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'model_name': 'VAE2D',
            'input_channels': self.input_channels,
            'latent_dim': self.latent_dim,
            'input_size': self.input_size,
            'decoder_start_hw': self.decoder_start_hw,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }

    def _initialize_weights(self) -> None:
        """Initialize model weights according to selected method."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if self.weight_init == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif self.weight_init == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif self.weight_init == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                if getattr(module, 'bias', None) is not None:
                    nn.init.constant_(module.bias, 0)


# Example usage and testing
if __name__ == "__main__":
    # Test model with sample config mirroring style in UNet3D
    test_config = {
        'model': {
            'input_channels': 15,
            'hidden_channels': [32, 64],
            'latent_dim': 64,
            'input_size': 64,
            'kernel_size': 3,
            'activation': 'relu',
            'batch_norm': True,
            'dropout_rate': 0.0,
            'weight_init': 'kaiming_normal',
            'use_reparameterization': True
        }
    }

    model = VAE2D(test_config)
    print(f"Model summary: {model.get_model_summary()}")

    batch_size = 2
    x = torch.randn(batch_size, 15, 64, 64)
    scales = torch.rand(batch_size, 6)
    centroids = torch.rand(batch_size, 6)
    with torch.no_grad():
        recon, pred_scales, pred_centroids, mu, logvar = model(x, scales, centroids)
        print(f"Recon shape: {recon.shape}")
        print(f"Pred scales shape: {pred_scales.shape}")
        print(f"Pred centroids shape: {pred_centroids.shape}")
        print(f"Mu shape: {mu.shape}, LogVar shape: {logvar.shape}")
