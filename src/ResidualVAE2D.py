"""
2D Variational Autoencoder (VAE) for multi-channel images.
Upgraded with Residual Blocks.
"""

from typing import Dict, Any, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Assuming you have this helper, otherwise replace with direct nn.ReLU, etc.
from activations import get_activation 

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    """
    Standard Residual Block:
    x -> Conv3x3 -> BN -> Act -> Dropout -> Conv3x3 -> BN -> (+) -> Act
         |                                                    |
         ----------------- (Shortcut) -------------------------
    
    Includes a 1x1 Conv shortcut if input/output channels differ.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        activation: str = 'relu',
        batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        self.act_fn = get_activation(activation)
        
        # 1. First Convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # 2. Second Convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        
        # 3. Shortcut Connection
        # If channels change, we need a 1x1 conv to match dimensions for addition
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fn(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.act_fn(out)
        
        return out


class EncoderBlock2D(nn.Module):
    """
    Encoder block: Residual Processing -> Strided Conv Downsampling
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = 'relu',
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        
        # 1. Feature Extraction (Residual Block)
        # We transform in_channels -> out_channels here
        self.res_block = ResidualBlock(
            in_channels, out_channels, activation, batch_norm, dropout_rate
        )
        
        # 2. Downsampling (Strided Convolution)
        # We keep channels same (out -> out) during downsampling
        self.downsample = nn.Identity()
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=not batch_norm),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x)
        x = self.downsample(x)
        return x


class DecoderBlock2D(nn.Module):
    """
    Decoder block: Upsample -> Residual Processing
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = 'relu',
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
        upsample_mode: str = 'bilinear',
    ) -> None:
        super().__init__()

        # 1. Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False if upsample_mode == 'bilinear' else None)
        
        # 2. Feature Processing (Residual Block)
        # We transform in_channels -> out_channels here
        self.res_block = ResidualBlock(
            in_channels, out_channels, activation, batch_norm, dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.res_block(x)
        return x


class ResidualVAE2D(nn.Module):
    """
    Residual VAE 2D Architecture.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        model_config = config.get('model', {})

        self.input_channels = int(model_config.get('input_channels', 15))
        hidden_channels: List[int] = list(model_config.get('hidden_channels', [32, 64]))
        self.latent_dim = int(model_config.get('latent_dim', 64))
        self.input_size = int(model_config.get('input_size', 64))
        activation = str(model_config.get('activation', 'relu'))
        batch_norm = bool(model_config.get('batch_norm', True))
        dropout_rate = float(model_config.get('dropout_rate', 0.0))
        self.weight_init = str(model_config.get('weight_init', 'kaiming_normal'))
        self.use_reparameterization = bool(model_config.get('use_reparameterization', True))

        if self.input_size % (2 ** len(hidden_channels)) != 0:
            raise ValueError(f"Input size {self.input_size} incompatible with {len(hidden_channels)} downsampling layers.")

        # --- Encoder ---
        self.encoder_blocks: nn.ModuleList = nn.ModuleList()
        in_ch = self.input_channels
        
        # Note: In VAEs, it's often good to project input to hidden dim first
        # to avoid the first ResBlock doing 15->32 immediately, but doing it 
        # inside the block is also acceptable and saves a layer.
        
        for out_ch in hidden_channels:
            block = EncoderBlock2D(
                in_channels=in_ch,
                out_channels=out_ch,
                activation=activation,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                downsample=True,
            )
            self.encoder_blocks.append(block)
            in_ch = out_ch

        # --- Bottleneck ---
        self.decoder_start_hw: int = self.input_size // (2 ** len(hidden_channels))
        bottleneck_features = hidden_channels[-1] * self.decoder_start_hw * self.decoder_start_hw
        
        self.fc_bottleneck = nn.Linear(bottleneck_features, self.latent_dim)
        self.bottleneck_activation = get_activation(activation)
        
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        # --- Decoder ---
        self.fc_proj = nn.Linear(self.latent_dim, hidden_channels[-1] * self.decoder_start_hw * self.decoder_start_hw)

        self.decoder_blocks: nn.ModuleList = nn.ModuleList()
        rev_channels = list(reversed(hidden_channels))
        in_ch = rev_channels[0]
        
        # We iterate through targets: e.g. [64, 32] -> targets 32 (then final proj to input)
        # Note on structure: UNets/VAEs are symmetric.
        # Encoder: 15->32 (down), 32->64 (down)
        # Decoder: 64->32 (up), 32->output
        
        for out_ch in rev_channels[1:]:
            block = DecoderBlock2D(
                in_channels=in_ch,
                out_channels=out_ch,
                activation=activation,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate,
                upsample_mode='bilinear',
            )
            self.decoder_blocks.append(block)
            in_ch = out_ch

        # Final Upsample + Convolution to get back to input size and channels
        # (Since the loop above stops before the final layer)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Using a small ResBlock here or just a Conv is a choice. 
        # Standard is just a final convolution.
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch) if batch_norm else nn.Identity(),
            get_activation(activation),
            nn.Conv2d(in_ch, self.input_channels, kernel_size=1) # 1x1 projection to exact output channels
        )

        self.output_normalization = get_activation('sigmoid')

        self._initialize_weights()

        summary = self.get_model_summary()
        logger.info(f"VAE2D (Residual) initialized with {summary['total_parameters']:,} params")

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        current = x
        for block in self.encoder_blocks:
            current = block(current)

        h = current.flatten(1)
        h = self.fc_bottleneck(h)
        h = self.bottleneck_activation(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # clamp for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10) 
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_proj(z)
        h = h.view(z.size(0), -1, self.decoder_start_hw, self.decoder_start_hw)
        
        current = h
        for block in self.decoder_blocks:
            current = block(current)

        current = self.final_upsample(current)
        current = self.final_conv(current)
        
        return self.output_normalization(current)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        if self.use_reparameterization and self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        recon = self.decode(z)
        return recon, mu, logvar

    def get_model_summary(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'model_name': 'ResidualVAE2D',
            'input_channels': self.input_channels,
            'latent_dim': self.latent_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if self.weight_init == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif self.weight_init == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                if getattr(module, 'bias', None) is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

if __name__ == "__main__":
    # Test Config
    test_config = {
        'model': {
            'input_channels': 15,
            'hidden_channels': [32, 64],
            'latent_dim': 64,
            'input_size': 64,
            'activation': 'relu',
            'batch_norm': True,
            'dropout_rate': 0.0,
        }
    }
    model = ResidualVAE2D(test_config)
    x = torch.randn(2, 15, 64, 64)
    r, m, l = model(x)
    print(f"In: {x.shape}, Out: {r.shape}")