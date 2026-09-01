"""Conditional VAE for 6-hour weather forecasting.

Architecture:
- Encoder: Conv2D layers with stride=2 downsampling → flatten → μ and log_σ heads
- Decoder: FC to spatial, transpose-conv upsampling → reconstruction
- Conditioning: the input x_t is concatenated with conditioning context (implicit in the x_t -> x_t+1 forecasting task)

Design decisions:
- Stride-2 convolutions preserve information better than pooling
- Latent dimension: 64 (balance between expressivity and training stability)
- Loss: reconstruction MSE + β*KL(N(μ,σ²)||N(0,I)) with β=1.0
"""

import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransposeConvBlock(nn.Module):
    """TransposeConv2D → BatchNorm → ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 4)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VAEEncoder(nn.Module):
    """Encodes (T, C, H, W) input to latent distribution parameters."""
    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 64,
        h: int = 91,
        w: int = 180,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_h, self.input_w = h, w

        # Encoder path: Conv2D with stride=2 downsampling
        self.enc1 = ConvBlock(in_channels, 64, kernel_size=3, stride=2)     # (h/2, w/2) = (45, 90)
        self.enc2 = ConvBlock(64, 128, kernel_size=3, stride=2)             # (h/4, w/4) = (22, 45)
        self.enc3 = ConvBlock(128, 256, kernel_size=3, stride=2)            # (h/8, w/8) = (11, 22)

        # Compute flattened size after downsampling by a factor of 8 in each spatial dim
        # h_out = ceil(h / 8) = ceil(91 / 8) = 12; w_out = ceil(w / 8) = ceil(180 / 8) = 23
        # Account for stride-induced rounding by computing actual output size
        import math
        self.flat_h = math.ceil(h / 8)
        self.flat_w = math.ceil(w / 8)
        self.flat_size = 256 * self.flat_h * self.flat_w

        # Latent heads
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = x.view(x.size(0), -1)

        # Latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)

        # Reparameterization sample
        z = mu + std * torch.randn_like(std)

        return z, mu, logvar


class VAEDecoder(nn.Module):
    """Decodes latent code z to reconstructed field."""
    def __init__(
        self,
        latent_dim: int = 64,
        out_channels: int = 2,
        h: int = 91,
        w: int = 180,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_h, self.out_w = h, w

        # After 3 stride-2 downsamples in encoder: h/8, w/8
        import math
        self.flat_h = math.ceil(h / 8)
        self.flat_w = math.ceil(w / 8)
        self.spatial_channels = 256

        # Latent to spatial
        self.fc = nn.Linear(latent_dim, self.spatial_channels * self.flat_h * self.flat_w)

        # Decoder path: TransposeConv2D with stride=2 upsampling
        self.dec1 = TransposeConvBlock(256, 128, kernel_size=4, stride=2)   # (h/4, w/4)
        self.dec2 = TransposeConvBlock(128, 64, kernel_size=4, stride=2)    # (h/2, w/2)
        self.dec3 = TransposeConvBlock(64, 32, kernel_size=4, stride=2)     # (h, w) [approximately]

        # Output layer
        self.output = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Latent to spatial
        x = self.fc(z)
        x = x.view(z.size(0), self.spatial_channels, self.flat_h, self.flat_w)

        # Decode
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        # Crop/pad to exact shape if needed (transpose conv can be slightly larger)
        if x.shape[2] != self.out_h or x.shape[3] != self.out_w:
            x = x[:, :, :self.out_h, :self.out_w]

        # Output (no activation; normalized space assumption)
        x = self.output(x)

        return x


class VAE(nn.Module):
    """Full VAE model for weather field forecasting."""
    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 64,
        h: int = 91,
        w: int = 180,
    ):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, h, w)
        self.decoder = VAEDecoder(latent_dim, in_channels, h, w)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, logvar) without sampling."""
        _, mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent code."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode and decode.
        
        Returns:
            recon_x: reconstructed input
            mu: latent mean
            logvar: latent log-variance
        """
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from standard normal prior."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)
