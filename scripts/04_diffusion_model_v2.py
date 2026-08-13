"""Simplified Diffusion model for sparse field reconstruction (U-Net architecture).

Inputs:
  - xt: noisy state at timestep t, shape (B, 2, H, W)
  - y: sparse observations, shape (B, 2, H, W)
  - M: binary mask, shape (B, 1, H, W) - 1 where observed, 0 where missing
  - t: diffusion timestep, scalar - encoded via sinusoidal embedding

Output:
  - Predicted noise ε, shape (B, 2, H, W) - ε-prediction objective

Architecture: Simple 3-level encoder-decoder with skip connections
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep t into sinusoidal positional embedding."""
        is_scalar = t.dim() == 0
        t = t.unsqueeze(0) if is_scalar else t
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, dtype=t.dtype, device=t.device) / half_dim
        )
        
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embed = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        
        return embed.squeeze(0) if is_scalar else embed


class DiffusionUNet(nn.Module):
    """Simplified U-Net for diffusion-based sparse field reconstruction."""
    def __init__(
        self,
        in_channels: int = 5,  # 2 (xt) + 2 (y) + 1 (M)
        out_channels: int = 2,  # predicted noise for 2 fields
        time_dim: int = 128,
        base_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        
        # Sinusoidal time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # Simple convolutional encoder-decoder structure
        c = base_channels
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, c * 2),
            nn.SiLU(),
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, c * 4),
            nn.SiLU(),
        )
        
        # Bottleneck with time embedding injection
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 4 + time_dim, c * 4, kernel_size=3, padding=1),
            nn.GroupNorm(32, c * 4),
            nn.SiLU(),
            nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=1),
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, c * 2),
            nn.SiLU(),
        )
        
        self.dec2_input = nn.Conv2d(c * 2, c * 2, kernel_size=1)  # Projection for skip concat
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        
        self.dec3_input = nn.Conv2d(c, c, kernel_size=1)  # Projection for skip concat
        
        # Output
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    @staticmethod
    def _match_spatial(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Center-crop tensors to a common spatial size for robust skip additions."""
        h = min(a.shape[-2], b.shape[-2])
        w = min(a.shape[-1], b.shape[-1])
        return a[..., :h, :w], b[..., :h, :w]

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        M: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: noisy state (B, 2, H, W)
            t: timestep (scalar or B,)
            y: sparse observations (B, 2, H, W)
            M: binary mask (B, 1, H, W)
        
        Returns:
            predicted noise (B, 2, H, W)
        """
        # Encode timestep
        time_embed = self.time_embed(t)
        if time_embed.dim() == 1:
            time_embed = time_embed.unsqueeze(0).expand(x.size(0), -1)
        
        # Concatenate input
        h = torch.cat([x, y, M], dim=1)
        
        # Encoder
        h1 = self.enc1(h)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        
        # Bottleneck with time embedding (broadcast)
        B, C, H, W = h3.shape
        time_embed_spatial = time_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)
        h_bottle = torch.cat([h3, time_embed_spatial], dim=1)
        h_bottle = self.bottleneck(h_bottle)
        
        # Decoder with skip connections
        d1 = self.dec1(h_bottle)
        s1 = self.dec2_input(h2)
        d1, s1 = self._match_spatial(d1, s1)
        d1 = d1 + s1  # Skip connection (addition instead of concat)
        
        d2 = self.dec2(d1)
        s2 = self.dec3_input(h1)
        d2, s2 = self._match_spatial(d2, s2)
        d2 = d2 + s2  # Skip connection
        
        # Output
        out = self.out_conv(d2)
        if out.shape[-2:] != x.shape[-2:]:
            out = out[..., : x.shape[-2], : x.shape[-1]]
        
        return out
