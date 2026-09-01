"""Minimal diffusion model for sparse field reconstruction.

This baseline uses a compact encoder-decoder with sinusoidal time embeddings
and intentionally omits skip connections to avoid shape mismatch instability
while validating end-to-end training and evaluation.
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
    """Minimal U-Net for diffusion-based sparse field reconstruction."""
    def __init__(
        self,
        in_channels: int = 5,  # 2 (xt) + 2 (y) + 1 (M)
        out_channels: int = 2,  # predicted noise for 2 fields
        time_dim: int = 128,
        base_channels: int = 64,
    ):
        super().__init__()
        
        # Sinusoidal time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        c = base_channels
        
        # Encoder (3 blocks, alternating conv+stride)
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, c, 3, 1, 1),
                nn.GroupNorm(8, c),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv2d(c, c * 2, 3, 2, 1),
                nn.GroupNorm(16, c * 2),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv2d(c * 2, c * 4, 3, 2, 1),
                nn.GroupNorm(32, c * 4),
                nn.SiLU(),
            ),
        ])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 4 + time_dim, c * 4, 3, 1, 1),
            nn.GroupNorm(32, c * 4),
            nn.SiLU(),
        )
        
        # Decoder (3 blocks, upsampling)
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1),
                nn.GroupNorm(16, c * 2),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(c * 2, c, 4, 2, 1),
                nn.GroupNorm(8, c),
                nn.SiLU(),
            ),
            nn.Conv2d(c, out_channels, 1),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Forward pass without skip connections."""
        # Time embedding
        time_embed = self.time_embed(t)
        if time_embed.dim() == 1:
            time_embed = time_embed.unsqueeze(0).expand(x.size(0), -1)
        
        # Concatenate input
        h = torch.cat([x, y, M], dim=1)
        
        # Encoder
        for block in self.enc_blocks:
            h = block(h)
        
        # Bottleneck with time (broadcast and concat)
        B, C, H, W = h.shape
        time_spatial = time_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)
        h = torch.cat([h, time_spatial], dim=1)
        h = self.bottleneck(h)
        
        # Decoder
        for block in self.dec_blocks[:-1]:
            h = block(h)
        
        # Final output (may crop to exact size if needed)
        out = self.dec_blocks[-1](h)
        
        # Crop to input size if transpose conv produced slightly different size
        if out.shape[2:] != x.shape[2:]:
            out = out[:, :, :x.shape[2], :x.shape[3]]
        
        return out
