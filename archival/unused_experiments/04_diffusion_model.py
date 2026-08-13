"""Diffusion model for sparse field reconstruction (U-Net architecture).

Inputs:
  - xt: noisy state at timestep t, shape (B, 2, H, W) - the fields to reconstruct
  - y: sparse observations, shape (B, 2, H, W) - only observed regions have data
  - M: binary mask, shape (B, 1, H, W) - 1 where observed, 0 where missing
  - t: diffusion timestep, scalar - encoded via sinusoidal embedding

Output:
  - Predicted noise ε, shape (B, 2, H, W) - ε-prediction objective

Architecture:
  - Sinusoidal time embedding → FC → broadcast to spatial
  - U-Net with skip connections
  - Conditioning on y and M via concatenation
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
        """
        Embed timestep t ∈ [0, T-1] into sinusoidal positional embedding.
        
        Args:
            t: scalar or (B,) shaped tensor
        
        Returns:
            embedding: (dim,) or (B, dim)
        """
        is_scalar = t.dim() == 0
        t = t.unsqueeze(0) if is_scalar else t
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, dtype=t.dtype, device=t.device) / half_dim
        )
        
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embed = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        
        return embed.squeeze(0) if is_scalar else embed


class ConvBlock(nn.Module):
    """Conv → GroupNorm → SiLU."""
    def __init__(self, in_ch: int, out_ch: int, groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(max(1, min(groups, out_ch)), out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor = None) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        
        # Add time embedding if provided (broadcast over spatial dims)
        if time_embed is not None:
            x = x + time_embed.view(x.size(0), -1, 1, 1)
        
        x = self.act(x)
        return x


class DownBlock(nn.Module):
    """Downsample: Conv → stride=2."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor = None) -> torch.Tensor:
        x = self.block(x, time_embed)
        x = self.down(x)
        return x


class UpBlock(nn.Module):
    """Upsample: TransposeConv → stride=2."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.block = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None, time_embed: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        x = self.block(x, time_embed)
        return x


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
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        ch = base_channels
        channels = [ch]
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        for _ in range(3):
            out_ch = min(ch * 2, max_channels)
            self.down_blocks.append(DownBlock(ch, out_ch))
            channels.append(out_ch)
            ch = out_ch
        
        # Bottleneck
        self.middle = nn.Sequential(
            ConvBlock(ch, ch),
            ConvBlock(ch, ch),
        )
        
        # Decoder (upsampling) with skip connections
        self.up_blocks = nn.ModuleList()
        for i in range(3, 0, -1):
            in_ch = channels[i]
            skip_ch = channels[i - 1]
            out_ch = channels[i - 1]
            
            # Upsample first, then concatenate with skip (which doubles channel count)
            self.up_blocks.append(UpBlock(in_ch, out_ch))
            # But the ConvBlock inside UpBlock needs to handle concatenation
            # So we'll need to adjust: after concat, we have out_ch + skip_ch channels
        
        # Output projection
        self.output_proj = nn.Conv2d(base_channels, out_channels, kernel_size=1)

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
        
        # Concatenate input: [xt, y, M]
        h = torch.cat([x, y, M], dim=1)
        h = self.input_proj(h)
        
        # Encoder with skip connections
        skips = [h]
        for down_block in self.down_blocks:
            h = down_block(h, time_embed)
            skips.append(h)
        
        # Bottleneck
        h = self.middle(h)
        
        # Decoder
        for i, (up_block, skip) in enumerate(zip(self.up_blocks, reversed(skips[:-1]))):
            h = up_block(h, skip, time_embed)
        
        # Output
        out = self.output_proj(h)
        
        return out
