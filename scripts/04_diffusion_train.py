"""Diffusion model training for sparse field reconstruction.

Diffusion process:
  - Forward: x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε, where ε ~ N(0,I)
  - Reverse: predict noise ε(x_t, y, M, t) via U-Net
  - Schedule: cosine schedule (selected in plan)
  - Timesteps: 300 steps (selected in plan)

Training:
  - For each sparsity level (30%, 60%, 90%), train a separate model
  - Create masks on-the-fly or use fixed-seed random masks for reproducibility
  - Loss: MSE(ε_pred, ε_true)

Sampling (inference):
  - Start from x_T ~ N(0,I)
  - Iteratively denoise: x_{t-1} = μ(x_t) + σ_t z, z ~ N(0,I)
  - Keep observed regions unconstrained in normalized space (allow optional safety clipping)
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def load_diffusion_class(model_type: str):
    model_files = {
        "minimal": "04_diffusion_model_minimal.py",
        "v2": "04_diffusion_model_v2.py",
    }
    if model_type not in model_files:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose from {tuple(model_files.keys())}")

    module_path = Path(__file__).with_name(model_files[model_type])
    spec = importlib.util.spec_from_file_location("am160_diffusion_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load diffusion module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DiffusionUNet


class CosineNoiseSchedule:
    """Cosine noise schedule for diffusion."""
    def __init__(self, num_steps: int = 300, s: float = 0.008):
        self.num_steps = num_steps
        self.s = s
        
        # Compute alphas
        steps = torch.arange(num_steps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(
            (steps / num_steps + s) / (1 + s) * math.pi * 0.5
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        self.alphas_cumprod = alphas_cumprod  # α̅_t
        self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(self.betas, 0.0001, 0.9999)
        
        self.alphas = 1 - self.betas
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[:-1])
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[:-1])

    def to(self, device: torch.device) -> "CosineNoiseSchedule":
        """Move schedule tensors to the target device."""
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε
        
        Returns:
            x_t: noisy sample
            noise: the added noise ε
        """
        noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        if sqrt_alpha_prod.dim() == 1:
            sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        return x_t, noise

    def sample_timestep(self, batch_size: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Sample random timestep indices."""
        return torch.randint(0, self.num_steps, (batch_size,), device=device)


def generate_mask(shape: Tuple[int, ...], sparsity: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate binary mask for sparse observations.
    
    Args:
        shape: (H, W)
        sparsity: fraction missing (e.g., 0.3 for 30% missing)
        seed: random seed for reproducibility
    
    Returns:
        mask: (H, W) with 1 where observed, 0 where missing
    """
    if seed is not None:
        np.random.seed(seed)
    
    mask = np.random.binomial(1, 1 - sparsity, size=shape)
    return mask


class DiffusionTrainer:
    def __init__(
        self,
        model: Any,
        schedule: CosineNoiseSchedule,
        device: torch.device = torch.device("cpu"),
        use_amp: bool = False,
    ):
        self.model = model
        self.schedule = schedule
        self.device = device
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_step(
        self,
        optimizer: optim.Optimizer,
        x_0: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """
        Single training step.
        
        Args:
            x_0: clean sample (B, 2, H, W)
            mask: observation mask (B, 1, H, W) - 1 where observed, 0 where missing
        
        Returns:
            loss value
        """
        B = x_0.shape[0]
        
        # Create sparse observations y = M ⊙ x_0
        y = x_0 * mask
        
        # Sample random timesteps
        t = self.schedule.sample_timestep(B, device=self.device)
        
        # Forward diffusion
        x_t, noise_true = self.schedule.add_noise(x_0, t)
        
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Predict noise
            noise_pred = self.model(x_t, t, y, mask)
            # Loss: MSE of predicted vs true noise
            loss = nn.MSELoss()(noise_pred, noise_true)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        shape: Tuple[int, int, int],
        y: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 300,
        clamp_observed: bool = True,
    ) -> torch.Tensor:
        """
        Reverse diffusion sampling.
        
        Args:
            shape: (C, H, W)
            y: sparse observations (B, C, H, W)
            mask: observation mask (B, 1, H, W)
            num_steps: number of denoising steps
        
        Returns:
            x_0: reconstructed sample
        """
        self.model.eval()
        
        # Start from noise
        x = torch.randn(batch_size, *shape, device=self.device)
        
        # Reverse diffusion
        for step in range(num_steps - 1, -1, -1):
            t = torch.full((batch_size,), step, dtype=torch.long, device=self.device)
            
            # Predict noise at this step
            noise_pred = self.model(x, t, y, mask)
            
            # Denoise
            alpha = self.schedule.alphas[step]
            alpha_cumprod = self.schedule.alphas_cumprod[step]
            alpha_cumprod_prev = (
                self.schedule.alphas_cumprod[step - 1]
                if step > 0
                else torch.tensor(1.0, device=x.device)
            )

            # Numerical guards prevent inf/NaN when values approach zero.
            alpha = torch.clamp(alpha, min=1e-8, max=1.0)
            alpha_cumprod = torch.clamp(alpha_cumprod, min=1e-8, max=1.0)
            alpha_cumprod_prev = torch.clamp(alpha_cumprod_prev, min=1e-8, max=1.0)
            
            # Compute posterior mean
            coeff1 = 1.0 / torch.sqrt(alpha)
            coeff2 = (1 - alpha) / torch.sqrt(torch.clamp(1 - alpha_cumprod, min=1e-8))
            
            mean = coeff1 * (x - coeff2 * noise_pred)
            
            # Posterior variance
            if step > 0:
                var = ((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * (1 - alpha)
                var = torch.clamp(var, min=0.0)
                sigma = torch.sqrt(var)
                z = torch.randn_like(x)
                x = mean + sigma * z
            else:
                x = mean

            # Keep observed pixels tied to the sparse observations at every reverse step.
            if clamp_observed:
                x = mask * y + (1 - mask) * x

        # Replace any lingering NaN/Inf with finite values for robust downstream metrics.
        return torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training diffusion model for {args.sparsity * 100:.0f}% sparsity")
    print(f"Model type: {args.model_type}")
    
    # Load data
    data_dir = Path(args.data_dir) / "results" / "processed"
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data dir not found: {data_dir}")
    
    train_data = np.load(data_dir / "train_pairs.npz")
    x_train_t = torch.from_numpy(train_data["x_t"]).float()
    
    # Create dataset
    dataset = TensorDataset(x_train_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Model and schedule
    DiffusionUNet = load_diffusion_class(args.model_type)
    schedule = CosineNoiseSchedule(num_steps=args.num_steps).to(device)
    model = DiffusionUNet(in_channels=5, out_channels=2, base_channels=args.base_channels)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = DiffusionTrainer(model, schedule, device=device, use_amp=args.use_amp)
    
    # Mask seed for reproducibility (fixed per sparsity level)
    mask_seed = int(args.sparsity * 1000)
    
    # Training loop
    models_dir = Path(args.out_root) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = f"diffusion_{int(args.sparsity * 100)}pct"
    if args.model_type != "minimal":
        checkpoint_name = f"{checkpoint_name}_{args.model_type}"
    best_loss = float("inf")
    patience = args.patience
    patience_counter = 0
    
    history = {"train_loss": []}
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for (x_0,) in loader:
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            
            # Generate random mask for this batch
            mask_spatial = generate_mask(
                (x_0.shape[2], x_0.shape[3]),
                args.sparsity,
                seed=mask_seed + epoch + n_batches,
            )
            mask = torch.from_numpy(mask_spatial[np.newaxis, np.newaxis, :, :]).float().to(device)
            mask = mask.expand(batch_size, -1, -1, -1)
            
            loss = trainer.train_step(optimizer, x_0, mask)
            epoch_loss += loss
            n_batches += 1
        
        epoch_loss /= n_batches
        history["train_loss"].append(epoch_loss)
        
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss:.6f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "hyperparams": {
                        "sparsity": args.sparsity,
                        "num_steps": args.num_steps,
                        "lr": args.lr,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "patience": args.patience,
                        "clamp_observed": args.clamp_observed,
                        "model_type": args.model_type,
                        "base_channels": args.base_channels,
                        "use_amp": args.use_amp,
                    },
                },
                models_dir / f"{checkpoint_name}_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Save history
    with (models_dir / f"{checkpoint_name}_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    
    print(f"Training complete. Best checkpoint: models/{checkpoint_name}_best.pt")
    print(f"Final loss: {best_loss:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion model for sparse reconstruction")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Project root")
    parser.add_argument("--out-root", type=Path, default=Path("."), help="Output root")
    parser.add_argument("--model-type", type=str, default="minimal", choices=["minimal", "v2"], help="Diffusion architecture variant")
    parser.add_argument("--base-channels", type=int, default=64, help="Model width (higher can improve quality at higher cost)")
    parser.add_argument("--sparsity", type=float, default=0.3, help="Fraction missing (0.3, 0.6, 0.9)")
    parser.add_argument("--num-steps", type=int, default=300, help="Number of diffusion steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Enable automatic mixed precision on CUDA")
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false", help="Disable automatic mixed precision")
    parser.add_argument(
        "--clamp-observed",
        action="store_true",
        default=True,
        help="Clamp observed grid cells to sparse input during reverse diffusion",
    )
    parser.add_argument(
        "--no-clamp-observed",
        dest="clamp_observed",
        action="store_false",
        help="Disable observed-cell clamping during reverse diffusion",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
