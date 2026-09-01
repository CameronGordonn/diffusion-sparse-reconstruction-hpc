"""VAE training script for 6-hour weather forecasting.

Loss function:
  L_total = L_recon + β * L_KL
  L_recon = MSE(x_recon, x_target)
  L_KL = KL(N(μ,σ²) || N(0,I)) = 0.5 * sum(-log(σ²) - 1 + μ² + σ²)

Hyperparameters (configurable via CLI; see CLAUDE.md for justifications):
  - latent_dim = 64 (selected for stability)
  - beta = 1.0 (no KL annealing, standard VAE)
  - batch_size = 32 (fits typical GPU memory)
  - lr = 1e-3 (Adam default for image models)
  - epochs = 50 (with early stopping on validation loss)

Data:
  - Train: 1979-1981 concatenated, normalized
  - Validation: 10% temporal tail of train
  - Test: 1983, 1985 held separately (evaluated after training)

Checkpointing:
  - Saves best model (lowest val loss) to models/vae_best.pt
  - Saves optimizer state for resuming
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def load_vae_class():
    module_path = Path(__file__).with_name("02_vae_model.py")
    spec = importlib.util.spec_from_file_location("am160_vae_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load VAE module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VAE


VAE = load_vae_class()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence: KL(N(μ,σ²) || N(0,I))."""
    return -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1).mean()


def reconstruction_loss(x_recon: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """L2 reconstruction error."""
    return nn.MSELoss()(x_recon, x_true)


def vae_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Total VAE loss and components."""
    l_recon = reconstruction_loss(x_recon, x_true)
    l_kl = kl_divergence(mu, logvar)
    return beta * l_kl + l_recon, l_recon, l_kl


def train_epoch(
    model: Any,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    beta: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float]:
    """Train one epoch; returns (total_loss, recon_loss, kl_loss)."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for x_t, x_tp1 in train_loader:
        x_t = x_t.to(device)
        x_tp1 = x_tp1.to(device)

        optimizer.zero_grad()

        # Forward: condition on x_t, predict x_t+1
        x_recon, mu, logvar = model(x_t)

        # Loss
        loss, recon, kl = vae_loss(x_recon, x_tp1, mu, logvar, beta=beta)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def validate(
    model: Any,
    val_loader: DataLoader,
    beta: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float]:
    """Validate; returns (total_loss, recon_loss, kl_loss)."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    with torch.no_grad():
        for x_t, x_tp1 in val_loader:
            x_t = x_t.to(device)
            x_tp1 = x_tp1.to(device)

            x_recon, mu, logvar = model(x_t)
            loss, recon, kl = vae_loss(x_recon, x_tp1, mu, logvar, beta=beta)

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_dir = Path(args.data_dir) / "results" / "processed"
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data dir not found: {data_dir}")

    train_data = np.load(data_dir / "train_pairs.npz")
    x_train_t = torch.from_numpy(train_data["x_t"]).float()
    x_train_tp1 = torch.from_numpy(train_data["x_tp1"]).float()

    # Split train/val
    n_train = x_train_t.shape[0]
    n_val = int(0.1 * n_train)
    n_actual_train = n_train - n_val

    x_train = x_train_t[:n_actual_train]
    x_train_target = x_train_tp1[:n_actual_train]

    x_val = x_train_t[n_actual_train:]
    x_val_target = x_train_tp1[n_actual_train:]

    train_dataset = TensorDataset(x_train, x_train_target)
    val_dataset = TensorDataset(x_val, x_val_target)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {n_actual_train}, Val samples: {n_val}")

    # Model setup
    model = VAE(in_channels=2, latent_dim=args.latent_dim, h=91, w=180)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    models_dir = Path(args.out_root) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10

    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
    }

    for epoch in range(args.epochs):
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, beta=args.beta, device=device
        )
        val_loss, val_recon, val_kl = validate(model, val_loader, beta=args.beta, device=device)

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} (R: {train_recon:.4f}, KL: {train_kl:.4f}) | "
                f"Val Loss: {val_loss:.4f} (R: {val_recon:.4f}, KL: {val_kl:.4f})"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "hyperparams": {
                        "latent_dim": args.latent_dim,
                        "beta": args.beta,
                        "lr": args.lr,
                        "batch_size": args.batch_size,
                    },
                },
                models_dir / "vae_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch} (patience exhausted)")
                break

    # Save history
    with (models_dir / "vae_training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best checkpoint: models/vae_best.pt")
    print(f"Final validation loss: {best_val_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE for weather forecasting")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Project root with data/")
    parser.add_argument("--out-root", type=Path, default=Path("."), help="Output root for models/")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight in VAE loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
