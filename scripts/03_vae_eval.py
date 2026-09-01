"""VAE evaluation: one-step and autoregressive forecasts.

Metrics:
1. One-step forecasts: ensemble mean + std, RMSE per field
2. Autoregressive rollout: iterative predictions at 24h, 48h, 72h lead times
3. Uncertainty analysis: ensemble variance as error bars
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_vae_class():
    module_path = Path(__file__).with_name("02_vae_model.py")
    spec = importlib.util.spec_from_file_location("am160_vae_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load VAE module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VAE


VAE = load_vae_class()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[Any, Dict]:
    """Load trained VAE model from checkpoint."""
    data = torch.load(checkpoint_path, map_location=device)
    
    latent_dim = data["hyperparams"]["latent_dim"]
    model = VAE(in_channels=2, latent_dim=latent_dim, h=91, w=180)
    model.load_state_dict(data["model_state"])
    model = model.to(device)
    model.eval()
    
    return model, data["hyperparams"]


def load_test_data(test_year: int, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load test pairs for given year."""
    npz_path = data_dir / f"test_pairs_{test_year}.npz"
    data = np.load(npz_path)
    return data["x_t"], data["x_tp1"]


def load_normalization(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load normalization stats."""
    npz = np.load(data_dir / "normalization_stats.npz")
    return npz["mean"], npz["std"]


def denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert normalized tensor to physical units."""
    return x * std + mean


def ensure_finite(name: str, arr: np.ndarray) -> None:
    """Fail fast on NaN/Inf metrics or model outputs."""
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values detected in {name}")


def one_step_forecast(
    model: Any,
    x_t: torch.Tensor,
    n_samples: int = 50,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ensemble one-step forecast using Welford online mean/variance.
    Processes in mini-batches to avoid OOM with large test sets.

    Returns:
        forecast_mean: (N, C, H, W)
        forecast_std:  (N, C, H, W)
    """
    N = x_t.shape[0]
    # Welford accumulators: count, mean, M2 (all shape N,C,H,W on CPU)
    count = 0
    mean_acc  = np.zeros((N, *x_t.shape[1:]), dtype=np.float64)
    M2_acc    = np.zeros_like(mean_acc)

    with torch.no_grad():
        for _ in range(n_samples):
            # Process full dataset in mini-batches to limit GPU/RAM pressure
            preds = []
            for start in range(0, N, batch_size):
                xb = x_t[start : start + batch_size]
                mu, logvar = model.encode(xb)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                preds.append(model.decode(z).cpu().numpy())
            pred_full = np.concatenate(preds, axis=0)  # (N, C, H, W)

            # Welford update
            count += 1
            delta = pred_full - mean_acc
            mean_acc += delta / count
            delta2 = pred_full - mean_acc
            M2_acc += delta * delta2

    forecast_std = np.sqrt(M2_acc / max(count - 1, 1)).astype(np.float32)
    forecast_mean = mean_acc.astype(np.float32)
    ensure_finite("one_step forecast mean", forecast_mean)
    ensure_finite("one_step forecast std", forecast_std)
    return forecast_mean, forecast_std


def compute_autoregressive_rmse(
    model: Any,
    x_t_full: torch.Tensor,
    x_tp1_full: np.ndarray,
    n_steps: int = 40,
    n_starts: int = 20,
    deterministic: bool = True,
    stochastic_members: int = 10,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[float]]:
    """
    Autoregressive rollout RMSE over multiple start states.
    
    Args:
        x_t_full: Full sequence of test inputs with shape (N, C, H, W)
        x_tp1_full: Full sequence of one-step targets with shape (N, C, H, W)
        n_steps: Number of 6-hour steps to forecast
        n_starts: Number of rollout starts to evaluate
        deterministic: If True, use encoder mean; if False, use ensemble mean each step
        stochastic_members: Number of ensemble samples used per step in stochastic mode
    
    Returns:
        Dict with lead hours and RMSE stats by lead.
    """
    n_total = x_t_full.shape[0]
    max_start = n_total - n_steps
    if max_start <= 0:
        raise ValueError(
            f"Not enough test samples for n_steps={n_steps}. "
            f"Available inputs={n_total}."
        )

    # Spread starts across the available period for robust lead-time skill estimates.
    starts = np.linspace(0, max_start - 1, num=min(n_starts, max_start), dtype=int)
    rmse_matrix = np.zeros((len(starts), n_steps), dtype=np.float64)
    rmse_matrix_phys: Optional[np.ndarray] = None
    if mean is not None and std is not None:
        rmse_matrix_phys = np.zeros((len(starts), n_steps), dtype=np.float64)

    for i_start, s in enumerate(starts):
        current_state = x_t_full[s : s + 1]

        for lead_idx in range(n_steps):
            with torch.no_grad():
                if deterministic:
                    mu, _ = model.encode(current_state)
                    next_state = model.decode(mu)
                else:
                    samples = []
                    for _ in range(stochastic_members):
                        mu, logvar = model.encode(current_state)
                        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                        samples.append(model.decode(z).cpu().numpy())
                    next_state_np = np.mean(samples, axis=0)
                    next_state = torch.from_numpy(next_state_np).float().to(device)

            pred_np = next_state.cpu().numpy()
            truth_np = x_tp1_full[s + lead_idx : s + lead_idx + 1]
            rmse_matrix[i_start, lead_idx] = compute_rmse(pred_np, truth_np)
            if rmse_matrix_phys is not None:
                pred_phys = denormalize(pred_np, mean, std)
                truth_phys = denormalize(truth_np, mean, std)
                rmse_matrix_phys[i_start, lead_idx] = compute_rmse(pred_phys, truth_phys)
            current_state = next_state

    ensure_finite("autoregressive rmse matrix", rmse_matrix)
    lead_times_hours = ((np.arange(n_steps) + 1) * 6).astype(int)
    out = {
        "lead_times_hours": lead_times_hours.tolist(),
        "rmse_by_lead_mean": rmse_matrix.mean(axis=0).tolist(),
        "rmse_by_lead_std": rmse_matrix.std(axis=0).tolist(),
        "n_starts": int(len(starts)),
    }
    if rmse_matrix_phys is not None:
        ensure_finite("autoregressive physical rmse matrix", rmse_matrix_phys)
        out["rmse_by_lead_mean_physical"] = rmse_matrix_phys.mean(axis=0).tolist()
        out["rmse_by_lead_std_physical"] = rmse_matrix_phys.std(axis=0).tolist()
    return out


def compute_rmse(pred: np.ndarray, truth: np.ndarray, axis: Optional[Tuple[int, ...]] = None) -> float:
    """MSE over space, then sqrt."""
    mse = np.mean((pred - truth) ** 2, axis=axis)
    return float(np.sqrt(np.mean(mse)))


def load_train_climatology(data_dir: Path, train_years=(1979, 1980, 1981)) -> np.ndarray:
    """Per-gridpoint mean over training years only, in normalized units."""
    total, count = None, 0
    for year in train_years:
        arr = np.load(data_dir / f'norm_{year}.npy', mmap_mode='r')
        block_sum = np.asarray(arr, dtype=np.float64).sum(axis=0)
        total = block_sum if total is None else total + block_sum
        count += arr.shape[0]
    return (total / count).astype(np.float32)


def one_step_baselines(x_t: np.ndarray, x_tp1: np.ndarray, climatology: np.ndarray) -> Dict[str, float]:
    """Trivial references for the 6 h forecast, in normalized units.

    Persistence (carry the input forward unchanged) is the bar a 6 h forecast
    must clear. Normalization is per-channel and global, so 'zero' is exactly
    the training mean field and scores ~1.0 by construction.
    """
    return {
        'persistence': compute_rmse(x_t, x_tp1),
        'gridpoint_climatology': compute_rmse(np.broadcast_to(climatology[np.newaxis], x_tp1.shape), x_tp1),
        'zero_global_mean': compute_rmse(np.zeros_like(x_tp1), x_tp1),
    }


def persistence_by_lead(
    data_dir: Path,
    year: int,
    lead_times_hours,
    climatology: np.ndarray,
    step_hours: int = 6,
) -> Dict[str, Any]:
    """Persistence RMSE at each rollout lead, plus the flat climatology line."""
    series = np.load(data_dir / f'norm_{year}.npy', mmap_mode='r')
    persistence = []
    for lead in lead_times_hours:
        k = int(round(lead / step_hours))
        if k < 1 or k >= series.shape[0]:
            persistence.append(float('nan'))
            continue
        diff = np.asarray(series[:-k], dtype=np.float32) - np.asarray(series[k:], dtype=np.float32)
        persistence.append(float(np.sqrt(np.mean(diff ** 2, dtype=np.float64))))
    diff = np.asarray(series, dtype=np.float32) - climatology[np.newaxis]
    clim = float(np.sqrt(np.mean(diff ** 2, dtype=np.float64)))
    return {'persistence': persistence, 'gridpoint_climatology': clim}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.out_root) / "models" / "vae_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model, hyperparams = load_checkpoint(checkpoint_path, device)
    print(f"Loaded trained VAE: latent_dim={hyperparams['latent_dim']}, beta={hyperparams['beta']}")
    
    # Load data and normalization
    data_dir = Path(args.data_dir) / "results" / "processed"
    mean, std = load_normalization(data_dir)
    climatology = load_train_climatology(data_dir)
    
    results = {}
    
    # Evaluate on both test years
    for test_year in [1983, 1985]:
        print(f"\n{'='*60}")
        print(f"Evaluating on {test_year}")
        print(f"{'='*60}")
        
        x_t, x_tp1 = load_test_data(test_year, data_dir)
        x_t = torch.from_numpy(x_t).float().to(device)
        x_tp1_np = x_tp1
        
        year_results = {}
        
        # One-step forecasts (50 ensemble members, mini-batch=64 to limit RAM)
        print("\nOne-step forecast...")
        x_recon_samples, x_recon_std = one_step_forecast(
            model, x_t, n_samples=50, device=device, batch_size=64
        )
        
        # Compute RMSE in normalized space
        rmse_overall_norm = compute_rmse(x_recon_samples, x_tp1_np)
        rmse_z500_norm = compute_rmse(x_recon_samples[:, 0], x_tp1_np[:, 0])
        rmse_z50_norm = compute_rmse(x_recon_samples[:, 1], x_tp1_np[:, 1])

        # Compute RMSE in physical units
        x_recon_phys = denormalize(x_recon_samples, mean, std)
        x_tp1_phys = denormalize(x_tp1_np, mean, std)
        rmse_overall_phys = compute_rmse(x_recon_phys, x_tp1_phys)
        rmse_z500_phys = compute_rmse(x_recon_phys[:, 0], x_tp1_phys[:, 0])
        rmse_z50_phys = compute_rmse(x_recon_phys[:, 1], x_tp1_phys[:, 1])
        
        year_results["one_step"] = {
            "rmse_normalized": {
                "overall": rmse_overall_norm,
                "z500": rmse_z500_norm,
                "z50": rmse_z50_norm,
            },
            "rmse_physical": {
                "overall": rmse_overall_phys,
                "z500": rmse_z500_phys,
                "z50": rmse_z50_phys,
            },
            # Backward-compatible keys map to normalized space.
            "rmse_overall": rmse_overall_norm,
            "rmse_z500": rmse_z500_norm,
            "rmse_z50": rmse_z50_norm,
        }

        print(f"  One-step RMSE (normalized): {rmse_overall_norm:.6f}")
        print(f"    Z500: {rmse_z500_norm:.6f}")
        print(f"    Z50: {rmse_z50_norm:.6f}")
        print(f"  One-step RMSE (physical): {rmse_overall_phys:.3f}")
        print(f"    Z500: {rmse_z500_phys:.3f}")
        print(f"    Z50: {rmse_z50_phys:.3f}")
        
        # Autoregressive rollout (deterministic for stability and fair lead-time skill)
        print("\nAutoregressive rollout (deterministic, 10 days = 40 steps)...")
        autoreg_norm = compute_autoregressive_rmse(
            model,
            x_t,
            x_tp1_np,
            n_steps=args.rollout_steps,
            n_starts=args.rollout_starts,
            deterministic=True,
            mean=mean,
            std=std,
            device=device,
        )
        lead_hours = autoreg_norm["lead_times_hours"]
        rmse_lead_phys = autoreg_norm.get("rmse_by_lead_mean_physical", [])

        year_results["autoregressive"] = {
            "lead_times_hours": lead_hours,
            "rmse_normalized": {
                "mean": autoreg_norm["rmse_by_lead_mean"],
                "std": autoreg_norm["rmse_by_lead_std"],
            },
            "rmse_physical": {
                "mean": rmse_lead_phys,
                "std": autoreg_norm.get("rmse_by_lead_std_physical", []),
            },
            # Backward-compatible key maps to normalized mean curve.
            "rmse_by_lead": autoreg_norm["rmse_by_lead_mean"],
            "n_starts": autoreg_norm["n_starts"],
        }

        # Trivial baselines: a forecast that loses to these has no skill.
        year_results['baselines'] = {
            'one_step': one_step_baselines(x_t.cpu().numpy(), x_tp1_np, climatology),
            'by_lead': persistence_by_lead(data_dir, test_year, lead_hours, climatology),
        }
        print(f"  Persistence one-step RMSE (normalized): {year_results['baselines']['one_step']['persistence']:.6f}")
        print(f"  Climatology RMSE (normalized):          {year_results['baselines']['one_step']['gridpoint_climatology']:.6f}")

        # Lead-time figure
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        lead_np = np.array(lead_hours)
        rmse_mean = np.array(autoreg_norm["rmse_by_lead_mean"])
        rmse_std = np.array(autoreg_norm["rmse_by_lead_std"])
        ax.plot(lead_np, rmse_mean, label="VAE", linewidth=2, color="C0")
        ax.fill_between(lead_np, rmse_mean - rmse_std, rmse_mean + rmse_std, alpha=0.2, color="C0")

        # Reference curves. A forecast above these lines has no skill.
        base_lead = year_results["baselines"]["by_lead"]
        ax.plot(lead_np, base_lead["persistence"], linestyle="--", linewidth=1.8,
                color="C1", label="Persistence")
        ax.axhline(base_lead["gridpoint_climatology"], linestyle=":", linewidth=1.8,
                   color="0.35", label="Climatology (per-gridpoint)")
        ax.axhline(1.0, linestyle="-.", linewidth=1.2, color="0.65",
                   label="Global training mean")

        ax.set_xlabel("Lead Time (hours)")
        ax.set_ylabel("RMSE (normalized)")
        ax.set_title(f"VAE Autoregressive Skill vs Lead Time ({test_year})")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="center right", fontsize=9)
        fig.savefig(str((Path(args.out_root) / "plots" / "phase1_vae" / f"phase1_rollout_{test_year}.png")), dpi=150)
        plt.close(fig)
        
        # Save sample visualization
        plots_dir = Path(args.out_root) / "plots" / "phase1_vae"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # One-step snapshot
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        
        idx = 0  # First sample
        ax = axes[0, 0]
        im = ax.imshow(x_tp1_np[idx, 0], cmap="viridis", origin="lower")
        ax.set_title("Ground Truth Z500")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax = axes[0, 1]
        im = ax.imshow(x_recon_samples[idx, 0], cmap="viridis", origin="lower")
        ax.set_title("Pred Mean Z500")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax = axes[0, 2]
        im = ax.imshow(x_recon_std[idx, 0], cmap="hot", origin="lower")
        ax.set_title("Pred Std Z500")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax = axes[1, 0]
        im = ax.imshow(x_tp1_np[idx, 1], cmap="plasma", origin="lower")
        ax.set_title("Ground Truth Z50")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax = axes[1, 1]
        im = ax.imshow(x_recon_samples[idx, 1], cmap="plasma", origin="lower")
        ax.set_title("Pred Mean Z50")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax = axes[1, 2]
        im = ax.imshow(x_recon_std[idx, 1], cmap="hot", origin="lower")
        ax.set_title("Pred Std Z50")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        fig.suptitle(f"VAE One-Step Forecast: {test_year}")
        fig.savefig(str(plots_dir / f"phase1_onestep_{test_year}.png"), dpi=150)
        plt.close(fig)
        
        results[test_year] = year_results
    
    # Save results
    results_dir = Path(args.out_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with (results_dir / "vae_evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete. Results saved to results/vae_evaluation.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VAE forecasts")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Project root")
    parser.add_argument("--out-root", type=Path, default=Path("."), help="Output root")
    parser.add_argument("--rollout-steps", type=int, default=40, help="Autoregressive lead steps (6h each)")
    parser.add_argument("--rollout-starts", type=int, default=20, help="Number of rollout start states")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
