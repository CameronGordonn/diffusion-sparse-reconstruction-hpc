"""Evaluate trained diffusion models on sparse reconstruction.

Produces:
- RMSE on unobserved and observed regions
- Qualitative plots: truth, sparse input, reconstruction, residual
- Uncertainty maps from repeated sampling
- Cross-sparsity results table for 30%, 60%, 90% missing
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


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


def load_diffusion_training_helpers():
    module_path = Path(__file__).with_name("04_diffusion_train.py")
    spec = importlib.util.spec_from_file_location("am160_diffusion_train", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load diffusion training helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.CosineNoiseSchedule,
        module.DiffusionTrainer,
        module.generate_mask,
        module.generate_structured_mask,
    )


CosineNoiseSchedule, DiffusionTrainer, generate_mask, generate_structured_mask = (
    load_diffusion_training_helpers()
)


def load_normalization(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load train-derived normalization stats for physical-unit metrics."""
    npz = np.load(data_dir / "normalization_stats.npz")
    return npz["mean"], npz["std"]


def denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert normalized tensor to physical units."""
    return x * std + mean


def ensure_finite(name: str, arr: np.ndarray) -> None:
    """Fail fast on invalid outputs/metrics."""
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values detected in {name}")


def load_model(checkpoint_path: Path, model_type: str, device: torch.device) -> Tuple[Any, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hyperparams = checkpoint["hyperparams"]
    model_type_ckpt = hyperparams.get("model_type", model_type)
    DiffusionUNet = load_diffusion_class(model_type_ckpt)
    base_channels = int(hyperparams.get("base_channels", 64))
    model = DiffusionUNet(in_channels=5, out_channels=2, base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, hyperparams


def masked_rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, observed: bool) -> float:
    region = mask == 1 if observed else mask == 0
    region = np.broadcast_to(region, pred.shape)
    diff = pred[region] - truth[region]
    finite = np.isfinite(diff)
    if not np.any(finite):
        return float("nan")
    diff = diff[finite]
    return float(np.sqrt(np.mean(diff ** 2)))


def sample_uncertainty(
    trainer: Any,
    y: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int,
    num_steps: int,
    clamp_observed: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    samples: List[np.ndarray] = []
    shape = tuple(y.shape[1:])
    for _ in range(num_samples):
        x = trainer.sample(
            y.shape[0],
            shape,
            y,
            mask,
            num_steps=num_steps,
            clamp_observed=clamp_observed,
        )
        samples.append(x.cpu().numpy())
    stacked = np.stack(samples, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def save_example_plot(
    out_path: Path,
    truth: np.ndarray,
    sparse: np.ndarray,
    recon: np.ndarray,
    uncertainty: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    titles = [
        "Truth Z500",
        "Sparse Z500",
        "Recon Z500",
        "Std Z500",
        "Truth Z50",
        "Sparse Z50",
        "Recon Z50",
        "Std Z50",
    ]
    fields = [
        truth[0], sparse[0], recon[0], uncertainty[0],
        truth[1], sparse[1], recon[1], uncertainty[1],
    ]
    cmaps = ["viridis", "viridis", "viridis", "hot", "plasma", "plasma", "plasma", "hot"]

    for ax, title, field, cmap in zip(axes.flat, titles, fields, cmaps):
        im = ax.imshow(field, origin="lower", cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def load_train_climatology(data_dir: Path, train_years=(1979, 1980, 1981)) -> np.ndarray:
    """Per-gridpoint mean over training years only, in normalized units."""
    total, count = None, 0
    for year in train_years:
        arr = np.load(data_dir / f'norm_{year}.npy', mmap_mode='r')
        block_sum = np.asarray(arr, dtype=np.float64).sum(axis=0)
        total = block_sum if total is None else total + block_sum
        count += arr.shape[0]
    return (total / count).astype(np.float32)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _separable_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur over (..., lat, lon): longitude wraps, latitude edge-pads."""
    kernel = _gaussian_kernel(sigma)
    radius = len(kernel) // 2
    padded = np.concatenate([field[..., -radius:], field, field[..., :radius]], axis=-1)
    out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode='valid'), -1, padded)
    padded = np.concatenate(
        [np.repeat(out[..., :1, :], radius, axis=-2), out, np.repeat(out[..., -1:, :], radius, axis=-2)],
        axis=-2,
    )
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode='valid'), -2, padded)


def gaussian_inpaint(field: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized convolution: Gaussian-weighted mean of observed neighbours."""
    num = _separable_blur(field * mask, sigma)
    den = _separable_blur(np.broadcast_to(mask, field.shape).astype(np.float64), sigma)
    return num / np.maximum(den, 1e-8)


def baseline_metrics(
    truth: np.ndarray,
    mask: np.ndarray,
    climatology: np.ndarray,
    sigmas=(1.0, 2.0, 3.0, 5.0),
) -> Dict[str, float]:
    """Trivial reference scores on the unobserved region.

    Any learned reconstruction must beat these to be worth reporting. The
    Gaussian-inpaint entry takes the best sigma on the evaluation set itself,
    which favours the baseline -- deliberately, so it is a hard bar.
    """
    zero_fill = masked_rmse(np.zeros_like(truth), truth, mask, observed=False)
    clim_pred = np.broadcast_to(climatology[np.newaxis], truth.shape)
    clim = masked_rmse(clim_pred, truth, mask, observed=False)
    inpaint = min(
        masked_rmse(gaussian_inpaint(truth, mask, sigma), truth, mask, observed=False)
        for sigma in sigmas
    )
    return {
        'zero_fill_global_mean': zero_fill,
        'gridpoint_climatology': clim,
        'gaussian_inpaint': inpaint,
    }

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path(args.data_dir) / "results" / "processed"
    results_dir = Path(args.out_root) / "results"
    plots_dir = Path(args.out_root) / "plots" / "phase2_diffusion"
    models_dir = Path(args.out_root) / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    evaluations: Dict[str, Dict] = {}
    mean, std = load_normalization(processed_dir)
    climatology = load_train_climatology(processed_dir)
    test_years = [1983, 1985]
    sparsities = [0.3, 0.6, 0.9]

    for sparsity in sparsities:
        checkpoint_stem = f"diffusion_{int(sparsity * 100)}pct"
        if args.mask_type != "iid":
            checkpoint_stem = f"{checkpoint_stem}_{args.mask_type}{args.mask_block}"
        if args.model_type != "minimal":
            checkpoint_stem = f"{checkpoint_stem}_{args.model_type}"
        checkpoint = models_dir / f"{checkpoint_stem}_best.pt"
        if not checkpoint.exists():
            print(f"Skipping {sparsity:.1f}: missing checkpoint {checkpoint.name}")
            continue

        model, hyperparams = load_model(checkpoint, args.model_type, device)
        schedule = CosineNoiseSchedule(num_steps=hyperparams["num_steps"]).to(device)
        trainer = DiffusionTrainer(model, schedule, device=device)
        sparsity_key = f"{int(sparsity * 100)}pct"
        evaluations[sparsity_key] = {}

        for year in test_years:
            norm = np.load(processed_dir / f"norm_{year}.npy")
            n_cases = norm.shape[0] if args.num_cases <= 0 else min(args.num_cases, norm.shape[0])
            # Spread cases evenly across the year. Taking norm[:n_cases] would
            # evaluate only the first n_cases * 6 h -- a few days of January.
            case_idx = np.linspace(0, norm.shape[0] - 1, n_cases).astype(int)
            x0 = torch.from_numpy(norm[case_idx]).float().to(device)
            # Evaluate with the mask geometry the checkpoint was trained on.
            mask_seed = int(sparsity * 1000) + year
            if hyperparams.get("mask_type", "iid") == "block":
                mask_np = generate_structured_mask(
                    (x0.shape[2], x0.shape[3]),
                    sparsity,
                    block=int(hyperparams.get("mask_block", 8)),
                    seed=mask_seed,
                )
            else:
                mask_np = generate_mask((x0.shape[2], x0.shape[3]), sparsity, seed=mask_seed)
            mask = torch.from_numpy(mask_np[np.newaxis, np.newaxis, :, :]).float().to(device)
            mask = mask.expand(x0.shape[0], -1, -1, -1)
            y = x0 * mask

            recon_mean, recon_std = sample_uncertainty(
                trainer,
                y,
                mask,
                num_samples=args.num_samples,
                num_steps=hyperparams["num_steps"],
                clamp_observed=hyperparams.get("clamp_observed", True),
            )
            ensure_finite(f"diffusion recon mean {sparsity_key}-{year}", recon_mean)
            ensure_finite(f"diffusion recon std {sparsity_key}-{year}", recon_std)

            truth_np = x0.cpu().numpy()
            sparse_np = y.cpu().numpy()
            mask_broadcast = mask.cpu().numpy()
            recon_mean_phys = denormalize(recon_mean, mean, std)
            truth_phys = denormalize(truth_np, mean, std)

            metrics = {
                "num_cases": int(n_cases),
                "rmse_normalized": {
                    "unobserved": masked_rmse(recon_mean, truth_np, mask_broadcast, observed=False),
                    "observed": masked_rmse(recon_mean, truth_np, mask_broadcast, observed=True),
                },
                "rmse_physical": {
                    "unobserved": masked_rmse(recon_mean_phys, truth_phys, mask_broadcast, observed=False),
                    "observed": masked_rmse(recon_mean_phys, truth_phys, mask_broadcast, observed=True),
                },
                # Backward-compatible keys map to normalized metrics.
                "rmse_unobserved": masked_rmse(recon_mean, truth_np, mask_broadcast, observed=False),
                "rmse_observed": masked_rmse(recon_mean, truth_np, mask_broadcast, observed=True),
            }
            metrics['baselines'] = baseline_metrics(truth_np, mask_broadcast, climatology)
            evaluations[sparsity_key][str(year)] = metrics

            save_example_plot(
                plots_dir / f"diffusion_{sparsity_key}_{year}.png",
                truth_np[0],
                sparse_np[0],
                recon_mean[0],
                recon_std[0],
            )

        # Cross-year summary for each sparsity level.
        year_metrics = [evaluations[sparsity_key][str(y)] for y in test_years if str(y) in evaluations[sparsity_key]]
        if year_metrics:
            evaluations[sparsity_key]["summary"] = {
                "num_years": len(year_metrics),
                "rmse_normalized": {
                    "unobserved_mean": float(np.mean([m["rmse_normalized"]["unobserved"] for m in year_metrics])),
                    "observed_mean": float(np.mean([m["rmse_normalized"]["observed"] for m in year_metrics])),
                },
                "rmse_physical": {
                    "unobserved_mean": float(np.mean([m["rmse_physical"]["unobserved"] for m in year_metrics])),
                    "observed_mean": float(np.mean([m["rmse_physical"]["observed"] for m in year_metrics])),
                },
            }

    with (results_dir / "diffusion_evaluation.json").open("w", encoding="utf-8") as fh:
        json.dump(evaluations, fh, indent=2)

    print("Saved diffusion evaluation to results/diffusion_evaluation.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate diffusion reconstruction models")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--out-root", type=Path, default=Path("."))
    parser.add_argument("--model-type", type=str, default="minimal", choices=["minimal", "v2"], help="Diffusion architecture variant")
    parser.add_argument("--mask-type", type=str, default="iid", choices=["iid", "block"],
                        help="Which trained variant to evaluate; must match how it was trained")
    parser.add_argument("--mask-block", type=int, default=8, help="Block side length when --mask-type block")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--num-cases", type=int, default=20, help="Use <=0 to evaluate all available cases")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
