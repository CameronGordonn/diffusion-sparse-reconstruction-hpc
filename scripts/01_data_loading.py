"""Phase 0 data loading and preprocessing for AM160 final exam.

This script:
1. Loads yearly NetCDF files (1979, 1980, 1981, 1983, 1985).
2. Extracts geopotential height fields at 500 hPa and 50 hPa into shape (T, C=2, H, W).
3. Verifies temporal cadence and reports dataset statistics.
4. Computes train-only normalization stats (1979-1981) and applies them to all years.
5. Builds one-step forecast pairs (x_t, x_t+1) for train and test years.
6. Saves processed arrays, normalization stats, and summary artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


TRAIN_YEARS = (1979, 1980, 1981)
TEST_YEARS = (1983, 1985)
ALL_YEARS = TRAIN_YEARS + TEST_YEARS
TARGET_LEVELS_HPA = (500, 50)

LEVEL_CANDIDATES = ("level", "lev", "plev", "isobaricInhPa")
TIME_CANDIDATES = ("time", "valid_time")
LAT_CANDIDATES = ("lat", "latitude")
LON_CANDIDATES = ("lon", "longitude")


def find_coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"None of {tuple(candidates)} found in dataset coords/dims.")


def find_main_data_var(ds: xr.Dataset, time_dim: str) -> str:
    for name, var in ds.data_vars.items():
        if time_dim in var.dims:
            return name
    raise ValueError("Could not find a data variable that includes the time dimension.")


def extract_two_level_tensor(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    time_dim = find_coord_name(ds, TIME_CANDIDATES)
    var_name = find_main_data_var(ds, time_dim)
    var = ds[var_name]

    lat_dim = find_coord_name(ds, LAT_CANDIDATES)
    lon_dim = find_coord_name(ds, LON_CANDIDATES)

    if len(var.dims) < 3:
        raise ValueError(f"Expected at least 3 dimensions for variable {var_name}, got {var.dims}")

    level_dim = None
    for candidate in LEVEL_CANDIDATES:
        if candidate in var.dims:
            level_dim = candidate
            break

    if level_dim is None:
        raise ValueError(
            "Could not find a pressure level dimension for selecting 500 hPa and 50 hPa. "
            f"Available variable dims: {var.dims}"
        )

    # Select nearest requested levels to be robust across coordinate conventions.
    selected = var.sel({level_dim: list(TARGET_LEVELS_HPA)}, method="nearest")
    selected = selected.transpose(time_dim, level_dim, lat_dim, lon_dim)

    tensor = selected.to_numpy().astype(np.float32)
    times = ds[time_dim].to_numpy()

    coords = {
        "time_dim": time_dim,
        "level_dim": level_dim,
        "lat_dim": lat_dim,
        "lon_dim": lon_dim,
        "var_name": var_name,
    }
    return tensor, times, coords


def compute_time_step_hours(times: np.ndarray) -> float:
    if times.shape[0] < 2:
        return float("nan")
    dt = times[1] - times[0]
    dt_hours = dt / np.timedelta64(1, "h")
    return float(dt_hours)


def make_pairs(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.shape[0] < 2:
        raise ValueError("Need at least two timesteps to build (x_t, x_t+1) pairs.")
    return x[:-1], x[1:]


def save_spatial_snapshots(raw_train: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_times = raw_train.shape[0]

    # Four evenly spaced snapshots over the train period.
    indices = np.linspace(0, n_times - 1, num=4, dtype=int)

    for i, idx in enumerate(indices, start=1):
        z500 = raw_train[idx, 0]
        z50 = raw_train[idx, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        im0 = axes[0].imshow(z500, origin="lower", cmap="viridis")
        axes[0].set_title(f"Z500 Snapshot #{i} (t={idx})")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(z50, origin="lower", cmap="plasma")
        axes[1].set_title(f"Z50 Snapshot #{i} (t={idx})")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.suptitle("Phase 0 Spatial Inspection")
        fig.savefig(out_dir / f"phase0_snapshot_{i}.png", dpi=150)
        plt.close(fig)


def run(data_dir: Path, out_root: Path) -> None:
    processed_dir = out_root / "results" / "processed"
    plots_dir = out_root / "plots" / "phase0"
    models_dir = out_root / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    raw_by_year: Dict[int, np.ndarray] = {}
    norm_by_year: Dict[int, np.ndarray] = {}
    times_by_year: Dict[int, np.ndarray] = {}
    coord_info_by_year: Dict[int, Dict[str, str]] = {}

    summary_rows = []

    for year in ALL_YEARS:
        file_path = data_dir / f"z{year}.nc"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing expected file: {file_path}")

        with xr.open_dataset(file_path) as ds:
            tensor, times, coords = extract_two_level_tensor(ds)

        raw_by_year[year] = tensor
        times_by_year[year] = times
        coord_info_by_year[year] = coords

        summary_rows.append(
            {
                "year": year,
                "shape": list(tensor.shape),
                "time_step_hours": compute_time_step_hours(times),
                "min": float(np.nanmin(tensor)),
                "max": float(np.nanmax(tensor)),
                "mean": float(np.nanmean(tensor)),
                "std": float(np.nanstd(tensor)),
                "nan_count": int(np.isnan(tensor).sum()),
                "var_name": coords["var_name"],
                "level_dim": coords["level_dim"],
            }
        )

    train_raw = np.concatenate([raw_by_year[y] for y in TRAIN_YEARS], axis=0)

    # Per-channel normalization using train years only.
    train_mean = train_raw.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    train_std = train_raw.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    train_std = np.clip(train_std, 1e-6, None)

    for year in ALL_YEARS:
        norm = (raw_by_year[year] - train_mean) / train_std
        if not np.isfinite(norm).all():
            raise ValueError(f"Non-finite values found after normalization for year {year}.")
        norm_by_year[year] = norm.astype(np.float32)

    # Build pairs for forecasting.
    train_norm = np.concatenate([norm_by_year[y] for y in TRAIN_YEARS], axis=0)
    x_train_t, x_train_tp1 = make_pairs(train_norm)
    x_test_1983_t, x_test_1983_tp1 = make_pairs(norm_by_year[1983])
    x_test_1985_t, x_test_1985_tp1 = make_pairs(norm_by_year[1985])

    np.savez_compressed(
        processed_dir / "normalization_stats.npz",
        mean=train_mean,
        std=train_std,
        train_years=np.array(TRAIN_YEARS),
        test_years=np.array(TEST_YEARS),
    )

    np.savez_compressed(
        processed_dir / "train_pairs.npz",
        x_t=x_train_t,
        x_tp1=x_train_tp1,
    )

    np.savez_compressed(
        processed_dir / "test_pairs_1983.npz",
        x_t=x_test_1983_t,
        x_tp1=x_test_1983_tp1,
    )

    np.savez_compressed(
        processed_dir / "test_pairs_1985.npz",
        x_t=x_test_1985_t,
        x_tp1=x_test_1985_tp1,
    )

    for year in ALL_YEARS:
        np.save(processed_dir / f"raw_{year}.npy", raw_by_year[year])
        np.save(processed_dir / f"norm_{year}.npy", norm_by_year[year])

    summary_payload = {
        "summary": summary_rows,
        "coord_info_by_year": {str(k): v for k, v in coord_info_by_year.items()},
        "train_concat_shape": list(train_raw.shape),
        "train_pair_shape": {
            "x_t": list(x_train_t.shape),
            "x_tp1": list(x_train_tp1.shape),
        },
        "normalization": {
            "mean_shape": list(train_mean.shape),
            "std_shape": list(train_std.shape),
            "mean_values": train_mean.flatten().tolist(),
            "std_values": train_std.flatten().tolist(),
        },
    }

    with (processed_dir / "phase0_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    save_spatial_snapshots(train_raw, plots_dir)

    print("Phase 0 complete.")
    print(f"Saved processed arrays to: {processed_dir}")
    print(f"Saved plots to: {plots_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AM160 Phase 0 data loading/preprocessing")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing zYYYY.nc files",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("."),
        help="Project root where results/, plots/, models/ will be written",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir.resolve(), args.out_root.resolve())
