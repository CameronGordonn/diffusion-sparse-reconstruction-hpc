from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
DOCS_DIR = ROOT / "docs" / "guides"


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def fmt_sci(x: float) -> str:
    return f"{x:.4e}"


def check_required_artifacts() -> Tuple[bool, str]:
    required = [
        RESULTS_DIR / "vae_evaluation.json",
        RESULTS_DIR / "diffusion_evaluation.json",
        PLOTS_DIR / "phase1_vae" / "phase1_onestep_1983.png",
        PLOTS_DIR / "phase1_vae" / "phase1_onestep_1985.png",
        PLOTS_DIR / "phase1_vae" / "phase1_rollout_1983.png",
        PLOTS_DIR / "phase1_vae" / "phase1_rollout_1985.png",
        PLOTS_DIR / "phase2_diffusion" / "diffusion_30pct_1983.png",
        PLOTS_DIR / "phase2_diffusion" / "diffusion_30pct_1985.png",
        PLOTS_DIR / "phase2_diffusion" / "diffusion_60pct_1983.png",
        PLOTS_DIR / "phase2_diffusion" / "diffusion_60pct_1985.png",
        PLOTS_DIR / "phase2_diffusion" / "diffusion_90pct_1983.png",
        PLOTS_DIR / "phase2_diffusion" / "diffusion_90pct_1985.png",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        return False, "\n".join(missing)
    return True, ""


def main() -> None:
    ok, missing = check_required_artifacts()
    if not ok:
        print("Missing required artifacts:")
        print(missing)
        raise SystemExit(1)

    vae = load_json(RESULTS_DIR / "vae_evaluation.json")
    diff = load_json(RESULTS_DIR / "diffusion_evaluation.json")

    out_md = []
    out_md.append("# Final Numbers (Auto-generated)\n")

    out_md.append("## VAE One-Step RMSE\n")
    out_md.append("| Year | Overall (norm) | Z500 (norm) | Z50 (norm) | Overall (phys) | Z500 (phys) | Z50 (phys) |")
    out_md.append("|---|---:|---:|---:|---:|---:|---:|")
    for year in ["1983", "1985"]:
        one = vae[year]["one_step"]
        out_md.append(
            f"| {year} | {fmt(one['rmse_normalized']['overall'])} | {fmt(one['rmse_normalized']['z500'])} | {fmt(one['rmse_normalized']['z50'])} | "
            f"{fmt(one['rmse_physical']['overall'], 2)} | {fmt(one['rmse_physical']['z500'], 2)} | {fmt(one['rmse_physical']['z50'], 2)} |"
        )

    out_md.append("\n## VAE Rollout Checkpoints (normalized)\n")
    out_md.append("| Year | 24h | 48h | 72h |")
    out_md.append("|---|---:|---:|---:|")
    for year in ["1983", "1985"]:
        curve = vae[year]["autoregressive"]["rmse_normalized"]["mean"]
        out_md.append(f"| {year} | {fmt(curve[3])} | {fmt(curve[7])} | {fmt(curve[11])} |")

    out_md.append("\n## Diffusion Summary (cross-year means)\n")
    out_md.append("| Sparsity | Unobserved (norm) | Observed (norm) | Unobserved (phys) | Observed (phys) |")
    out_md.append("|---|---:|---:|---:|---:|")
    for s in ["30pct", "60pct", "90pct"]:
        sm = diff[s]["summary"]
        out_md.append(
            f"| {s} | {fmt(sm['rmse_normalized']['unobserved_mean'])} | {fmt_sci(sm['rmse_normalized']['observed_mean'])} | "
            f"{fmt(sm['rmse_physical']['unobserved_mean'], 2)} | {fmt_sci(sm['rmse_physical']['observed_mean'])} |"
        )

    md_path = DOCS_DIR / "FINAL_NUMBERS.md"
    md_path.write_text("\n".join(out_md) + "\n", encoding="utf-8")

    tex_lines = []
    tex_lines.append("% Auto-generated V2 diffusion table rows")
    for s in ["30pct", "60pct", "90pct"]:
        sm = diff[s]["summary"]
        tex_lines.append(
            f"{s} & {fmt(sm['rmse_normalized']['unobserved_mean'])} & {fmt_sci(sm['rmse_normalized']['observed_mean'])} & "
            f"{fmt(sm['rmse_physical']['unobserved_mean'], 2)} & {fmt_sci(sm['rmse_physical']['observed_mean'])} \\\\"
        )

    print(f"Wrote: {md_path}")
    print("V2 table rows:")
    for line in tex_lines[1:]:
        print(line)


if __name__ == "__main__":
    main()
