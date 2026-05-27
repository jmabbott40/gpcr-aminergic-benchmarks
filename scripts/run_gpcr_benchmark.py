"""Run the GPCR aminergic benchmark: 7 models × 3 splits × 5 seeds = 105 runs.

Adapted from kinase-affinity-baselines/scripts/rerun_kinase_v1.py.
Key difference: data_dir_override points at this repo's data/processed/v1/,
not the kinase repo's data. This is the L3 data_dir parameter added to the
library in Task 3 of Plan 2.

Model names (internal registry names → on-disk config filenames):
    random_forest → rf_baseline.yaml
    xgboost       → xgb_baseline.yaml
    elasticnet    → elasticnet_baseline.yaml
    mlp           → mlp_baseline.yaml
    esm_fp_mlp    → esm_fp_mlp.yaml
    gnn           → gnn.yaml
    fusion        → fusion.yaml

Output: results/gpcr_v1_benchmark/<model>__<split>__seed<N>__metrics.json
        results/gpcr_v1_benchmark/all_seeds_metrics.csv

ESM-2 note: esm_fp_mlp / fusion / gnn require esm2_embeddings.npz (Task 11).
Run RF/XGB/ElasticNet/MLP first if embeddings aren't ready yet.

Estimated wall-clock: ~2 days at 4-way GPU parallelism (Task 13 on AWS).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from target_affinity_ml.training import train_and_evaluate
from target_affinity_ml.training.deep_trainer import deep_train_and_evaluate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GPCR_REPO = Path(__file__).resolve().parent.parent

# Models routed through the baseline trainer (config["features"]["type"] is a string)
BASELINE_MODELS = {"random_forest", "xgboost", "elasticnet", "mlp"}
# Models routed through the deep trainer (config["features"] has ligand/protein keys)
DEEP_MODELS = {"esm_fp_mlp", "gnn", "fusion"}

# Mirror the kinase v1 protocol: same 7 models in same order
ALL_MODELS = [
    "random_forest",
    "xgboost",
    "elasticnet",
    "mlp",
    "esm_fp_mlp",
    "gnn",
    "fusion",
]
ALL_SPLITS = ["random", "scaffold", "target"]
ALL_SEEDS = [42, 123, 456, 789, 1024]  # match kinase preprint v1 seeds (spec Section 6.2)

DATASET_VERSION = "v1"

# Model registry name → on-disk YAML filename.
# Mirrors rerun_kinase_v1.py CONFIG_FILENAMES exactly (Plan 1 filename-mismatch
# bug fix: the registry uses long names like "random_forest" but YAML files use
# short names like "rf_baseline.yaml").
CONFIG_FILENAMES = {
    "random_forest": "rf_baseline.yaml",
    "xgboost":       "xgb_baseline.yaml",
    "elasticnet":    "elasticnet_baseline.yaml",
    "mlp":           "mlp_baseline.yaml",
    "esm_fp_mlp":    "esm_fp_mlp.yaml",
    "gnn":           "gnn.yaml",
    "fusion":        "fusion.yaml",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def config_path_for(model: str, configs_dir: Path) -> Path:
    """Resolve a model's YAML config path (registry name → on-disk filename)."""
    return configs_dir / CONFIG_FILENAMES[model]


def output_stem(model: str, split: str, seed: int) -> str:
    """Canonical output filename stem: <model>__<split>__seed<N>."""
    return f"{model}__{split}__seed{seed}"


def _existing_successes(metrics_csv: Path) -> set[tuple[str, str, int]]:
    """Read prior partial results to skip already-successful (model, split, seed)
    triples. Used by --resume to avoid redoing baselines after a bug fix.
    """
    if not metrics_csv.exists():
        return set()
    df = pd.read_csv(metrics_csv)
    # NaN-safe: treat null/empty error column as a successful run (L5 fix)
    successes = df[df["error"].isna() | (df["error"] == "")]
    return {(r["model"], r["split"], int(r["seed"])) for _, r in successes.iterrows()}


def _execute_one(
    model: str,
    split: str,
    seed: int,
    configs_dir: Path,
    data_dir: Path,
    results_dir: Path,
) -> dict:
    """Dispatch to the correct trainer and return its metrics dict."""
    cfg_path = config_path_for(model, configs_dir)
    suffix = f"_gpcr_seed{seed}"

    kwargs = dict(
        config_path=cfg_path,
        split_strategy=split,
        dataset_version=DATASET_VERSION,
        training_seed=seed,
        data_dir_override=data_dir,
        output_suffix=suffix,
    )
    if model in DEEP_MODELS:
        return deep_train_and_evaluate(**kwargs)
    if model in BASELINE_MODELS:
        return train_and_evaluate(**kwargs)
    raise ValueError(f"Unknown model: {model!r}")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_gpcr_benchmark(
    models: list[str],
    splits: list[str],
    seeds: list[int],
    data_dir: Path,
    results_dir: Path,
    configs_dir: Path,
    resume: bool = False,
    max_runs: int | None = None,
) -> pd.DataFrame:
    """Execute all runs and write per-seed metrics.

    Parameters
    ----------
    models : list[str]
        Model registry names to run (subset of ALL_MODELS).
    splits : list[str]
        Split strategies to run.
    seeds : list[int]
        Random seeds.
    data_dir : Path
        Path to data/processed/v1/ (the GPCR repo's data, not the kinase repo's).
    results_dir : Path
        Root output directory; per-run JSONs land here.
    configs_dir : Path
        Directory containing the 7 model YAML configs.
    resume : bool
        If True, read all_seeds_metrics.csv and skip already-successful runs.
    max_runs : int or None
        If set, stop after this many runs (smoke-test mode).
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = results_dir / "all_seeds_metrics.csv"
    skip_set: set[tuple[str, str, int]] = (
        _existing_successes(metrics_csv) if resume else set()
    )
    if skip_set:
        print(f"Resume mode: skipping {len(skip_set)} already-successful runs.")

    # Preserve prior successful rows so the final CSV is complete
    rows: list[dict] = []
    if resume and metrics_csv.exists():
        prev = pd.read_csv(metrics_csv)
        rows = prev[prev["error"].isna() | (prev["error"] == "")].to_dict("records")

    total_runs = len(models) * len(splits) * len(seeds)
    run_idx = 0
    runs_executed = 0
    overall_start = time.time()

    for model in models:
        for split in splits:
            for seed in seeds:
                run_idx += 1
                if (model, split, seed) in skip_set:
                    print(
                        f"[{run_idx}/{total_runs}] {model} | {split} | seed={seed}"
                        " ... SKIPPED (already done)"
                    )
                    continue

                if max_runs is not None and runs_executed >= max_runs:
                    print(f"--max-runs {max_runs} reached; stopping early.")
                    break

                start = time.time()
                print(
                    f"[{run_idx}/{total_runs}] {model} | {split} | seed={seed} ... ",
                    end="",
                    flush=True,
                )
                try:
                    result = _execute_one(
                        model, split, seed, configs_dir, data_dir, results_dir,
                    )
                except Exception as e:
                    elapsed = time.time() - start
                    print(f"FAILED in {elapsed:.0f}s: {type(e).__name__}: {e}")
                    rows.append({
                        "model": model,
                        "split": split,
                        "seed": seed,
                        "test_rmse": None,
                        "test_r2": None,
                        "test_pearson_r": None,
                        "wallclock_seconds": elapsed,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    runs_executed += 1
                    continue

                elapsed = time.time() - start
                rmse = result.get("test_rmse")
                # NaN-safe error counter: only real exceptions are errors (L5 fix)
                rows.append({
                    "model": model,
                    "split": split,
                    "seed": seed,
                    "test_rmse": rmse,
                    "test_mae": result.get("test_mae"),
                    "test_r2": result.get("test_r2"),
                    "test_pearson_r": result.get("test_pearson_r"),
                    "test_spearman_rho": result.get("test_spearman_rho"),
                    "test_auroc": result.get("test_auroc"),
                    "wallclock_seconds": elapsed,
                    "error": None,
                })
                rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
                print(f"OK in {elapsed:.0f}s, RMSE={rmse_str}")

                # Write per-run JSON alongside the aggregate CSV
                run_json = results_dir / f"{output_stem(model, split, seed)}__metrics.json"
                with open(run_json, "w") as f:
                    json.dump(result, f, indent=2, default=_json_default)

                runs_executed += 1

                # Periodic progress save in case the process crashes mid-run
                if runs_executed % 10 == 0:
                    pd.DataFrame(rows).to_csv(
                        results_dir / "all_seeds_metrics_partial.csv",
                        index=False,
                    )
            else:
                continue
            break  # max_runs reached inner loop
        else:
            continue
        break  # max_runs reached middle loop

    # Final aggregate CSV
    df = pd.DataFrame(rows)
    df.to_csv(metrics_csv, index=False)
    # Drop the partial file
    partial = results_dir / "all_seeds_metrics_partial.csv"
    if partial.exists():
        partial.unlink()

    overall_elapsed = time.time() - overall_start
    print()
    print(f"Runs complete ({runs_executed} executed) in {overall_elapsed/3600:.1f}h.")
    print(f"Per-seed metrics: {metrics_csv}")

    # NaN-safe error count (L5 fix from Task 4)
    n_errors = sum(
        1 for r in rows
        if r.get("error") is not None
        and str(r.get("error")).strip() not in ("", "nan")
    )
    if n_errors:
        print(f"WARNING: {n_errors}/{runs_executed} runs failed. See 'error' column.")

    return df


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serializer for numpy scalars / arrays."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=GPCR_REPO / "data" / "processed" / "v1",
        help="Path to data/processed/v1/ (default: <repo>/data/processed/v1)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=GPCR_REPO / "results" / "gpcr_v1_benchmark",
        help="Root output directory (default: <repo>/results/gpcr_v1_benchmark)",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=GPCR_REPO / "configs",
        help="Directory containing model YAML configs (default: <repo>/configs)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        metavar="MODEL",
        help=(
            "Models to run. Default: all 7. "
            f"Choices: {ALL_MODELS}"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=ALL_SPLITS,
        choices=ALL_SPLITS,
        metavar="SPLIT",
        help="Split strategies (default: random scaffold target)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=ALL_SEEDS,
        metavar="N",
        help="Random seeds (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip (model, split, seed) triples that already succeeded in "
            "results/gpcr_v1_benchmark/all_seeds_metrics.csv."
        ),
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N runs (smoke-test mode).",
    )
    args = parser.parse_args()

    # Validate config files exist before starting
    for model in args.models:
        cfg = args.configs_dir / CONFIG_FILENAMES[model]
        if not cfg.exists():
            parser.error(
                f"Config file not found for model '{model}': {cfg}\n"
                "Ensure configs/ is populated (copy from target-affinity-ml/configs/)."
            )

    df = run_gpcr_benchmark(
        models=args.models,
        splits=args.splits,
        seeds=args.seeds,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        configs_dir=args.configs_dir,
        resume=args.resume,
        max_runs=args.max_runs,
    )

    # Exit non-zero if any run failed (mirrors kinase script)
    if df["error"].notna().any():
        sys.exit(1)


if __name__ == "__main__":
    main()
