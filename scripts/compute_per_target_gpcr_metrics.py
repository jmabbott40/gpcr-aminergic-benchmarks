#!/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
"""Compute per-target test-set metrics for GPCR-aminergic benchmark runs.

Mirrors the kinase per-target CSV format at
``data/kinase_reference/benchmark_v1/per_target/per_target_<model>_<split>.csv``.
Plan 2 saved per-seed prediction NPZs but didn't produce per-target rollups;
this script fills that gap for Plan 3's H3 cross-class analysis (T18 + T19).

For each (model, split): aggregates per-seed predictions across all 5 seeds,
joins to the curated parquet via the split's test row indices to recover
target_chembl_id, then computes per-target rmse/mae/r2/pearson/spearman.

Output: ``data/processed/v1/per_target_<model>_<split>.csv`` (one file per
(model, split) — 7 models × 3 splits = 21 files, mirroring the kinase set).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIRS = [REPO_ROOT / f"results/predictions_gpcr_seed{s}" for s in (42, 123, 456, 789, 1024)]
SPLITS_DIR = REPO_ROOT / "data/processed/v1/splits"
CURATED = REPO_ROOT / "data/processed/v1/curated_activities.parquet"
OUT_DIR = REPO_ROOT / "data/processed/v1/per_target"

MODELS = ["random_forest", "xgboost", "elasticnet", "mlp", "esm_fp_mlp", "fusion", "gnn"]
SPLITS = ["random", "scaffold", "target"]

logger = logging.getLogger(__name__)


def _per_target_metrics(group: pd.DataFrame) -> pd.Series:
    """Compute rmse, mae, r2, pearson_r, spearman_rho on a per-target subgroup."""
    y_true = group["y_true"].to_numpy()
    y_pred = group["y_pred"].to_numpy()
    n = len(group)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if n >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(stats.pearsonr(y_true, y_pred).statistic)
        spearman_rho = float(stats.spearmanr(y_true, y_pred).statistic)
    else:
        pearson_r = float("nan")
        spearman_rho = float("nan")
    return pd.Series({
        "n_compounds": n,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
    })


def _compute_one(model: str, split: str, curated: pd.DataFrame, split_test_idx: list[int]) -> pd.DataFrame:
    """Pool predictions across seeds, compute per-target metrics, return DataFrame."""
    rows = []
    target_lookup = curated.iloc[split_test_idx]["target_chembl_id"].reset_index(drop=True)
    for pred_dir in PRED_DIRS:
        npz_path = pred_dir / f"{model}_{split}.npz"
        if not npz_path.exists():
            logger.warning("Missing prediction file: %s", npz_path)
            continue
        data = np.load(npz_path)
        y_true = data["y_test_true"]
        y_pred = data["y_test_pred"]
        if len(y_true) != len(split_test_idx):
            logger.warning("Length mismatch for %s/%s in %s: %d vs %d",
                           model, split, pred_dir.name, len(y_true), len(split_test_idx))
            continue
        for tgt, yt, yp in zip(target_lookup, y_true, y_pred):
            rows.append({"target_chembl_id": tgt, "y_true": float(yt), "y_pred": float(yp)})
    if not rows:
        return pd.DataFrame(columns=["target_id", "n_compounds", "rmse", "mae", "r2", "pearson_r", "spearman_rho"])
    df = pd.DataFrame(rows)
    per_target = df.groupby("target_chembl_id").apply(_per_target_metrics, include_groups=False).reset_index()
    per_target = per_target.rename(columns={"target_chembl_id": "target_id"})
    return per_target[["target_id", "n_compounds", "rmse", "mae", "r2", "pearson_r", "spearman_rho"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    curated = pd.read_parquet(CURATED)
    logger.info("Loaded curated GPCR data: %d rows, %d targets",
                len(curated), curated["target_chembl_id"].nunique())

    for split in SPLITS:
        with (SPLITS_DIR / f"{split}_split.json").open() as fh:
            split_data = json.load(fh)
        test_idx = split_data["test"]
        logger.info("Split %s: %d test rows", split, len(test_idx))
        for model in MODELS:
            out_path = OUT_DIR / f"per_target_{model}_{split}.csv"
            if out_path.exists():
                logger.info("  [cached] %s", out_path.name)
                continue
            per_target = _compute_one(model, split, curated, test_idx)
            per_target.to_csv(out_path, index=False)
            logger.info("  Wrote %d targets to %s", len(per_target), out_path.name)

    logger.info("DONE. Per-target metrics under %s", OUT_DIR)


if __name__ == "__main__":
    main()
