"""Apply the library's scaffold-diversity metrics to both classes (kinase + GPCR).

See Plan 3 design spec Section 4 for the methodology; the heavy lifting lives
in target_affinity_ml.benchmarks.scaffold_diversity. This module is the thin
application layer: load the two curated parquets, dispatch to the library,
concatenate, save.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from target_affinity_ml.benchmarks.scaffold_diversity import compute_scaffold_metrics


def compute_both_classes(
    gpcr_curated: Path,
    kinase_curated: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Compute per-target scaffold metrics for GPCR + kinase; save concatenated CSV.

    Parameters
    ----------
    gpcr_curated : Path
        Path to GPCR curated parquet (e.g., data/processed/v1/curated_activities.parquet).
    kinase_curated : Path
        Path to kinase curated parquet (e.g., data/kinase_reference/curated_activities.parquet).
    output_csv : Path
        Destination CSV path. Parent directory is created if missing.

    Returns
    -------
    pd.DataFrame
        Concatenated per-target metrics with a ``class_name`` column tagging the source.
    """
    gpcr = pd.read_parquet(gpcr_curated)
    gpcr_m = compute_scaffold_metrics(gpcr).assign(class_name="gpcr_aminergic")

    kin = pd.read_parquet(kinase_curated)
    kin_m = compute_scaffold_metrics(kin).assign(class_name="kinase")

    out = pd.concat([gpcr_m, kin_m], ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
