#!/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
"""Compute per-target scaffold-diversity metrics for kinase + GPCR (Plan 3 T12).

Usage
-----
    python scripts/compute_scaffold_metrics.py

Outputs
-------
    results/supplement/per_target_metrics_scaffold.csv
"""
from __future__ import annotations

from pathlib import Path

from gpcr_aminergic_benchmarks.analyses.scaffold_diversity import compute_both_classes


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    gpcr = REPO_ROOT / "data/processed/v1/curated_activities.parquet"
    kinase = REPO_ROOT / "data/kinase_reference/curated_activities.parquet"
    out = REPO_ROOT / "results/supplement/per_target_metrics_scaffold.csv"

    print(f"GPCR  curated: {gpcr}")
    print(f"Kinase curated: {kinase}")
    print(f"Output:         {out}")

    df = compute_both_classes(gpcr, kinase, out)
    print(f"\nWrote {len(df)} rows ({(df.class_name == 'gpcr_aminergic').sum()} GPCR + "
          f"{(df.class_name == 'kinase').sum()} kinase)")


if __name__ == "__main__":
    main()
