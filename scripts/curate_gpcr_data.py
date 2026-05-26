#!/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
"""Curate aminergic GPCR bioactivity data using the class-agnostic pipeline.

Usage
-----
    python scripts/curate_gpcr_data.py

This script:
    1. Loads configs/dataset_aminergic_v1.yaml.
    2. Rebuilds the aminergic TargetClassConfig from
       data/processed/v1/resolved_target_ids.json (no re-query of ChEMBL).
    3. Calls target_affinity_ml.data.curate.curate_activities, which:
         - Standardizes molecules (RDKit)
         - Converts standard_value -> pactivity
         - Deduplicates (canonical_smiles, target_chembl_id, standard_type) -> median
         - Flags noisy compounds (std > 1.0, n >= 3)
         - Applies quality filters [pactivity 3.0, 12.0]
         - Labels active/inactive (pactivity >= 6.0)
         - Populates subfamily from config.subfamily_map (explicit-target-list path)
    4. Saves curated parquet and curation_stats.json.
    5. Smoke-verifies key properties.

Outputs
-------
    data/processed/v1/curated_activities.parquet
    data/processed/v1/curation_stats.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ── Resolve paths relative to this file so script is cwd-agnostic ──────────
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "dataset_aminergic_v1.yaml"
RESOLVED_IDS_PATH = REPO_ROOT / "data" / "processed" / "v1" / "resolved_target_ids.json"
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("curate_gpcr_data")

VALID_SUBFAMILIES = {"dopamine", "serotonin", "adrenergic", "histamine", "muscarinic", "trace_amine"}


def compute_scaffold_count(smiles_series: pd.Series) -> int:
    """Count unique Bemis-Murcko generic scaffolds in a SMILES series."""
    scaffolds: set[str] = set()
    for smi in smiles_series.dropna():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaf_mol = MurckoScaffold.GetScaffoldForMol(mol)
            generic = MurckoScaffold.MakeScaffoldGeneric(scaf_mol)
            if generic is not None:
                scaf_smi = Chem.MolToSmiles(generic)
                if scaf_smi:
                    scaffolds.add(scaf_smi)
        except Exception:
            continue
    return len(scaffolds)


def main() -> None:
    # ── Step 1: Load dataset config ────────────────────────────────────────
    logger.info("=== Step 1: Loading dataset config from %s ===", CONFIG_PATH)
    with open(CONFIG_PATH) as fh:
        dataset_config: dict = yaml.safe_load(fh)
    logger.info("Config version: %s", dataset_config.get("version"))

    # ── Step 2: Rebuild aminergic TargetClassConfig ────────────────────────
    logger.info(
        "=== Step 2: Rebuilding TargetClassConfig from %s ===", RESOLVED_IDS_PATH
    )
    with open(RESOLVED_IDS_PATH) as fh:
        resolved_ids: dict[str, str] = json.load(fh)
    logger.info("Loaded %d resolved target IDs.", len(resolved_ids))

    from gpcr_aminergic_benchmarks.target_class import build_aminergic_config

    aminergic_config = build_aminergic_config(resolved_ids)
    logger.info(
        "Built aminergic TargetClassConfig: class_name=%r, %d explicit targets.",
        aminergic_config.class_name,
        len(aminergic_config.explicit_target_ids),
    )

    # ── Step 3: Run curation ───────────────────────────────────────────────
    logger.info("=== Step 3: Running curate_activities ===")
    from target_affinity_ml.data.curate import curate_activities

    curation_stats: dict = {}
    df = curate_activities(
        config=aminergic_config,
        dataset_config=dataset_config,
        raw_dir=RAW_DIR,
        stats=curation_stats,
    )
    logger.info("curate_activities returned %d records.", len(df))

    # ── Step 4: Build full stats dict ─────────────────────────────────────
    logger.info("=== Step 4: Computing curation statistics ===")

    # Per-subfamily breakdown
    subfamily_breakdown: dict[str, dict] = {}
    for subfamily, grp in df.groupby("subfamily"):
        subfamily_breakdown[str(subfamily)] = {
            "records": int(len(grp)),
            "unique_compounds": int(grp["std_smiles"].nunique()),
            "unique_targets": int(grp["target_chembl_id"].nunique()),
        }

    # Per-target record distribution
    per_target_counts = df["target_chembl_id"].value_counts()
    # Reverse resolved_ids to get gene symbol for reporting
    chembl_to_gene = {v: k for k, v in resolved_ids.items()}
    per_target_table = [
        {
            "target_chembl_id": tid,
            "gene_symbol": chembl_to_gene.get(tid, tid),
            "subfamily": str(aminergic_config.subfamily_map.get(tid, "unknown")),
            "records": int(count),
        }
        for tid, count in per_target_counts.items()
    ]

    # Top-5 and bottom-5 for logging
    top5 = per_target_table[:5]
    bottom5 = per_target_table[-5:]

    # Scaffold count (run on unique std_smiles)
    logger.info("Computing Bemis-Murcko scaffold count (may take a moment)...")
    unique_smiles = df["std_smiles"].dropna().unique()
    n_scaffolds = compute_scaffold_count(pd.Series(unique_smiles))
    logger.info("Unique Bemis-Murcko scaffolds: %d", n_scaffolds)

    stats: dict = {
        "n_raw_records": 89339,  # from Task 7 fetch output
        "n_curated_records": int(len(df)),
        "n_unique_compounds": int(df["std_smiles"].nunique()),
        "n_unique_targets": int(df["target_chembl_id"].nunique()),
        "n_noisy_records": int(df["is_noisy"].sum()),
        "n_active": int(df["is_active"].sum()),
        "n_inactive": int((~df["is_active"]).sum()),
        "activity_type_breakdown": df["standard_type"].value_counts().to_dict(),
        "n_unique_scaffolds": n_scaffolds,
        "per_subfamily": subfamily_breakdown,
        "per_target_distribution": {
            "min": int(per_target_counts.min()),
            "median": float(per_target_counts.median()),
            "max": int(per_target_counts.max()),
            "table": per_target_table,
        },
        "standardization": curation_stats.get("standardization", {}),
    }

    # ── Step 5: Save outputs ───────────────────────────────────────────────
    logger.info("=== Step 5: Saving outputs ===")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    curated_path = PROCESSED_DIR / "curated_activities.parquet"
    df = df.reset_index(drop=True)
    df.to_parquet(curated_path, index=False)
    logger.info("Saved curated parquet: %s (%d records)", curated_path, len(df))

    stats_path = PROCESSED_DIR / "curation_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Saved curation stats: %s", stats_path)

    # ── Step 6: Smoke verification ─────────────────────────────────────────
    logger.info("=== Step 6: Smoke verification ===")
    errors: list[str] = []

    # 6a. Every row has non-null subfamily in valid set
    null_subfamily = df["subfamily"].isna().sum()
    invalid_subfamily = (~df["subfamily"].isin(VALID_SUBFAMILIES)).sum()
    if null_subfamily > 0:
        errors.append(f"FAIL: {null_subfamily} rows have null subfamily")
    elif invalid_subfamily > 0:
        errors.append(
            f"FAIL: {invalid_subfamily} rows have unexpected subfamily value"
        )
    else:
        logger.info("  [PASS] All %d rows have valid non-null subfamily.", len(df))

    # 6b. Every row has pactivity in [3.0, 12.0]
    out_of_range = ((df["pactivity"] < 3.0) | (df["pactivity"] > 12.0)).sum()
    if out_of_range > 0:
        errors.append(
            f"FAIL: {out_of_range} rows have pactivity outside [3.0, 12.0]"
        )
    else:
        pmin = df["pactivity"].min()
        pmax = df["pactivity"].max()
        logger.info(
            "  [PASS] All rows have pactivity in [3.0, 12.0] (observed: [%.3f, %.3f]).",
            pmin, pmax,
        )

    # 6c. Every row has target_chembl_id in the 36 resolved IDs
    resolved_set = set(resolved_ids.values())
    unknown_targets = (~df["target_chembl_id"].isin(resolved_set)).sum()
    if unknown_targets > 0:
        errors.append(
            f"FAIL: {unknown_targets} rows have target_chembl_id not in resolved 36"
        )
    else:
        n_unique_targets = df["target_chembl_id"].nunique()
        logger.info(
            "  [PASS] All rows have target_chembl_id in resolved IDs (%d unique targets).",
            n_unique_targets,
        )

    # 6d. Unique compounds < raw 33,814 (raw unique by molecule_chembl_id in Task 7)
    # The task spec says: unique compounds after curation < raw 33,814
    n_unique_compounds = df["std_smiles"].nunique()
    RAW_UNIQUE_COMPOUNDS = 33814  # from Task 7 fetch summary
    if n_unique_compounds >= RAW_UNIQUE_COMPOUNDS:
        errors.append(
            f"FAIL: unique compounds {n_unique_compounds} >= raw {RAW_UNIQUE_COMPOUNDS} "
            "(expected some reduction from standardization + filtering)"
        )
    else:
        logger.info(
            "  [PASS] Unique compounds %d < raw %d (reduction from standardization + dedup).",
            n_unique_compounds,
            RAW_UNIQUE_COMPOUNDS,
        )

    if errors:
        for e in errors:
            logger.error("SMOKE CHECK — %s", e)
        sys.exit(1)
    else:
        logger.info("All smoke checks passed.")

    # ── Final summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("CURATION COMPLETE")
    logger.info("  Raw records       : %d", stats["n_raw_records"])
    logger.info("  Curated records   : %d", stats["n_curated_records"])
    logger.info("  Unique compounds  : %d", stats["n_unique_compounds"])
    logger.info("  Unique targets    : %d", stats["n_unique_targets"])
    logger.info("  Noisy records     : %d", stats["n_noisy_records"])
    logger.info(
        "  Active            : %d (%.1f%%)",
        stats["n_active"],
        stats["n_active"] / stats["n_curated_records"] * 100,
    )
    logger.info(
        "  Inactive          : %d (%.1f%%)",
        stats["n_inactive"],
        stats["n_inactive"] / stats["n_curated_records"] * 100,
    )
    logger.info("  Unique scaffolds  : %d", stats["n_unique_scaffolds"])
    logger.info("  Activity types    : %s", stats["activity_type_breakdown"])
    logger.info("  Per-subfamily:")
    for sf, sf_stats in sorted(subfamily_breakdown.items()):
        logger.info(
            "    %-15s  records=%-6d  compounds=%-6d  targets=%d",
            sf, sf_stats["records"], sf_stats["unique_compounds"], sf_stats["unique_targets"]
        )
    logger.info(
        "  Per-target (min/median/max): %d / %.0f / %d",
        stats["per_target_distribution"]["min"],
        stats["per_target_distribution"]["median"],
        stats["per_target_distribution"]["max"],
    )
    logger.info("  Top-5 targets by record count:")
    for row in top5:
        logger.info(
            "    %-10s %-15s (%s): %d records",
            row["target_chembl_id"], row["gene_symbol"], row["subfamily"], row["records"]
        )
    logger.info("  Bottom-5 targets by record count:")
    for row in bottom5:
        logger.info(
            "    %-10s %-15s (%s): %d records",
            row["target_chembl_id"], row["gene_symbol"], row["subfamily"], row["records"]
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
