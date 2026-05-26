#!/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
"""Generate and cache molecular features for the aminergic GPCR dataset.

Usage
-----
    python scripts/build_gpcr_features.py
    python scripts/build_gpcr_features.py --force   # Recompute even if cached

This script computes two complementary molecular representations for every
unique compound in the curated GPCR dataset, using the same library call and
the same parameters as the kinase pipeline for cross-class consistency:

1. **Morgan fingerprints** (binary, 2048-bit, radius 2)
   Saved to: data/processed/v1/features/morgan_fp.npz  (key: "fingerprints")

2. **RDKit 2D descriptors** (~200 physicochemical/topological properties)
   Saved to: data/processed/v1/features/rdkit_descriptors.npz
              (keys: "descriptors", "descriptor_names")

A SMILES index file preserving row order is also written:
   data/processed/v1/features/smiles_index.json

All outputs are written under data/processed/ (the ``data_dir`` argument),
using the version "v1" from configs/features_v1.yaml to resolve subdirectories
(i.e. data/processed/v1/features/).

Inputs
------
    configs/features_v1.yaml                       — feature parameters
    data/processed/v1/curated_activities.parquet   — curated records (Task 8)

Outputs
-------
    data/processed/v1/features/morgan_fp.npz
    data/processed/v1/features/rdkit_descriptors.npz
    data/processed/v1/features/smiles_index.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ── Resolve paths relative to this file so script is cwd-agnostic ───────────
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "features_v1.yaml"
# data_dir is the directory CONTAINING the v1/ versioned subdir
DATA_DIR = REPO_ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("build_gpcr_features")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cached feature files already exist.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=== GPCR Molecular Feature Generation ===")
    logger.info("Config path : %s", CONFIG_PATH)
    logger.info("Data dir    : %s", DATA_DIR)
    logger.info("Force recompute: %s", args.force)

    # ── Step 1: Compute and cache features via library ───────────────────────
    logger.info("=== Step 1: Calling compute_and_cache_features ===")
    from target_affinity_ml.features import compute_and_cache_features

    saved_paths = compute_and_cache_features(
        config_path=CONFIG_PATH,
        force=args.force,
        data_dir=DATA_DIR,
    )

    logger.info("compute_and_cache_features returned paths:")
    for key, path in saved_paths.items():
        logger.info("  %-20s -> %s", key, path)

    # ── Step 2: Verify and log output shapes ────────────────────────────────
    logger.info("=== Step 2: Verifying output files and shapes ===")

    # Morgan fingerprints
    fp_path: Path = saved_paths["morgan_fp"]
    if not fp_path.exists():
        logger.error("FAIL: Morgan fingerprint file missing: %s", fp_path)
        sys.exit(1)
    fp_data = np.load(fp_path)
    fp_matrix = fp_data["fingerprints"]
    logger.info(
        "Morgan fingerprints: shape=%s, dtype=%s, file=%.2f MB",
        fp_matrix.shape, fp_matrix.dtype, fp_path.stat().st_size / 1e6,
    )
    assert fp_matrix.ndim == 2, f"Expected 2-D Morgan matrix, got shape {fp_matrix.shape}"
    assert fp_matrix.shape[1] == 2048, (
        f"Expected 2048 bits, got {fp_matrix.shape[1]}"
    )

    # RDKit descriptors
    desc_path: Path = saved_paths["rdkit_descriptors"]
    if not desc_path.exists():
        logger.error("FAIL: RDKit descriptors file missing: %s", desc_path)
        sys.exit(1)
    desc_data = np.load(desc_path, allow_pickle=True)
    desc_matrix = desc_data["descriptors"]
    desc_names = desc_data["descriptor_names"].tolist()
    logger.info(
        "RDKit descriptors: shape=%s, dtype=%s, n_descriptor_names=%d, file=%.2f MB",
        desc_matrix.shape, desc_matrix.dtype, len(desc_names),
        desc_path.stat().st_size / 1e6,
    )
    assert desc_matrix.ndim == 2, (
        f"Expected 2-D descriptor matrix, got shape {desc_matrix.shape}"
    )
    assert desc_matrix.shape[1] == len(desc_names), (
        f"Descriptor matrix columns {desc_matrix.shape[1]} != "
        f"len(descriptor_names) {len(desc_names)}"
    )

    # SMILES index
    smiles_path: Path = saved_paths["smiles_index"]
    if not smiles_path.exists():
        logger.error("FAIL: SMILES index missing: %s", smiles_path)
        sys.exit(1)
    with open(smiles_path) as fh:
        smiles_list = json.load(fh)
    logger.info("SMILES index: %d entries", len(smiles_list))

    # Row count consistency
    n_morgan = fp_matrix.shape[0]
    n_desc = desc_matrix.shape[0]
    n_smiles = len(smiles_list)
    if not (n_morgan == n_desc == n_smiles):
        logger.error(
            "FAIL: Row count mismatch — Morgan=%d, RDKit=%d, SMILES index=%d",
            n_morgan, n_desc, n_smiles,
        )
        sys.exit(1)
    logger.info(
        "[PASS] All three outputs have consistent row count: %d compounds",
        n_smiles,
    )

    # ── Step 3: Re-load via library loaders ─────────────────────────────────
    logger.info("=== Step 3: Re-loading via library loaders ===")
    from target_affinity_ml.features import (
        load_morgan_fingerprints,
        load_rdkit_descriptors,
    )

    fp_loaded, smiles_loaded_fp = load_morgan_fingerprints(
        version="v1", data_dir=DATA_DIR
    )
    logger.info(
        "load_morgan_fingerprints: shape=%s, n_smiles=%d",
        fp_loaded.shape, len(smiles_loaded_fp),
    )
    assert fp_loaded.shape == fp_matrix.shape, (
        f"Re-loaded Morgan shape {fp_loaded.shape} != cached {fp_matrix.shape}"
    )

    desc_loaded, names_loaded, smiles_loaded_desc = load_rdkit_descriptors(
        version="v1", data_dir=DATA_DIR
    )
    logger.info(
        "load_rdkit_descriptors: shape=%s, n_descriptor_names=%d, n_smiles=%d",
        desc_loaded.shape, len(names_loaded), len(smiles_loaded_desc),
    )
    assert desc_loaded.shape == desc_matrix.shape, (
        f"Re-loaded RDKit shape {desc_loaded.shape} != cached {desc_matrix.shape}"
    )

    logger.info("[PASS] Library loaders successfully read both feature files.")

    # ── Final summary ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FEATURE GENERATION COMPLETE")
    logger.info("  Compounds         : %d", n_smiles)
    logger.info("  Morgan FP shape   : %s (n x 2048 bits)", fp_matrix.shape)
    logger.info("  RDKit desc shape  : %s (n x %d descriptors)",
                desc_matrix.shape, desc_matrix.shape[1])
    logger.info("  SMILES index      : %d entries", n_smiles)
    logger.info("  morgan_fp.npz     : %s", fp_path)
    logger.info("  rdkit_desc.npz    : %s", desc_path)
    logger.info("  smiles_index.json : %s", smiles_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
