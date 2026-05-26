#!/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
"""Generate train/val/test index splits for the aminergic GPCR dataset.

Usage
-----
    python scripts/build_gpcr_splits.py
    python scripts/build_gpcr_splits.py --curated-path /path/to/curated.parquet \\
                                         --out-dir /path/to/splits/

This script produces three JSON split files, each containing integer indices
into the curated parquet's row order (same format as the kinase pipeline):

    random_split.json   {"train": [...], "val": [...], "test": [...]}
    scaffold_split.json {"train": [...], "val": [...], "test": [...]}
    target_split.json   {"train": [...], "val": [...], "test": [...]}

All splits use seed=42 and 80/10/10 proportions to match the kinase pipeline.

Split strategies
----------------
Random split:
    Randomly assigns rows to 80% train / 10% val / 10% test using
    stratify_col=None (the library function does not currently support
    stratification, but the 80/10/10 proportions keep all targets well
    represented given 36 targets and ~70k records). Seed=42 ensures
    reproducibility.

Scaffold split:
    Groups molecules by their generic Murcko scaffold.  Entire scaffold
    groups are assigned to one split — no scaffold leakage.  Large scaffolds
    go to train; smaller ones fill val then test (greedy assignment).

Target split:
    Holds out entire receptor targets.  With 36 aminergic targets and
    seed=42 this produces approximately 3–4 test targets (~10% by target
    count, which corresponds to a modest fraction of records).

    IMPORTANT: this is the exact same call used by the kinase pipeline —
    ``target_split(df, target_col="target_chembl_id", seed=42)``.  Targets
    are split individually (not by subfamily group), which is the faithful
    analog of the kinase protocol.

    The ``subfamily`` column in the curated data (dopamine, serotonin,
    adrenergic, histamine, muscarinic, trace_amine) is metadata for Plan 3's
    optional leave-one-family-out supplementary analysis.  It is NOT used by
    the target split in Plan 2.

    With only 36 targets the test fold is small (~3–4 targets).  This is a
    known limitation of the dataset size and is the faithful analog of the
    kinase protocol, where ``target_split`` also holds out individual targets
    per-seed.

Outputs
-------
    data/processed/v1/splits/random_split.json
    data/processed/v1/splits/scaffold_split.json
    data/processed/v1/splits/target_split.json

Integrity checks (5 assertions, all must pass)
-----------------------------------------------
1. No index overlap between any pair of folds for all three splits.
2. Total coverage: len(train)+len(val)+len(test) == len(df).
3. Random split target stratification: every target_chembl_id appears in all
   three folds (80/10/10 with 70k records keeps every target represented).
4. Scaffold split — no scaffold leakage: Murcko scaffold sets across
   train/val/test are disjoint.
5. Target split — test targets isolated: target_chembl_ids in the test fold
   do not overlap with train or val.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from target_affinity_ml.data.splits import (
    random_split,
    scaffold_split,
    target_split,
    save_splits,
)

# ── Paths relative to this file so script is cwd-agnostic ───────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CURATED = REPO_ROOT / "data" / "processed" / "v1" / "curated_activities.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed" / "v1" / "splits"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("build_gpcr_splits")


# ── Murcko scaffold helper (mirrors library internals) ───────────────────────

def _get_murcko_scaffold(smiles: str) -> str:
    """Compute generic Murcko scaffold SMILES for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        smi = Chem.MolToSmiles(generic)
        return smi if smi else "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


# ── Integrity checks ─────────────────────────────────────────────────────────

def _check_no_overlap(splits: dict[str, np.ndarray], split_name: str) -> None:
    """Check 1: no index overlap between any pair of folds."""
    train_set = set(splits["train"].tolist())
    val_set = set(splits["val"].tolist())
    test_set = set(splits["test"].tolist())

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set

    assert len(tv) == 0, f"[{split_name}] train∩val overlap: {len(tv)} indices"
    assert len(tt) == 0, f"[{split_name}] train∩test overlap: {len(tt)} indices"
    assert len(vt) == 0, f"[{split_name}] val∩test overlap: {len(vt)} indices"
    logger.info("[CHECK 1] %s — no index overlap: PASS", split_name)


def _check_coverage(splits: dict[str, np.ndarray], n_total: int, split_name: str) -> None:
    """Check 2: total coverage equals dataset size."""
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == n_total, (
        f"[{split_name}] coverage mismatch: {total} != {n_total}"
    )
    logger.info("[CHECK 2] %s — total coverage %d == %d: PASS", split_name, total, n_total)


def _check_random_target_stratification(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    target_col: str,
) -> None:
    """Check 3: every target appears in all three folds of the random split."""
    all_targets = set(df[target_col].unique())
    train_targets = set(df.iloc[splits["train"]][target_col].unique())
    val_targets = set(df.iloc[splits["val"]][target_col].unique())
    test_targets = set(df.iloc[splits["test"]][target_col].unique())

    missing_from_train = all_targets - train_targets
    missing_from_val = all_targets - val_targets
    missing_from_test = all_targets - test_targets

    if missing_from_train or missing_from_val or missing_from_test:
        logger.warning(
            "[CHECK 3] Random split target stratification — some targets missing: "
            "train missing=%s, val missing=%s, test missing=%s",
            missing_from_train, missing_from_val, missing_from_test,
        )
        # With 36 targets and ~70k records this should hold; assert if it doesn't
        assert not missing_from_train, f"Targets missing from random train: {missing_from_train}"
        assert not missing_from_val, f"Targets missing from random val: {missing_from_val}"
        assert not missing_from_test, f"Targets missing from random test: {missing_from_test}"
    else:
        logger.info(
            "[CHECK 3] Random split — all %d targets represented in each fold: PASS",
            len(all_targets),
        )


def _check_scaffold_no_leakage(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    smiles_col: str,
) -> None:
    """Check 4: scaffold sets are disjoint across train/val/test."""
    logger.info("[CHECK 4] Computing scaffolds for leakage check...")

    def scaffolds_for_idx(idx: np.ndarray) -> set[str]:
        smiles = df.iloc[idx][smiles_col]
        return {_get_murcko_scaffold(s) for s in smiles}

    train_sc = scaffolds_for_idx(splits["train"])
    val_sc = scaffolds_for_idx(splits["val"])
    test_sc = scaffolds_for_idx(splits["test"])

    tv = train_sc & val_sc
    tt = train_sc & test_sc
    vt = val_sc & test_sc

    # INVALID/NO_SCAFFOLD tags appear in all folds — exclude them
    noise = {"INVALID", "NO_SCAFFOLD"}
    tv -= noise
    tt -= noise
    vt -= noise

    assert len(tv) == 0, f"Scaffold leakage train∩val: {len(tv)} scaffolds"
    assert len(tt) == 0, f"Scaffold leakage train∩test: {len(tt)} scaffolds"
    assert len(vt) == 0, f"Scaffold leakage val∩test: {len(vt)} scaffolds"
    logger.info("[CHECK 4] Scaffold split — no scaffold leakage: PASS")


def _check_target_split_isolation(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    target_col: str,
) -> None:
    """Check 5: test targets do not appear in train or val."""
    train_targets = set(df.iloc[splits["train"]][target_col].unique())
    val_targets = set(df.iloc[splits["val"]][target_col].unique())
    test_targets = set(df.iloc[splits["test"]][target_col].unique())

    leaking_to_train = test_targets & train_targets
    leaking_to_val = test_targets & val_targets

    assert len(leaking_to_train) == 0, (
        f"[CHECK 5] Test targets appear in train: {leaking_to_train}"
    )
    assert len(leaking_to_val) == 0, (
        f"[CHECK 5] Test targets appear in val: {leaking_to_val}"
    )
    logger.info(
        "[CHECK 5] Target split — test targets isolated (%d test targets, none in train/val): PASS",
        len(test_targets),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--curated-path",
        type=Path,
        default=DEFAULT_CURATED,
        help="Path to curated_activities.parquet (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write split JSON files (default: %(default)s)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    curated_path: Path = args.curated_path
    out_dir: Path = args.out_dir

    # ── Load ────────────────────────────────────────────────────────────────
    logger.info("Loading curated activities from %s", curated_path)
    df = pd.read_parquet(curated_path)
    logger.info("Loaded %d records, %d compounds, %d targets",
                len(df),
                df["molecule_chembl_id"].nunique(),
                df["target_chembl_id"].nunique())

    smiles_col = "std_smiles"
    target_col = "target_chembl_id"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Random split ──────────────────────────────────────────────────────
    logger.info("=== Building random split ===")
    rand_splits = random_split(df, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42)
    logger.info(
        "Random split sizes: train=%d, val=%d, test=%d",
        len(rand_splits["train"]), len(rand_splits["val"]), len(rand_splits["test"]),
    )
    save_splits(rand_splits, out_dir / "random_split.json")

    # ── 2. Scaffold split ────────────────────────────────────────────────────
    logger.info("=== Building scaffold split ===")
    scaffold_splits = scaffold_split(
        df, smiles_col=smiles_col, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42
    )
    logger.info(
        "Scaffold split sizes: train=%d, val=%d, test=%d",
        len(scaffold_splits["train"]), len(scaffold_splits["val"]), len(scaffold_splits["test"]),
    )
    save_splits(scaffold_splits, out_dir / "scaffold_split.json")

    # ── 3. Target split ──────────────────────────────────────────────────────
    logger.info("=== Building target split ===")
    target_splits = target_split(df, target_col=target_col, seed=42)
    logger.info(
        "Target split sizes: train=%d, val=%d, test=%d",
        len(target_splits["train"]), len(target_splits["val"]), len(target_splits["test"]),
    )

    # Report which targets are in each fold
    train_targets = sorted(df.iloc[target_splits["train"]][target_col].unique())
    val_targets = sorted(df.iloc[target_splits["val"]][target_col].unique())
    test_targets = sorted(df.iloc[target_splits["test"]][target_col].unique())
    logger.info("Target split — train targets (%d): %s", len(train_targets), train_targets)
    logger.info("Target split — val targets (%d): %s", len(val_targets), val_targets)
    logger.info("Target split — TEST targets (%d): %s", len(test_targets), test_targets)

    save_splits(target_splits, out_dir / "target_split.json")

    # ── Integrity verification ───────────────────────────────────────────────
    logger.info("=== Running integrity checks ===")

    for name, spl in [
        ("random", rand_splits),
        ("scaffold", scaffold_splits),
        ("target", target_splits),
    ]:
        _check_no_overlap(spl, name)
        _check_coverage(spl, len(df), name)

    _check_random_target_stratification(df, rand_splits, target_col)
    _check_scaffold_no_leakage(df, scaffold_splits, smiles_col)
    _check_target_split_isolation(df, target_splits, target_col)

    logger.info("=== All 5 integrity checks PASSED ===")
    logger.info(
        "Splits written to %s:\n"
        "  random_split.json   (train=%d, val=%d, test=%d)\n"
        "  scaffold_split.json (train=%d, val=%d, test=%d)\n"
        "  target_split.json   (train=%d, val=%d, test=%d)",
        out_dir,
        len(rand_splits["train"]), len(rand_splits["val"]), len(rand_splits["test"]),
        len(scaffold_splits["train"]), len(scaffold_splits["val"]), len(scaffold_splits["test"]),
        len(target_splits["train"]), len(target_splits["val"]), len(target_splits["test"]),
    )


if __name__ == "__main__":
    main()
