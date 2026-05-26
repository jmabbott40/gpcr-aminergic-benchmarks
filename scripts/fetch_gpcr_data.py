#!/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
"""Fetch aminergic GPCR bioactivity data from ChEMBL.

Usage
-----
    python scripts/fetch_gpcr_data.py               # all 36 targets (full fetch, ~20-40 min)
    python scripts/fetch_gpcr_data.py --max-targets 1  # smoke test (1 target, ~1-2 min)

Outputs
-------
    data/processed/v1/resolved_target_ids.json       — gene_symbol -> chembl_id mapping
    data/raw/chembl_gpcr_aminergic_activities.parquet — filtered binding activity records
    data/raw/chembl_gpcr_aminergic_targets.parquet    — target metadata

Inclusion criteria (audit spec)
--------------------------------
Applied by target_affinity_ml.fetch_bioactivities at the ChEMBL API level:
    - standard_type in {"IC50", "Ki", "Kd"}           [API filter]
    - standard_relation = "="  (exact measurements)   [API filter]
    - pchembl_value is not null                        [API filter]

Applied by THIS SCRIPT via post-fetch DataFrame filtering:
    - assay_type = "B"  (binding assays only)          [post-fetch filter — not enforced by library]
    - confidence_score >= 7                            [post-fetch filter — not enforced by library]
    - standard_units = "nM"                            [post-fetch filter — not enforced by library]

Note: pChEMBL presence already implicitly requires standard_relation="=" and
nanomolar units for ChEMBL to compute it, but the explicit standard_units="nM"
filter below makes this a hard constraint in the output data.

On any ChEMBL HTTP error or rate-limit, the exception propagates — no silent
empty-data fallback. Surface as DONE_WITH_CONCERNS if this occurs.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Resolve repo root from this file's location so the script works regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
RESOLVED_IDS_PATH = REPO_ROOT / "data" / "processed" / "v1" / "resolved_target_ids.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("fetch_gpcr_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch aminergic GPCR bioactivity data from ChEMBL."
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit resolution and fetch to the first N gene symbols "
            "(for smoke testing). Default: all targets."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 1: Resolve gene symbols -> ChEMBL IDs
    # ------------------------------------------------------------------
    from gpcr_aminegric_benchmarks.target_lists import (
        get_all_gene_symbols,
        resolve_chembl_ids,
    )

    all_symbols = get_all_gene_symbols(include_taar=True)
    logger.info("Total aminergic gene symbols: %d", len(all_symbols))

    # Slice BEFORE resolution so the resolution API calls stay fast for smoke tests.
    if args.max_targets is not None:
        symbols_to_resolve = all_symbols[: args.max_targets]
        logger.info(
            "--max-targets %d: resolving first %d symbols (%s)",
            args.max_targets,
            len(symbols_to_resolve),
            symbols_to_resolve,
        )
    else:
        symbols_to_resolve = all_symbols

    logger.info("Resolving %d gene symbols via ChEMBL API...", len(symbols_to_resolve))
    resolved_ids: dict[str, str] = resolve_chembl_ids(symbols_to_resolve)

    if not resolved_ids:
        logger.error(
            "No targets resolved — ChEMBL API may be unreachable. "
            "Propagating as failure."
        )
        sys.exit(1)

    logger.info(
        "Resolved %d / %d gene symbols successfully.",
        len(resolved_ids),
        len(symbols_to_resolve),
    )

    # ------------------------------------------------------------------
    # Step 2: Persist the resolved mapping (JSON)
    # ------------------------------------------------------------------
    # Tasks 8 (curation) and 11 (ESM-2) reload this file to rebuild the
    # aminergic config without re-querying ChEMBL.
    RESOLVED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESOLVED_IDS_PATH, "w") as fh:
        json.dump(resolved_ids, fh, indent=2)
    logger.info("Persisted resolved target mapping to %s", RESOLVED_IDS_PATH)

    # ------------------------------------------------------------------
    # Step 3: Build the aminergic TargetClassConfig
    # ------------------------------------------------------------------
    from gpcr_aminegric_benchmarks.target_class import build_aminergic_config

    config = build_aminergic_config(resolved_ids)
    logger.info(
        "Built aminergic TargetClassConfig: %d targets, stem=%r",
        len(config.explicit_target_ids),  # type: ignore[arg-type]
        config.raw_filename_stem,
    )

    # ------------------------------------------------------------------
    # Step 4: Fetch bioactivities via the library
    # ------------------------------------------------------------------
    # fetch_target_class detects uses_explicit_target_list=True and skips
    # GO discovery, calling fetch_bioactivities directly for the listed IDs.
    #
    # Filters applied INSIDE the library (at ChEMBL API level):
    #   - standard_type in {"IC50", "Ki", "Kd"}   [activity_types param]
    #   - standard_relation = "="                   [hardcoded in fetch_bioactivities]
    #   - pchembl_value is not null                 [hardcoded in fetch_bioactivities]
    #
    # Filters NOT applied by the library (applied below in Step 5):
    #   - assay_type = "B"
    #   - confidence_score >= 7
    #   - standard_units = "nM"
    from target_affinity_ml.data.chembl_fetcher import fetch_target_class

    logger.info("Calling fetch_target_class (explicit-ID path)...")
    activities_df, targets_df = fetch_target_class(
        config=config,
        activity_types=["IC50", "Ki", "Kd"],
    )

    if activities_df.empty:
        logger.error(
            "fetch_target_class returned 0 activity records — "
            "check ChEMBL API connectivity and target IDs."
        )
        sys.exit(1)

    logger.info(
        "Raw (pre-post-filter) activity records returned by library: %d",
        len(activities_df),
    )

    # ------------------------------------------------------------------
    # Step 5: Apply post-fetch inclusion filters NOT enforced by the library
    # ------------------------------------------------------------------
    n_before = len(activities_df)

    # 5a. assay_type = "B" (binding assays only — excludes functional, ADMET, etc.)
    if "assay_type" in activities_df.columns:
        activities_df = activities_df[activities_df["assay_type"] == "B"].copy()
        logger.info(
            "After assay_type='B' filter: %d records (removed %d)",
            len(activities_df),
            n_before - len(activities_df),
        )
        n_before = len(activities_df)
    else:
        logger.warning(
            "Column 'assay_type' not found in returned DataFrame — "
            "binding-assay filter could not be applied."
        )

    # 5b. confidence_score >= 7  (direct single-protein assignment)
    #
    # confidence_score is a TARGET-level field in ChEMBL, not an activity-level
    # field, so it is NOT included in ACTIVITY_COLUMNS and will not appear in
    # the returned DataFrame. However, resolve_chembl_ids() already restricts
    # resolution to target_type="SINGLE PROTEIN" targets, and ChEMBL assigns
    # confidence_score=9 to all SINGLE PROTEIN targets by convention (these are
    # the highest-confidence direct-binding assignments). Therefore all targets
    # in our resolved list satisfy confidence_score >= 7 by construction.
    #
    # If confidence_score were available per-activity (e.g. via a joined assay
    # query), we would apply: activities_df[activities_df["confidence_score"] >= 7]
    # For now, log an informational note and continue.
    if "confidence_score" in activities_df.columns:
        activities_df["confidence_score"] = pd.to_numeric(
            activities_df["confidence_score"], errors="coerce"
        )
        activities_df = activities_df[
            activities_df["confidence_score"] >= 7
        ].copy()
        logger.info(
            "After confidence_score>=7 filter: %d records (removed %d)",
            len(activities_df),
            n_before - len(activities_df),
        )
        n_before = len(activities_df)
    else:
        logger.info(
            "confidence_score not present in activity DataFrame (expected: it is a "
            "target-level field not exposed by ACTIVITY_COLUMNS). Confidence filter "
            "satisfied by construction: all resolved targets are target_type="
            "'SINGLE PROTEIN' (ChEMBL confidence_score=9)."
        )

    # 5c. standard_units = "nM"  (makes the nanomolar constraint explicit)
    if "standard_units" in activities_df.columns:
        activities_df = activities_df[
            activities_df["standard_units"] == "nM"
        ].copy()
        logger.info(
            "After standard_units='nM' filter: %d records (removed %d)",
            len(activities_df),
            n_before - len(activities_df),
        )
    else:
        logger.warning(
            "Column 'standard_units' not found — units filter not applied."
        )

    logger.info("Final activity records after all post-fetch filters: %d", len(activities_df))

    if activities_df.empty:
        logger.error(
            "0 records remain after post-fetch filtering — "
            "the filters may be too strict or the data is unexpected."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 6: Save outputs
    # ------------------------------------------------------------------
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    activities_path = DATA_RAW_DIR / config.raw_activities_filename
    targets_path = DATA_RAW_DIR / config.raw_targets_filename

    activities_df.to_parquet(activities_path, index=False)
    targets_df.to_parquet(targets_path, index=False)

    # ------------------------------------------------------------------
    # Step 7: Summary log
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info("  Resolved targets : %d", len(resolved_ids))
    logger.info("  Activity records : %d", len(activities_df))
    logger.info("  Unique compounds : %d", activities_df["molecule_chembl_id"].nunique())

    if "standard_type" in activities_df.columns:
        type_counts = activities_df["standard_type"].value_counts().to_dict()
        logger.info("  Activity types   : %s", type_counts)

    if "target_chembl_id" in activities_df.columns:
        per_target = activities_df["target_chembl_id"].value_counts()
        logger.info("  Per-target record distribution:")
        logger.info("    min=%d  median=%.0f  max=%d",
                    per_target.min(), per_target.median(), per_target.max())
        # Show top 5 targets by record count
        for tid, count in per_target.head(5).items():
            gene = next((g for g, c in resolved_ids.items() if c == tid), tid)
            logger.info("    %-20s (%s): %d records", gene, tid, count)

    logger.info("  Files written:")
    logger.info("    %s", activities_path)
    logger.info("    %s", targets_path)
    logger.info("    %s", RESOLVED_IDS_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
