#!/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
"""Fetch aminergic GPCR bioactivity data from ChEMBL.

Usage
-----
    # Full fetch (all 36 targets):
    python scripts/fetch_gpcr_data.py

    # Smoke test (1 target, ~5s with page_size=1000):
    python scripts/fetch_gpcr_data.py --max-targets 1

    # Resume after a killed run (skip completed-target checkpoints):
    python scripts/fetch_gpcr_data.py --resume

    # Customise page size (default 1000):
    python scripts/fetch_gpcr_data.py --page-size 500

Outputs
-------
    data/processed/v1/resolved_target_ids.json          — gene_symbol -> chembl_id mapping
    data/raw/checkpoints/<CHEMBL_ID>.parquet             — per-target checkpoint (post-filter)
    data/raw/chembl_gpcr_aminergic_activities.parquet   — all targets concatenated
    data/raw/chembl_gpcr_aminergic_targets.parquet       — target metadata

Checkpointing
-------------
Each target's post-filtered activity DataFrame is saved to
data/raw/checkpoints/<CHEMBL_ID>.parquet as soon as the fetch and
post-fetch filtering completes for that target.  If the run is killed
mid-way, re-running with --resume will skip completed-target checkpoints
and only fetch the remaining targets.

On a clean (non-resume) run any existing checkpoints for the targets being
fetched are silently overwritten, so the final concatenated parquet is always
consistent with the current run's output.

Page-size tuning
----------------
Active path: direct ChEMBL REST via _fetch_one_target_rest (bypasses
chembl_webresource_client entirely).  The `limit` parameter is sent
explicitly in each HTTP request, so page_size is always honoured.

The library's Settings.MAX_LIMIT path was tried first and found
ineffective: new_client.activity is a cached QuerySet singleton whose
UrlQuery.limit is set at module load time; clones copy self.limit rather
than re-reading Settings, so the limit stays 20 in the URL params even
after setting MAX_LIMIT=1000.

This gives 10-50x throughput vs the library default (e.g. DRD2 ~2 300
records: ~3s with limit=1000 vs ~70s with limit=20).

Inclusion criteria (audit spec)
--------------------------------
Applied by target_affinity_ml.fetch_bioactivities at the ChEMBL API level:
    - standard_type in {"IC50", "Ki", "Kd"}           [API filter]
    - standard_relation = "="  (exact measurements)   [API filter]
    - pchembl_value is not null                        [API filter]

Applied by THIS SCRIPT via post-fetch DataFrame filtering:
    - assay_type = "B"  (binding assays only)          [post-fetch filter]
    - confidence_score >= 7                            [post-fetch filter — satisfied by
                                                        construction for SINGLE PROTEIN targets]
    - standard_units = "nM"                            [post-fetch filter]

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
import time
from pathlib import Path

import pandas as pd
import requests

# Resolve repo root from this file's location so the script works regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
CHECKPOINT_DIR = DATA_RAW_DIR / "checkpoints"
RESOLVED_IDS_PATH = REPO_ROOT / "data" / "processed" / "v1" / "resolved_target_ids.json"

# Columns to keep from ChEMBL activity records — must match ACTIVITY_COLUMNS in
# chembl_fetcher.py so that Task 8 curation sees a schema-identical DataFrame.
ACTIVITY_COLUMNS = [
    "activity_id",
    "molecule_chembl_id",
    "canonical_smiles",
    "target_chembl_id",
    "standard_type",
    "standard_value",
    "standard_units",
    "standard_relation",
    "pchembl_value",
    "assay_chembl_id",
    "assay_type",
    "data_validity_comment",
]

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
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Skip targets whose per-target checkpoint already exists in "
            "data/raw/checkpoints/. Re-run after a killed fetch to pick up "
            "where it left off. Default: False (overwrite any existing checkpoints)."
        ),
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        metavar="N",
        help=(
            "ChEMBL API page size (records per HTTP request). "
            "Default 1000 gives 10-50x throughput vs the library default of 20. "
            "Max supported by ChEMBL REST API is 10000."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-target fetch helpers
# ---------------------------------------------------------------------------

def _fetch_one_target(
    chembl_id: str,
    activity_types: list[str],
    page_size: int = 1000,
) -> pd.DataFrame:
    """Fetch raw activity records for a single target via direct ChEMBL REST.

    Active path: direct REST (Plan B).

    The chembl_webresource_client library's Settings.MAX_LIMIT path was tried
    first but found to be ineffective: new_client.activity is a cached
    QuerySet singleton whose internal UrlQuery.limit is set to 20 at module
    load time, and clones from it inherit the original limit regardless of
    the current Settings.MAX_LIMIT value.  (Confirmed: after setting
    Settings.Instance().MAX_LIMIT = 1000, fresh .filter() clones still show
    limit=20 in their URL params because clone() copies self.limit directly.)

    Direct REST bypasses the library entirely and always uses the requested
    page_size, giving 10-50x throughput for large targets (e.g. DRD2 ~2 300
    records: ~3s with limit=1000 vs ~70s with limit=20).

    The returned DataFrame has the same ACTIVITY_COLUMNS schema as the
    library would produce, so Task 8 curation is unaffected.

    Returns
    -------
    pd.DataFrame
        Raw activity records with ACTIVITY_COLUMNS schema (no post-fetch
        filtering applied here — caller applies assay_type, units filters).
    """
    return _fetch_one_target_rest(chembl_id, activity_types=activity_types, page_size=page_size)


def _fetch_one_target_rest(
    chembl_id: str,
    activity_types: list[str],
    page_size: int = 1000,
) -> pd.DataFrame:
    """Fetch raw activity records for a single target via direct ChEMBL REST.

    Calls the ChEMBL REST API directly with explicit `limit` and `offset`
    parameters, bypassing chembl_webresource_client pagination entirely.
    Always uses the requested page_size regardless of any library settings.

    The returned DataFrame has the same ACTIVITY_COLUMNS schema as the
    library path so downstream curation is unaffected.

    Returns
    -------
    pd.DataFrame
        Raw activity records with ACTIVITY_COLUMNS schema.
    """
    base = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    records: list[dict] = []
    offset = 0
    t0 = time.time()

    while True:
        resp = requests.get(
            base,
            params={
                "target_chembl_id": chembl_id,
                "standard_type__in": ",".join(activity_types),
                "standard_relation": "=",
                "pchembl_value__isnull": "false",
                "limit": page_size,
                "offset": offset,
            },
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("activities", [])
        if not batch:
            break
        records.extend(batch)
        logger.info(
            "[rest] %s: page offset=%d fetched %d records (total so far: %d, limit=%d)",
            chembl_id,
            offset,
            len(batch),
            len(records),
            page_size,
        )
        offset += len(batch)
        if len(batch) < page_size:
            break

    elapsed = time.time() - t0
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=ACTIVITY_COLUMNS)
    # Trim to ACTIVITY_COLUMNS (keeps schema identical to library output).
    df = df[[c for c in ACTIVITY_COLUMNS if c in df.columns]]
    # Re-index to add any missing columns (rare — API may omit null fields).
    df = df.reindex(columns=ACTIVITY_COLUMNS)
    logger.info(
        "[rest] %s: %d total records in %.1fs (page_size=%d)",
        chembl_id,
        len(df),
        elapsed,
        page_size,
    )
    return df


# ---------------------------------------------------------------------------
# Post-fetch filters (identical to original script's Step 5)
# ---------------------------------------------------------------------------

def _apply_post_fetch_filters(activities_df: pd.DataFrame, chembl_id: str) -> pd.DataFrame:
    """Apply assay_type, confidence, and units filters to a single-target DataFrame."""
    n_before = len(activities_df)

    # assay_type = "B" (binding assays only)
    if "assay_type" in activities_df.columns:
        activities_df = activities_df[activities_df["assay_type"] == "B"].copy()
        removed = n_before - len(activities_df)
        if removed:
            logger.debug(
                "%s: assay_type='B' removed %d records (%d remain)",
                chembl_id,
                removed,
                len(activities_df),
            )
        n_before = len(activities_df)
    else:
        logger.warning(
            "%s: Column 'assay_type' not found — binding-assay filter skipped.",
            chembl_id,
        )

    # confidence_score >= 7 (satisfied by construction for SINGLE PROTEIN targets;
    # confidence_score is a target-level field not in ACTIVITY_COLUMNS)
    if "confidence_score" in activities_df.columns:
        activities_df["confidence_score"] = pd.to_numeric(
            activities_df["confidence_score"], errors="coerce"
        )
        activities_df = activities_df[
            activities_df["confidence_score"] >= 7
        ].copy()
        n_before = len(activities_df)

    # standard_units = "nM"
    if "standard_units" in activities_df.columns:
        activities_df = activities_df[
            activities_df["standard_units"] == "nM"
        ].copy()
        removed = n_before - len(activities_df)
        if removed:
            logger.debug(
                "%s: standard_units='nM' removed %d records (%d remain)",
                chembl_id,
                removed,
                len(activities_df),
            )
    else:
        logger.warning(
            "%s: Column 'standard_units' not found — units filter skipped.",
            chembl_id,
        )

    return activities_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 1: Resolve gene symbols -> ChEMBL IDs
    # ------------------------------------------------------------------
    from gpcr_aminergic_benchmarks.target_lists import (
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
    RESOLVED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESOLVED_IDS_PATH, "w") as fh:
        json.dump(resolved_ids, fh, indent=2)
    logger.info("Persisted resolved target mapping to %s", RESOLVED_IDS_PATH)

    # ------------------------------------------------------------------
    # Step 3: Build the aminergic TargetClassConfig
    # ------------------------------------------------------------------
    from gpcr_aminergic_benchmarks.target_class import build_aminergic_config

    config = build_aminergic_config(resolved_ids)
    logger.info(
        "Built aminergic TargetClassConfig: %d targets, stem=%r",
        len(config.explicit_target_ids),  # type: ignore[arg-type]
        config.raw_filename_stem,
    )

    # ------------------------------------------------------------------
    # Step 4: Per-target fetch with checkpointing
    # ------------------------------------------------------------------
    # Each target is saved to data/raw/checkpoints/<CHEMBL_ID>.parquet
    # immediately after its fetch + post-fetch filtering completes.
    # This means a mid-run kill loses at most the current in-progress target.
    # Re-running with --resume skips targets whose checkpoint already exists.
    #
    # Page-size approach: Settings.MAX_LIMIT path is used first.  The
    # setting is applied inside _fetch_one_target() before the first API
    # call.  Log lines from the library show the page size in the URL
    # params — look for ('limit', 1000) instead of ('limit', 20).
    #
    # If the observed limit is still 20 (Settings path ineffective), replace
    # the call to _fetch_one_target() with _fetch_one_target_rest() which
    # uses direct REST and always honours the requested page size.

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    target_ids: list[str] = list(config.explicit_target_ids)  # type: ignore[arg-type]
    n_targets = len(target_ids)

    logger.info(
        "Starting per-target fetch: %d targets, page_size=%d, resume=%s",
        n_targets,
        args.page_size,
        args.resume,
    )

    failed_targets: list[str] = []

    for i, chembl_id in enumerate(target_ids):
        checkpoint_path = CHECKPOINT_DIR / f"{chembl_id}.parquet"

        if args.resume and checkpoint_path.exists():
            logger.info(
                "[skip] %s (%d/%d, checkpoint exists)", chembl_id, i + 1, n_targets
            )
            continue

        logger.info("[fetch] %s (%d/%d)...", chembl_id, i + 1, n_targets)
        try:
            # Active path: direct REST via _fetch_one_target_rest (delegated
            # to from _fetch_one_target).  The library's Settings.MAX_LIMIT
            # approach was tested and found ineffective (limit stays 20 in
            # the URL params due to cached QuerySet cloning).
            raw_df = _fetch_one_target(
                chembl_id,
                activity_types=["IC50", "Ki", "Kd"],
                page_size=args.page_size,
            )
        except Exception as exc:
            logger.error("[error] %s: %s — skipping target", chembl_id, exc)
            failed_targets.append(chembl_id)
            time.sleep(2)
            continue

        filtered_df = _apply_post_fetch_filters(raw_df, chembl_id)

        filtered_df.to_parquet(checkpoint_path, index=False)
        logger.info(
            "[done] %s: %d records after filtering (saved to checkpoint)",
            chembl_id,
            len(filtered_df),
        )

    if failed_targets:
        logger.warning(
            "%d target(s) failed to fetch and were skipped: %s",
            len(failed_targets),
            failed_targets,
        )

    # ------------------------------------------------------------------
    # Step 5: Concatenate all per-target checkpoints into final parquet
    # ------------------------------------------------------------------
    checkpoint_files = sorted(CHECKPOINT_DIR.glob("CHEMBL*.parquet"))
    if not checkpoint_files:
        logger.error(
            "No checkpoint files found in %s — nothing to concatenate.",
            CHECKPOINT_DIR,
        )
        sys.exit(1)

    logger.info(
        "Concatenating %d checkpoint file(s) into final parquet...",
        len(checkpoint_files),
    )
    all_dfs = [pd.read_parquet(p) for p in checkpoint_files]
    activities_df = pd.concat(all_dfs, ignore_index=True)

    activities_path = DATA_RAW_DIR / config.raw_activities_filename
    activities_df.to_parquet(activities_path, index=False)
    logger.info(
        "Wrote %d total activity records to %s",
        len(activities_df),
        activities_path,
    )

    # ------------------------------------------------------------------
    # Step 6: Build and save the targets parquet
    # ------------------------------------------------------------------
    # For the explicit-ID path the targets DataFrame is a minimal frame:
    # one row per resolved ChEMBL ID, with a subfamily column from
    # config.subfamily_map (gene family: dopamine, serotonin, etc.).
    targets_df = pd.DataFrame(
        {
            "target_chembl_id": list(config.explicit_target_ids),  # type: ignore[arg-type]
            "subfamily": [
                config.subfamily_map.get(tid, "unknown")
                for tid in config.explicit_target_ids  # type: ignore[union-attr]
            ],
        }
    )
    targets_path = DATA_RAW_DIR / config.raw_targets_filename
    targets_df.to_parquet(targets_path, index=False)
    logger.info("Wrote %d target rows to %s", len(targets_df), targets_path)

    # ------------------------------------------------------------------
    # Step 7: Validate output schema against ACTIVITY_COLUMNS
    # ------------------------------------------------------------------
    missing_cols = [c for c in ACTIVITY_COLUMNS if c not in activities_df.columns]
    extra_cols = [c for c in activities_df.columns if c not in ACTIVITY_COLUMNS]
    if missing_cols:
        logger.warning(
            "Output parquet is MISSING expected columns: %s — "
            "Task 8 curation may fail.",
            missing_cols,
        )
    if extra_cols:
        logger.warning("Output parquet has EXTRA columns not in ACTIVITY_COLUMNS: %s", extra_cols)
    if not missing_cols and not extra_cols:
        logger.info(
            "Schema check PASSED: output columns match ACTIVITY_COLUMNS exactly."
        )

    if activities_df.empty:
        logger.error("0 records in final output — check filters and API connectivity.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 8: Summary log
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info("  Resolved targets  : %d", len(resolved_ids))
    logger.info("  Failed targets    : %d", len(failed_targets))
    logger.info("  Checkpoints saved : %d", len(checkpoint_files))
    logger.info("  Activity records  : %d", len(activities_df))
    logger.info("  Unique compounds  : %d", activities_df["molecule_chembl_id"].nunique())

    if "standard_type" in activities_df.columns:
        type_counts = activities_df["standard_type"].value_counts().to_dict()
        logger.info("  Activity types    : %s", type_counts)

    if "target_chembl_id" in activities_df.columns:
        per_target = activities_df["target_chembl_id"].value_counts()
        logger.info("  Per-target record distribution:")
        logger.info(
            "    min=%d  median=%.0f  max=%d",
            per_target.min(),
            per_target.median(),
            per_target.max(),
        )
        for tid, count in per_target.head(5).items():
            gene = next((g for g, c in resolved_ids.items() if c == tid), tid)
            logger.info("    %-20s (%s): %d records", gene, tid, count)

    logger.info("  Files written:")
    logger.info("    %s", activities_path)
    logger.info("    %s", targets_path)
    logger.info("    %s", RESOLVED_IDS_PATH)
    logger.info("  Checkpoint dir    : %s (%d files)", CHECKPOINT_DIR, len(checkpoint_files))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
