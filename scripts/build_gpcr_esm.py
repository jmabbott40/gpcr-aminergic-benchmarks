#!/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
"""Fetch protein sequences and compute ESM-2 embeddings for aminergic GPCR targets.

Usage
-----
    python scripts/build_gpcr_esm.py
    python scripts/build_gpcr_esm.py --max-targets 1   # smoke test
    python scripts/build_gpcr_esm.py --force            # re-fetch + re-embed even if cached
    python scripts/build_gpcr_esm.py --resume           # skip fetch if sequences already on disk

This script has two sequential stages:

Stage 1 — Sequence fetching
    Reads data/processed/v1/resolved_target_ids.json (gene_symbol → ChEMBL ID),
    queries ChEMBL for UniProt accessions, then fetches amino acid sequences from
    the UniProt REST API.  Output:
        data/processed/v1/protein_sequences.json

Stage 2 — ESM-2 embeddings
    Loads the cached sequences and runs Meta's ESM-2 (esm2_t33_650M_UR50D) as a
    frozen feature extractor.  Each target gets a single 1280-dim float32 vector
    formed by mean-pooling over residue token representations.  Sequences longer
    than 1022 residues are truncated (ESM-2 hard limit).  Outputs:
        data/processed/v1/features/esm2_embeddings.npz   (shape: N_targets × 1280)
        data/processed/v1/features/target_index.json     (ChEMBL ID → row index)

GPU detection
    The script logs CUDA availability and GPU count at startup.  If no GPU is
    detected it emits a loud WARNING — embedding on CPU is ~50-100× slower but
    the script does NOT abort.  For a laptop smoke test on 1-2 targets, CPU is
    acceptable.  On AWS (A10G / A100) ensure CUDA_VISIBLE_DEVICES is set and the
    correct torch/fair-esm packages are installed in the environment.

Inputs
------
    data/processed/v1/resolved_target_ids.json   — 36 aminergic GPCR targets

Outputs
-------
    data/processed/v1/protein_sequences.json
    data/processed/v1/features/esm2_embeddings.npz
    data/processed/v1/features/target_index.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Resolve paths relative to this file so the script is cwd-agnostic ────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("build_gpcr_esm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--max-targets",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N targets (useful for smoke-testing on a laptop).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        metavar="DIR",
        help=(
            "Root of the processed data directory (default: %(default)s). "
            "Version subdir 'v1/' is always appended."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch sequences and re-compute embeddings even if outputs already exist.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If protein_sequences.json already exists, skip Stage 1 and jump "
            "straight to embedding.  Useful when the network fetch succeeded but "
            "the GPU job was interrupted."
        ),
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_resolved_ids(data_dir: Path) -> dict[str, str]:
    """Return {gene_symbol: chembl_id} from resolved_target_ids.json."""
    resolved_path = data_dir / "resolved_target_ids.json"
    if not resolved_path.exists():
        logger.error("resolved_target_ids.json not found: %s", resolved_path)
        sys.exit(1)
    with open(resolved_path) as fh:
        resolved: dict[str, str] = json.load(fh)
    logger.info("Loaded %d resolved target IDs from %s", len(resolved), resolved_path)
    return resolved


def _fetch_all_uniprot_accessions(
    target_chembl_ids: list[str],
) -> dict[str, list[str]]:
    """Return ALL UniProt accessions per ChEMBL target (not just the first one).

    The library's fetch_uniprot_accessions picks one accession per target using
    a 6-char heuristic (Swiss-Prot vs TrEMBL) which can select a reviewed-looking
    accession that nonetheless returns an empty sequence in the JSON batch endpoint
    (e.g. CHEMBL2056 → B2RA44 vs P21728).  We collect all accessions so the
    script can try them in order and pick the first with a non-empty sequence.
    """
    from chembl_webresource_client.new_client import new_client
    import time

    target_api = new_client.target
    target_to_accessions: dict[str, list[str]] = {}

    for i, tid in enumerate(target_chembl_ids):
        try:
            target = target_api.get(tid)
            if target is None:
                logger.warning("ChEMBL returned nothing for %s", tid)
                continue
            accessions: list[str] = []
            for comp in target.get("target_components", []):
                for xref in comp.get("target_component_xrefs", []):
                    if xref.get("xref_src_db") == "UniProt":
                        accessions.append(xref["xref_id"])
            if accessions:
                # Deduplicate while preserving order
                seen: set[str] = set()
                unique_acc: list[str] = []
                for acc in accessions:
                    if acc not in seen:
                        seen.add(acc)
                        unique_acc.append(acc)
                # Put 6-char (Swiss-Prot) candidates first as a hint,
                # but keep all so the fallback loop can try them.
                unique_acc.sort(key=lambda a: (0 if len(a) == 6 else 1))
                target_to_accessions[tid] = unique_acc
                logger.debug("  %s → %s", tid, unique_acc)
            else:
                logger.warning("No UniProt xrefs for %s", tid)
            if (i + 1) % 20 == 0:
                time.sleep(0.5)
        except Exception as exc:
            logger.warning("Error querying ChEMBL for %s: %s", tid, exc)

    logger.info(
        "UniProt accession lists fetched: %d/%d targets",
        len(target_to_accessions), len(target_chembl_ids),
    )
    return target_to_accessions


def _fetch_sequence_fasta(uniprot_id: str, timeout: int = 20) -> str:
    """Fetch a single protein sequence via the UniProt FASTA endpoint.

    Returns the amino acid sequence string, or '' on failure.
    """
    import requests

    try:
        resp = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta",
            timeout=timeout,
        )
        if resp.ok and resp.text.strip():
            lines = resp.text.strip().split("\n")
            return "".join(lines[1:])  # Skip FASTA header line
    except Exception as exc:
        logger.debug("FASTA fetch failed for %s: %s", uniprot_id, exc)
    return ""


def _build_sequence_cache(
    chembl_ids: list[str],
    gene_symbol_map: dict[str, str],
    seq_path: Path,
) -> dict:
    """Fetch UniProt accessions via ChEMBL then sequences from UniProt REST.

    Returns the per-target sequence cache dict and saves it to *seq_path*.
    The cache schema mirrors build_protein_sequence_cache from the kinase library
    but is built from a pre-resolved chembl_id list rather than a parquet file:

        {
          "CHEMBL210": {
            "uniprot_id": "P07550",
            "gene_symbol": "ADRB2",
            "pref_name": "",
            "sequence": "MGQPGNGSAAF...",
            "length": 413
          },
          ...
        }

    Note on robustness: the library's batch JSON search endpoint can return an
    empty sequence object for some accessions (e.g. early TrEMBL entries merged
    into Swiss-Prot).  This function falls back to per-accession FASTA fetches
    and tries all accessions listed for a target until one yields a sequence.
    """
    import time

    # Invert gene_symbol_map so we can tag cache entries with gene symbols
    # gene_symbol_map: {gene_symbol: chembl_id}  →  {chembl_id: gene_symbol}
    chembl_to_gene: dict[str, str] = {v: k for k, v in gene_symbol_map.items()}

    # Step 1a — ChEMBL → ALL UniProt accessions (ordered, 6-char first)
    logger.info("Stage 1a: Fetching UniProt accessions from ChEMBL (%d targets)...", len(chembl_ids))
    target_to_accessions: dict[str, list[str]] = _fetch_all_uniprot_accessions(chembl_ids)

    # Step 1b — For each target, try accessions in order until we get a sequence
    logger.info("Stage 1b: Fetching sequences from UniProt (FASTA endpoint, with fallback)...")
    cache: dict = {}
    total = len(target_to_accessions)

    for idx, (tid, accessions) in enumerate(target_to_accessions.items()):
        gene = chembl_to_gene.get(tid, "?")
        seq = ""
        chosen_uid = ""
        for uid in accessions:
            seq = _fetch_sequence_fasta(uid)
            if seq:
                chosen_uid = uid
                break
            logger.debug("  %s: empty sequence for %s, trying next...", gene, uid)
            time.sleep(0.2)

        if seq:
            cache[tid] = {
                "uniprot_id": chosen_uid,
                "gene_symbol": gene,
                "pref_name": "",
                "sequence": seq,
                "length": len(seq),
            }
            logger.info(
                "  [%d/%d] %s (%s): UniProt=%s  len=%d",
                idx + 1, total, gene, tid, chosen_uid, len(seq),
            )
        else:
            logger.warning(
                "  [%d/%d] %s (%s): no sequence from any of %s — skipping",
                idx + 1, total, gene, tid, accessions,
            )
        # Light rate-limit between targets
        time.sleep(0.3)

    logger.info("Sequence cache assembled: %d/%d targets have sequences",
                len(cache), len(chembl_ids))

    if not cache:
        logger.error("No sequences fetched — cannot proceed to embedding stage.")
        sys.exit(1)

    # Save between stages so a partial run is recoverable
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    with open(seq_path, "w") as fh:
        json.dump(cache, fh, indent=2)
    logger.info("Saved protein_sequences.json to %s", seq_path)

    lengths = [v["length"] for v in cache.values()]
    logger.info(
        "Sequence lengths — min=%d  max=%d  mean=%.0f  median=%.0f",
        min(lengths), max(lengths),
        sum(lengths) / len(lengths),
        sorted(lengths)[len(lengths) // 2],
    )

    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_embeddings(
    seq_cache: dict,
    data_dir: Path,
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 4,
    device: str | None = None,
) -> None:
    """Compute ESM-2 mean-pooled embeddings and save npz + index.

    Calls target_affinity_ml.features.protein_embeddings.compute_esm2_embeddings
    which reads protein_sequences.json directly from disk.  This function ensures
    the on-disk file is present (it was written by Stage 1) before delegating.

    Parameters
    ----------
    seq_cache : dict
        The in-memory cache (used only to log counts; the library re-reads from disk).
    data_dir : Path
        Root processed data directory (e.g. data/processed).  The library appends
        'v1/' internally via its ``dataset_version`` parameter.
    model_name : str
        ESM-2 variant.  Must match a key in ESM2_MODELS inside protein_embeddings.py.
    batch_size : int
        Sequences per GPU batch.  Reduce to 1-2 on small GPUs or CPU.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """
    from target_affinity_ml.features.protein_embeddings import compute_esm2_embeddings

    logger.info("Stage 2: Computing ESM-2 embeddings for %d targets...", len(seq_cache))
    logger.info("  model      : %s", model_name)
    logger.info("  batch_size : %d", batch_size)
    logger.info("  device     : %s", device if device else "auto")

    # compute_esm2_embeddings uses DATA_DIR / dataset_version internally.
    # Its DATA_DIR is hard-coded to Path("data/processed") in the library.
    # We work around this by checking that our data_dir resolves to the same
    # location that the library expects, and warn if not.
    import target_affinity_ml.features.protein_embeddings as _emb_mod
    lib_data_dir: Path = _emb_mod.DATA_DIR.resolve() if _emb_mod.DATA_DIR.is_absolute() \
        else (REPO_ROOT / _emb_mod.DATA_DIR).resolve()

    if data_dir.resolve() != lib_data_dir:
        logger.warning(
            "Library DATA_DIR (%s) differs from --data-dir (%s). "
            "The library will write outputs to its own DATA_DIR/v1/features/. "
            "This is expected when --data-dir was customised.",
            lib_data_dir, data_dir.resolve(),
        )

    embeddings, target_to_row = compute_esm2_embeddings(
        dataset_version="v1",
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )

    logger.info(
        "[PASS] Embeddings: shape=%s  dtype=%s",
        embeddings.shape, embeddings.dtype,
    )
    logger.info("       target_index entries: %d", len(target_to_row))


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── GPU detection ─────────────────────────────────────────────────────────
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        n_gpus = torch.cuda.device_count() if cuda_available else 0
        if cuda_available:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
            logger.info("CUDA available: YES — %d GPU(s): %s", n_gpus, gpu_names)
        else:
            logger.warning(
                "CUDA is NOT available. ESM-2 embedding will run on CPU, "
                "which is ~50-100x slower than a GPU. For a full run (36 targets) "
                "strongly prefer an AWS GPU instance (A10G / A100). "
                "Proceeding anyway — for a 1-target smoke test this is fine."
            )
    except ImportError:
        logger.warning(
            "torch is not importable in this environment. "
            "The sequence-fetch stage will still run, but embedding will be skipped."
        )
        torch = None  # type: ignore[assignment]

    data_dir = args.data_dir.resolve()
    v1_dir = data_dir / "v1"

    logger.info("=" * 60)
    logger.info("GPCR Protein Sequence + ESM-2 Embedding Pipeline")
    logger.info("  repo root  : %s", REPO_ROOT)
    logger.info("  data dir   : %s", data_dir)
    logger.info("  max targets: %s", args.max_targets if args.max_targets else "all (36)")
    logger.info("  force      : %s", args.force)
    logger.info("  resume     : %s", args.resume)
    logger.info("=" * 60)

    # ── Paths ─────────────────────────────────────────────────────────────────
    seq_path = v1_dir / "protein_sequences.json"
    features_dir = v1_dir / "features"
    emb_path = features_dir / "esm2_embeddings.npz"
    idx_path = features_dir / "target_index.json"

    # ── Stage 1: Protein sequence fetching ───────────────────────────────────
    seq_exists = seq_path.exists()

    if seq_exists and not args.force and args.resume:
        logger.info("=== Stage 1: SKIPPED (--resume and sequences already on disk) ===")
        logger.info("Loading existing sequences from %s", seq_path)
        with open(seq_path) as fh:
            seq_cache: dict = json.load(fh)
        logger.info("Loaded %d sequences from cache", len(seq_cache))

    elif seq_exists and not args.force:
        logger.info(
            "=== Stage 1: SKIPPED (sequences already on disk; use --force to re-fetch) ==="
        )
        with open(seq_path) as fh:
            seq_cache = json.load(fh)
        logger.info("Loaded %d sequences from cache", len(seq_cache))

    else:
        logger.info("=== Stage 1: Fetching protein sequences ===")

        # Load resolved target IDs (gene_symbol → chembl_id)
        resolved_ids = _load_resolved_ids(v1_dir)

        # Optionally limit for smoke testing
        if args.max_targets is not None:
            resolved_ids = dict(list(resolved_ids.items())[:args.max_targets])
            logger.info("Limiting to %d target(s) for smoke test", args.max_targets)

        chembl_ids = list(resolved_ids.values())
        seq_cache = _build_sequence_cache(chembl_ids, resolved_ids, seq_path)

    # Filter seq_cache to match max_targets if we loaded from disk but user
    # specified --max-targets (e.g. for a partial embedding smoke test)
    if args.max_targets is not None and len(seq_cache) > args.max_targets:
        # Keep the first N by insertion order
        limited_keys = list(seq_cache.keys())[: args.max_targets]
        seq_cache = {k: seq_cache[k] for k in limited_keys}
        logger.info("Limiting in-memory cache to %d target(s) for embedding step", len(seq_cache))

    # ── Stage 2: ESM-2 embeddings ─────────────────────────────────────────────
    emb_exists = emb_path.exists() and idx_path.exists()

    if emb_exists and not args.force:
        logger.info(
            "=== Stage 2: SKIPPED (embeddings already on disk; use --force to re-embed) ==="
        )
        import numpy as np
        emb = np.load(emb_path)["embeddings"]
        with open(idx_path) as fh:
            target_to_row = json.load(fh)
        logger.info(
            "Loaded existing embeddings: shape=%s, index entries=%d",
            emb.shape, len(target_to_row),
        )
    else:
        logger.info("=== Stage 2: Computing ESM-2 embeddings ===")

        if torch is None:
            logger.warning(
                "torch not available — SKIPPING embedding stage. "
                "Sequence-fetch stage completed successfully."
            )
        else:
            # When --max-targets is set we wrote a full (or cached) seq_cache but
            # we need to temporarily slim the on-disk file so the library only
            # embeds the targets we want.  We restore it after.
            import tempfile, shutil

            if args.max_targets is not None and seq_path.exists():
                # Overwrite seq_path with the limited cache for the duration of
                # the library call, then restore the original.
                backup = seq_path.with_suffix(".json.bak")
                shutil.copy2(seq_path, backup)
                with open(seq_path, "w") as fh:
                    json.dump(seq_cache, fh, indent=2)
                try:
                    _compute_embeddings(
                        seq_cache=seq_cache,
                        data_dir=data_dir,
                    )
                finally:
                    # Restore original (or limited) sequences file
                    shutil.move(str(backup), str(seq_path))
            else:
                _compute_embeddings(
                    seq_cache=seq_cache,
                    data_dir=data_dir,
                )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  protein_sequences.json : %s", seq_path)
    logger.info("  esm2_embeddings.npz    : %s", emb_path)
    logger.info("  target_index.json      : %s", idx_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
