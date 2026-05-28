#!/opt/homebrew/Caskroom/miniforge/base/envs/kinase-affinity/bin/python
"""Compute mean binding-site pLDDT for 543 targets (Plan 3 T14).

Per the P3-T6 metric pivot, this replaces the original 3-5 hour AWS RNS
pipeline with a local script: ~15-30 min wall-clock on first run, <1 minute
on cached re-runs.

For each target (36 aminergic GPCRs + 507 kinases), fetches the AlphaFold
structure and the binding-site residue list (GPCRdb for GPCRs, KLIFS for
kinases), then computes the mean pLDDT over those residues. Output drives
the H3 cross-class analysis in T19.

Usage
-----
    python scripts/compute_per_target_plddt.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from target_affinity_ml.benchmarks.rns_scoring import (
    fetch_structure,
    fetch_binding_site,
    compute_binding_site_plddt,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data/processed/v1/structure_cache"
OUTPUT_CSV = REPO_ROOT / "data/processed/v1/per_target_plddt.csv"
KINASE_MAPPING = REPO_ROOT / "data/kinase_reference/chembl_to_uniprot.json"

logger = logging.getLogger(__name__)


def _build_kinase_mapping() -> dict[str, str]:
    """Build or load the kinase ChEMBL->UniProt mapping (cached JSON).

    Uses UniProt's REST search to resolve canonical Swiss-Prot (reviewed=true)
    via gene_symbol + organism=human. The simpler ChEMBL-xref approach in
    ``target_affinity_ml.data.protein_sequences.fetch_uniprot_accessions``
    sometimes returns TrEMBL (unreviewed) accessions that AlphaFold doesn't
    serve, so we prefer the gene-symbol path here.
    """
    if KINASE_MAPPING.exists():
        with KINASE_MAPPING.open() as fh:
            mapping = json.load(fh)
        logger.info("Loaded cached kinase mapping (%d targets)", len(mapping))
        return mapping

    logger.info("Building kinase ChEMBL->UniProt mapping (this takes ~5-10 min)...")
    import time
    import requests

    kinase_parquet = REPO_ROOT / "data/kinase_reference/curated_activities.parquet"
    df = pd.read_parquet(kinase_parquet)
    chembl_to_gene = df.drop_duplicates("target_chembl_id").set_index("target_chembl_id")["gene_symbol"].to_dict()

    session = requests.Session()
    mapping: dict[str, str] = {}
    no_swissprot: list[str] = []
    for i, (chembl_id, gene) in enumerate(sorted(chembl_to_gene.items()), start=1):
        if not gene or pd.isna(gene):
            no_swissprot.append(chembl_id)
            continue
        url = (
            f"https://rest.uniprot.org/uniprotkb/search?"
            f"query=gene_exact:{gene}+AND+organism_id:9606+AND+reviewed:true"
            f"&format=json&size=1"
        )
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200 and r.json().get("results"):
                mapping[chembl_id] = r.json()["results"][0]["primaryAccession"]
            else:
                no_swissprot.append(chembl_id)
        except Exception:
            no_swissprot.append(chembl_id)
        if i % 50 == 0:
            logger.info("  [%d/%d] resolved=%d, missing=%d", i, len(chembl_to_gene), len(mapping), len(no_swissprot))
        time.sleep(0.15)

    if no_swissprot:
        logger.warning("%d kinase targets have no canonical Swiss-Prot mapping", len(no_swissprot))

    KINASE_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    with KINASE_MAPPING.open("w") as fh:
        json.dump(mapping, fh, indent=2, sort_keys=True)
    logger.info("Saved kinase mapping (%d targets) to %s", len(mapping), KINASE_MAPPING)
    return mapping


def _load_gpcr_mapping() -> dict[str, str]:
    """Load the GPCR ChEMBL->UniProt mapping from the Plan 2 artifact."""
    path = REPO_ROOT / "data/processed/v1/protein_sequences.json"
    with path.open() as fh:
        data = json.load(fh)
    return {chembl_id: entry["uniprot_id"] for chembl_id, entry in data.items()}


def _build_target_index() -> list[tuple[str, str, str]]:
    """Return [(chembl_id, uniprot_id, class_name), ...] for all 543 targets."""
    gpcr_map = _load_gpcr_mapping()
    kinase_map = _build_kinase_mapping()
    index: list[tuple[str, str, str]] = []
    for chembl_id, uniprot in sorted(gpcr_map.items()):
        index.append((chembl_id, uniprot, "gpcr_aminergic"))
    for chembl_id, uniprot in sorted(kinase_map.items()):
        index.append((chembl_id, uniprot, "kinase"))
    return index


def process_one_target(chembl_id: str, uniprot_id: str, class_name: str) -> dict:
    """Compute mean binding-site pLDDT for a single target. Never raises."""
    if not uniprot_id:
        return {
            "chembl_id": chembl_id,
            "class_name": class_name,
            "uniprot": uniprot_id or "",
            "structure_source": None,
            "n_binding_residues": 0,
            "mean_binding_site_plddt": float("nan"),
            "error": "no uniprot_id (mapping missing)",
        }

    try:
        structure, prov = fetch_structure(uniprot_id, CACHE_DIR, prefer="alphafold")
    except Exception as e:
        return {
            "chembl_id": chembl_id,
            "class_name": class_name,
            "uniprot": uniprot_id,
            "structure_source": None,
            "n_binding_residues": 0,
            "mean_binding_site_plddt": float("nan"),
            "error": f"fetch_structure failed: {e}",
        }

    try:
        binding_site = fetch_binding_site(
            chembl_id, class_name, CACHE_DIR, uniprot_id=uniprot_id
        )
    except Exception as e:
        return {
            "chembl_id": chembl_id,
            "class_name": class_name,
            "uniprot": uniprot_id,
            "structure_source": prov.get("source"),
            "n_binding_residues": 0,
            "mean_binding_site_plddt": float("nan"),
            "error": f"fetch_binding_site failed: {e}",
        }

    if not binding_site:
        return {
            "chembl_id": chembl_id,
            "class_name": class_name,
            "uniprot": uniprot_id,
            "structure_source": prov.get("source"),
            "n_binding_residues": 0,
            "mean_binding_site_plddt": float("nan"),
            "error": "no binding site (API returned empty list)",
        }

    try:
        plddt = compute_binding_site_plddt(structure, binding_site, prov)
    except Exception as e:
        return {
            "chembl_id": chembl_id,
            "class_name": class_name,
            "uniprot": uniprot_id,
            "structure_source": prov.get("source"),
            "n_binding_residues": len(binding_site),
            "mean_binding_site_plddt": float("nan"),
            "error": f"compute_binding_site_plddt failed: {e}",
        }

    return {
        "chembl_id": chembl_id,
        "class_name": class_name,
        "uniprot": uniprot_id,
        "structure_source": prov.get("source"),
        "n_binding_residues": len(binding_site),
        "mean_binding_site_plddt": plddt,
        "error": "",
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    targets = _build_target_index()
    n_gpcr = sum(1 for _, _, c in targets if c == "gpcr_aminergic")
    n_kinase = sum(1 for _, _, c in targets if c == "kinase")
    logger.info(
        "Processing %d targets (%d GPCR + %d kinase)",
        len(targets),
        n_gpcr,
        n_kinase,
    )

    rows = []
    for i, (chembl_id, uniprot, class_name) in enumerate(targets, start=1):
        row = process_one_target(chembl_id, uniprot, class_name)
        rows.append(row)
        if i % 25 == 0 or i == len(targets):
            plddt_val = row["mean_binding_site_plddt"]
            plddt_str = f"{plddt_val:.2f}" if not pd.isna(plddt_val) else "nan"
            logger.info(
                "  [%d/%d] %s (%s) -> pLDDT=%s",
                i,
                len(targets),
                chembl_id,
                class_name,
                plddt_str,
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    success = int((~df["mean_binding_site_plddt"].isna()).sum())
    logger.info("DONE. Wrote %d rows to %s", len(df), OUTPUT_CSV)
    logger.info("  Successful targets: %d", success)
    logger.info("  NaN (missing binding site / structure / etc.): %d", len(df) - success)
    print()
    print(df.groupby("class_name")["mean_binding_site_plddt"].describe())


if __name__ == "__main__":
    main()
