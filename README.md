# gpcr-aminergic-benchmarks

## Purpose

This repository is the **Plan 2 application repo** for the GPCR side of a cross-class comparison: *when do complex ML models outperform simple cheminformatics baselines for protein–ligand binding affinity prediction?*

Specifically it targets **Class A aminergic GPCRs** — dopamine, serotonin, adrenergic, histamine, muscarinic, and trace-amine (TAAR1) receptors — fetched from ChEMBL, curated, and benchmarked with the same model suite as kinases.

## Dependency

All class-agnostic data-pipeline and modelling code lives in the shared library:

- **target-affinity-ml v1.1.0** — <https://github.com/jmabbott40/target-affinity-ml>

This application repo pins that library and adds:

- The aminergic GPCR target list (gene symbols, receptor families)
- A `build_aminergic_config()` builder that constructs the `TargetClassConfig` from a resolved gene → ChEMBL ID map
- Data-ingestion, curation, feature, and benchmark scripts (subsequent commits)

## Sister repo

Kinase application repo (Plan 1): <https://github.com/jmabbott40/kinase-affinity-baselines>

## Cross-class data

As of Plan 3, this repo hosts **both** the GPCR primary data AND the kinase
reference data needed for the cross-class comparison analysis. The kinase
data lives in `data/kinase_reference/` (Morgan FP + ESM-2 embeddings,
curated activities, splits, benchmark CSVs, and per-target metrics — the
RDKit descriptors file is local-only pending Zenodo deposit due to GitHub
file-size limits; see `data/kinase_reference/README.md` for full provenance).
This resolves Plan 1 limitation L2 (kinase reference data previously
local-only). Plan 3's cross-class analysis scripts and notebooks read from
both `data/processed/v1/` (GPCR side) and `data/kinase_reference/` (kinase
side) so the comparison is fully reproducible from this single repo.

## Installation

```bash
git clone https://github.com/jmabbott40/gpcr-aminergic-benchmarks.git
cd gpcr-aminergic-benchmarks
pip install -e ".[dev]"
```

For deep-learning extras (GNN, ESM-2):

```bash
pip install -e ".[deep]"
```

## Repository layout

```
src/gpcr_aminergic_benchmarks/   # Python package
    __init__.py
    target_lists.py              # aminergic receptor gene symbols + families
    target_class.py              # build_aminergic_config() builder
scripts/
    fetch_gpcr_data.py           # ChEMBL data ingestion (per-target checkpointed)
tests/
```

Data ingestion, curation, splits, features, and benchmark scripts land in subsequent commits (Tasks 7–13).
