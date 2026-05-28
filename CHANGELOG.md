# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] – 2026-05-28

Plan 3 cross-class analysis complete; analysis frozen for the Plan 4 manuscript.

### Added

- **Kinase reference data** under `data/kinase_reference/` (resolves Plan 1 limitation L2):
  curated parquet, splits, features (Morgan FP + ESM-2 embeddings + indexes), and
  21 per-target benchmark CSVs (7 models × 3 splits). Sourced from kinase-affinity-baselines
  v1 + the AWS kinase benchmark run.
- **GPCR per-target benchmark CSVs** at `data/processed/v1/per_target/` (Plan 2 supplement
  computed during P3-T14): 21 files mirroring the kinase format, materialized from the
  Plan 2 per-seed prediction NPZs via `scripts/compute_per_target_gpcr_metrics.py`.
- **Per-target mean binding-site pLDDT** at `data/processed/v1/per_target_plddt.csv`
  (P3-T14). 543 targets attempted, 303 successful (36/36 GPCR + 267/507 kinase).
  Driver: `scripts/compute_per_target_plddt.py` using the v1.2.0 library's
  `fetch_structure` + `fetch_binding_site` + `compute_binding_site_plddt`.
- **Three analysis notebooks**: `notebooks/05_scaffold_diversity.ipynb` (P3-T18 — Figure 3 +
  Table 4 scaffold portion), `notebooks/06_plddt_analysis.ipynb` (P3-T19 — Figure 4 +
  Table 4 pLDDT rows, renamed from `06_rns_analysis.ipynb` after the P3-T6 metric pivot),
  and `notebooks/07_cross_class_comparison.ipynb` (P3-T20 — Tables 1-3 + Figures 1, 2, 5
  using the library's H1-H4 hypothesis tests).
- **Application module** `src/gpcr_aminergic_benchmarks/analyses/scaffold_diversity.py`
  (P3-T12) — thin wrapper around the v1.2.0 library's `compute_scaffold_metrics`.
- **Results inventory** `results/README.md` (P3-T21) — maps every table/figure to a
  manuscript section + producing notebook + caveats.
- **Four main-text tables** + **five main-text figures** under `results/tables/` and
  `results/figures/`.
- **Two supplementary tables**: `per_target_metrics_scaffold.csv` (P3-T12, 543 rows)
  and `structure_provenance.csv` (P3-T19, 543 rows with AlphaFold source / binding-site
  residue count / NaN-cause documentation).

### Changed

- **`pyproject.toml` library dependency** bumped from `@plan3-development` to
  `@v1.2.0` (P3-T22) — the analysis is now pinned to a stable library release.
- **`data/processed/v1/protein_sequences.json`**: replaced 22 TrEMBL accessions
  with canonical Swiss-Prot via UniProt's reviewed-search REST API (AlphaFold
  DB only serves Swiss-Prot; the original TrEMBL mappings produced 100%
  AlphaFold 404s).

### Documentation

- Three significant cross-class findings detailed in `results/README.md`:
  scaffold-diversity slopes differ for 3/5 metrics in the random→scaffold direction;
  ESM-2 advantage × pLDDT cross-class interaction (p = 0.031); H2 class × split
  interaction on raw RMSE (p = 5.9e-05).
- Four caveats documented: kinase per-seed RF/XGB/EN/MLP gap (Plan 1 data
  limitation); kinase pLDDT coverage at 53% (KLIFS API limitations); GPCR
  target-split n=5; bootstrap CI nominal coverage at n=5 seeds.
- RNS → mean binding-site pLDDT metric pivot narrative documented in
  `kinase-affinity-baselines/docs/superpowers/plans/2026-05-28-plan3-completion-summary.md`.

## [1.0.0] – 2026-05-26

Initial repo skeleton; depends on target-affinity-ml v1.1.0. Aminergic target list moved from kinase-affinity-baselines; aminergic TargetClassConfig builder added. Data ingestion, features, and benchmark land in subsequent commits.
