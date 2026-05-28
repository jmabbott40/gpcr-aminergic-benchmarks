# Kinase reference data

This directory mirrors the kinase Plan 1 outputs that Plan 3's
cross-class analysis needs.

**Provenance:**
- Source repo: https://github.com/jmabbott40/kinase-affinity-baselines
- Source data version: v1 (kinase preprint v1)
- Source commit at copy time: a3eba41eb3321941b72b69156012a2cb1c571fdd
- Library version that produced this data: target-affinity-ml v1.0.0
- Date copied: 2026-05-28

**Why hosted here (and not in the kinase repo):**
The kinase application repo is frozen at v1.0 for reproducibility of
the published preprint. The cross-class analysis (Plan 3) consumes this
data, so it lives in the application repo that runs the analysis. A
future Zenodo deposit will replace this in-git copy with a versioned DOI.

**Files:**

```
features/                     — Morgan FP (morgan_fp.npz, 13 MB)
                                RDKit 2D descriptors (rdkit_descriptors.npz, 80 MB)
                                Compound SMILES index (smiles_index.json, 12 MB)
                                NOTE: esm2_embeddings.npz not present locally;
                                      ESM-2 embeddings were computed on AWS
                                      and consumed in-place. Not required for
                                      cross-class analysis (Plan 3 uses Morgan FP
                                      and RDKit descriptors only).
                                NOTE: target_index.json not present; target IDs
                                      are embedded in per-target CSV metric files.

curated_activities.parquet    — 353K curated kinase records (~206K compounds, 13 MB)

splits/                       — random_split.json, scaffold_split.json,
                                target_split.json (2.6 MB each)

benchmark_v1/
  multi_seed_aggregated.csv   — per-model/split aggregated metrics across 5 seeds
                                (equivalent to all_seeds_metrics.csv;
                                 columns: model, split, metric, mean, std, min, max, n_seeds)
  per_seed_metrics.csv        — per-seed raw metrics (S6 supplement table;
                                 columns: model, split, seed, rmse, r2)
  per_target/                 — per-target metrics for all 6 models × 3 splits
                                (18 CSV files, format: target_id, n_compounds,
                                 rmse, mae, r2, pearson_r, spearman_rho)
                                These are the CRITICAL ARTIFACT for Plan 3's
                                notebook 06 per-target cross-class correlation.
  predictions/                — flat prediction arrays per model × split
                                ({model}_{split}.npz, keys: y_test_true,
                                 y_test_pred, y_test_active, y_test_mean,
                                 y_test_std, y_val_true, y_val_pred)
                                NOTE: arrays are indexed (not labelled with
                                      compound/target IDs); use smiles_index.json
                                      + split JSONs to recover labels.
```

**Note on missing `all_seeds_metrics.csv`:**
Plan 3 spec references `benchmark_v1/all_seeds_metrics.csv`. The kinase repo
uses the naming `multi_seed_aggregated.csv` (same content). The per-seed
equivalent is `per_seed_metrics.csv`. Notebook 06 should load
`benchmark_v1/multi_seed_aggregated.csv`.
