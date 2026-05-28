# Dataset Card: Aminergic GPCR Binding Affinity (v1)

## Overview

Curated aminergic G protein-coupled receptor (GPCR) binding affinity dataset from ChEMBL
for benchmarking ML models. Produced by Plan 2 of the kinase-vs-GPCR cross-class comparison.
Mirrors the kinase v1 dataset in curation methodology so that any difference in model
performance is attributable to biology and data characteristics rather than pipeline differences.

| Property | Value |
|----------|-------|
| Source | ChEMBL (current release) |
| Target family | Aminergic GPCRs (Homo sapiens) |
| Receptor subfamilies | dopamine, serotonin, adrenergic, histamine, muscarinic, trace_amine |
| Targets | 36 (gene → ChEMBL-ID mapping in `data/processed/v1/resolved_target_ids.json`) |
| Activity types | Ki, IC50, Kd |
| Response variable | pActivity (−log₁₀ M) |
| Dataset version | v1 |
| Creation date | May 2026 |

## Target List

36 targets across 6 receptor subfamilies:

| Subfamily | Count | Gene symbols |
|-----------|-------|-------------|
| Serotonin | 12 | HTR1A, HTR1B, HTR1D, HTR1E, HTR1F, HTR2A, HTR2B, HTR2C, HTR4, HTR5A, HTR6, HTR7 |
| Adrenergic | 9 | ADRA1A, ADRA1B, ADRA1D, ADRA2A, ADRA2B, ADRA2C, ADRB1, ADRB2, ADRB3 |
| Dopamine | 5 | DRD1, DRD2, DRD3, DRD4, DRD5 |
| Muscarinic | 5 | CHRM1, CHRM2, CHRM3, CHRM4, CHRM5 |
| Histamine | 4 | HRH1, HRH2, HRH3, HRH4 |
| Trace amine | 1 | TAAR1 |

The Plan 2 audit (spec Section 4.4) identified 30 / 36 targets as viable at the ≥ 500-record
threshold. Curation does NOT drop low-record targets — they remain in the output — but see
"Known Limitations" for the six below-threshold targets.

## Inclusion Criteria

Applied by `target_affinity_ml.fetch_bioactivities` at the ChEMBL API level:

- `standard_type` ∈ {IC50, Ki, Kd}
- `standard_relation = "="` (exact measurements; no inequalities)
- `pchembl_value` present (non-null)

Applied by `scripts/fetch_gpcr_data.py` via post-fetch DataFrame filtering:

- `assay_type = "B"` (binding assays only)
- `confidence_score >= 7` (satisfied by construction; see Known Limitations)
- `standard_units = "nM"` (hard constraint on units)

## Molecule Standardization

Applied using RDKit via `target_affinity_ml.data.standardize.standardize_dataframe`:

1. Salt removal (keep largest fragment)
2. Charge neutralization
3. Canonical SMILES generation
4. Molecular weight filter: 100–900 Da
5. Maximum 100 heavy atoms

**Molecules removed during standardization:**
- Invalid SMILES: 124
- Failed MW / heavy-atom filter: 789
- Total removed: 913 (1.0% of raw records; 99.0% retained)

## Duplicate Handling

For identical (canonical_smiles, target_chembl_id, standard_type) groups:

- Aggregation: **median** pActivity value
- Noise flag: if ≥ 3 measurements and std > 1.0 pActivity units, the record is flagged `is_noisy = True`
- All measurements contribute to the median regardless of noise flag

Pre-dedup unique groups: 88,426 → post-dedup: 70,163 (18,263 collapsed duplicates)

## Quality Filters

- pActivity range: [3.0, 12.0] (removes values below 1 mM or above 1 pM)
- No records were removed by this filter (all pActivity values fell within range after dedup)
- Classification threshold: pActivity ≥ 6.0 (Ki/IC50/Kd ≤ 1 μM) = **active**

## Final Dataset Statistics

| Property | Value |
|----------|-------|
| Raw records (pre-curation) | 89,339 |
| Curated records | 70,163 |
| Unique compounds (canonical SMILES) | 33,195 |
| Unique targets | 36 |
| Unique Bemis-Murcko scaffolds | 7,017 |
| Active records (pActivity ≥ 6.0) | 55,788 (79.5%) |
| Inactive records | 14,375 (20.5%) |
| Noisy records (std > 1.0, n ≥ 3) | 114 |

### Activity-Type Breakdown

| Type | Records | Percentage |
|------|---------|-----------|
| Ki | 59,997 | 85.5% |
| IC50 | 9,307 | 13.3% |
| Kd | 859 | 1.2% |

### Per-Subfamily Breakdown

| Subfamily | Records | Unique Compounds | Unique Targets |
|-----------|---------|-----------------|----------------|
| Serotonin | 26,305 | 15,062 | 12 |
| Dopamine | 19,036 | 10,213 | 5 |
| Adrenergic | 9,338 | 4,109 | 9 |
| Muscarinic | 8,021 | 3,670 | 5 |
| Histamine | 7,416 | 5,985 | 4 |
| Trace amine | 47 | 47 | 1 |
| **Total** | **70,163** | **33,195** | **36** |

### Per-Target Record-Count Distribution

- Min: 47 records (TAAR1)
- Median: 1,310 records
- Max: 8,518 records (DRD2)

**All 36 targets ranked by record count:**

| Rank | ChEMBL ID | Gene | Subfamily | Records |
|------|-----------|------|-----------|---------|
| 1 | CHEMBL217 | DRD2 | dopamine | 8,518 |
| 2 | CHEMBL234 | DRD3 | dopamine | 5,767 |
| 3 | CHEMBL224 | HTR2A | serotonin | 5,497 |
| 4 | CHEMBL214 | HTR1A | serotonin | 5,303 |
| 5 | CHEMBL3371 | HTR6 | serotonin | 4,128 |
| 6 | CHEMBL264 | HRH3 | histamine | 3,816 |
| 7 | CHEMBL225 | HTR2C | serotonin | 3,325 |
| 8 | CHEMBL219 | DRD4 | dopamine | 2,982 |
| 9 | CHEMBL3155 | HTR7 | serotonin | 2,957 |
| 10 | CHEMBL245 | CHRM3 | muscarinic | 2,303 |
| 11 | CHEMBL211 | CHRM2 | muscarinic | 1,853 |
| 12 | CHEMBL1833 | HTR2B | serotonin | 1,792 |
| 13 | CHEMBL216 | CHRM1 | muscarinic | 1,735 |
| 14 | CHEMBL229 | ADRA1A | adrenergic | 1,537 |
| 15 | CHEMBL3759 | HRH4 | histamine | 1,492 |
| 16 | CHEMBL231 | HRH1 | histamine | 1,481 |
| 17 | CHEMBL2056 | DRD1 | dopamine | 1,407 |
| 18 | CHEMBL223 | ADRA1D | adrenergic | 1,322 |
| 19 | CHEMBL1821 | CHRM4 | muscarinic | 1,299 |
| 20 | CHEMBL232 | ADRA1B | adrenergic | 1,298 |
| 21 | CHEMBL210 | ADRB2 | adrenergic | 1,285 |
| 22 | CHEMBL1983 | HTR1D | serotonin | 1,236 |
| 23 | CHEMBL1867 | ADRA2A | adrenergic | 1,068 |
| 24 | CHEMBL1898 | HTR1B | serotonin | 1,016 |
| 25 | CHEMBL213 | ADRB1 | adrenergic | 924 |
| 26 | CHEMBL2035 | CHRM5 | muscarinic | 831 |
| 27 | CHEMBL1916 | ADRA2C | adrenergic | 779 |
| 28 | CHEMBL1942 | ADRA2B | adrenergic | 697 |
| 29 | CHEMBL1941 | HRH2 | histamine | 627 |
| 30 | CHEMBL1875 | HTR4 | serotonin | 494 |
| 31 | CHEMBL246 | ADRB3 | adrenergic | 428 |
| 32 | CHEMBL3426 | HTR5A | serotonin | 366 |
| 33 | CHEMBL1850 | DRD5 | dopamine | 362 |
| 34 | CHEMBL1805 | HTR1F | serotonin | 116 |
| 35 | CHEMBL2182 | HTR1E | serotonin | 75 |
| 36 | CHEMBL5857 | TAAR1 | trace_amine | 47 |

Targets at ranks 30–36 fall below the 500-record viability threshold identified in the
Plan 2 audit. They are retained in the dataset; downstream analysis (Plan 3) should be
aware that individual-target metrics for these 7 targets will be higher-variance.

## Splitting Strategies

Defined in `configs/dataset_aminergic_v1.yaml` (identical to kinase v1 for cross-class
comparability). Splits are created in Task 9.

### Random Split
- 80% train / 10% validation / 10% test
- Stratified by target to maintain target representation in all splits
- Seed: 42

### Scaffold Split
- Murcko generic scaffolds (ring systems only, side chains removed)
- Entire scaffold groups assigned to one split (no scaffold leakage)
- Greedy assignment sorted by scaffold group size
- Seed: 42

### Target Split
- Entire receptor subfamilies held out for testing
- Tests generalization to unseen receptor families
- Seed: 42

## Known Limitations

### 1. Activity-type distribution differs from the kinase dataset

This dataset is **~85.5% Ki** (radioligand binding competition assays) whereas the kinase
dataset is ~80% IC50 (enzymatic inhibition assays). All three types (Ki, IC50, Kd) are
converted to pActivity via the same −log₁₀(M) transform, so combining them is numerically
consistent. However, Ki and IC50 are conceptually different quantities: IC50 depends on
substrate concentration and assay format (Cheng-Prusoff relationship), while Ki is an
intrinsic thermodynamic constant. Within each target the relationship is monotonic, so
the ranking signal is preserved. This asymmetry is a transparency item, not a confound,
but it should be acknowledged when comparing absolute pActivity values or model
performance between the GPCR and kinase datasets.

### 2. `confidence_score` filter satisfied by construction, not verified at record level

The library's `fetch_bioactivities` does not retrieve `confidence_score` as part of
`ACTIVITY_COLUMNS`, so it cannot be filtered post-fetch at the record level. The
inclusion criterion (`confidence_score >= 7`) is satisfied "by construction" because
all 36 targets were identified as `target_type = "SINGLE PROTEIN"`, which ChEMBL
convention assigns confidence_score = 9. This is the same behavior as the kinase
pipeline and is important for the cross-class comparison guardrail. However, it is
a convention not a per-record guarantee. Any records from assays that were
reassigned or reannotated after ChEMBL ingestion could in principle have lower
confidence scores that we would not detect.

### 3. TAAR1 (CHEMBL5857) — very low record count

TAAR1 (trace amine-associated receptor 1) has only **47 post-curation records**, well
below the audit's 500-record viability threshold. TAAR1 is included in the dataset for
completeness but individual-target metrics for TAAR1 should be treated as indicative
only. The trace_amine subfamily should be excluded from subfamily-level held-out splits
(Task 9) to avoid splits with too few test records.

### 4. Below-threshold targets (< 500 records)

Seven targets fall below the 500-record viability threshold: HTR4 (494), ADRB3 (428),
HTR5A (366), DRD5 (362), HTR1F (116), HTR1E (75), TAAR1 (47). These targets are
retained in the curated output. Downstream analyses that aggregate per-target metrics
should apply a minimum-record filter (suggested: ≥ 500 records) consistent with the
audit spec.

### 5. Activity-type mixing and inter-lab variability

Measurements from different labs and assay formats are combined. Inter-lab variability
is a known source of noise in ChEMBL data. The noise-flag mechanism (`is_noisy = True`
for 114 records with std > 1.0 pActivity over n ≥ 3 measurements) makes this explicit
but does not remove affected records. The noise rate (0.16% of curated records) is much
lower than the kinase dataset (the kinase dataset had ~1,965 noisy records from ~353K
curated), which likely reflects the narrower assay-type distribution here (most Ki
values come from standardized radioligand binding protocols).

### 6. Stereochemistry

Some compounds may have unresolved stereochemistry, which affects binding but is lost
in 2D canonical SMILES. This affects any fingerprint or 2D descriptor representation.

## Curation Configuration

Parameters from `configs/dataset_aminergic_v1.yaml`:

```yaml
standardization:
  mw_min: 100.0
  mw_max: 900.0
  max_heavy_atoms: 100

duplicates:
  aggregation: median
  noise_std_threshold: 1.0
  min_measurements_for_noise_flag: 3

quality:
  pactivity_min: 3.0
  pactivity_max: 12.0

classification:
  active_pactivity_threshold: 6.0
```

All parameters are identical to kinase `configs/dataset_v1.yaml` (no deviations).

## Citation

If using this dataset, please cite:

- ChEMBL: Mendez et al. (2019) "ChEMBL: towards direct deposition of bioassay data."
  *Nucleic Acids Research*, 47(D1), D930–D940.

---

# Kinase reference data (Plan 3 — Plan 1 L2 resolution)

As of Plan 3, this repo hosts the kinase Plan 1 reference data needed for
the cross-class analysis. Provenance and contents:

## Source

- **Source application repo:** [kinase-affinity-baselines](https://github.com/jmabbott40/kinase-affinity-baselines) (frozen at v1.0; the kinase preprint v1)
- **Library version that produced this data:** `target-affinity-ml` v1.0.0 (kinase pipeline, pre-class-agnostic refactor)
- **Date copied:** 2026-05-28
- **Hosting rationale:** The kinase repo is frozen at v1.0 for reproducibility of the published preprint, so the cross-class analysis cannot live there. Plan 3's analyses (notebooks 05–07) read from this in-repo copy. A future Zenodo deposit will replace the in-git copy with a versioned DOI.

## Contents (`data/kinase_reference/`)

| Path | Contents |
|---|---|
| `features/morgan_fp.npz` | Morgan FP (2048-bit, radius 2) for ~206K kinase compounds |
| `features/esm2_embeddings.npz` | ESM-2 (esm2_t33_650M_UR50D) mean-pooled embeddings for kinase targets |
| `features/smiles_index.json` | Compound index for the feature arrays |
| `features/target_index.json` | Target index for the ESM-2 embeddings |
| `curated_activities.parquet` | 353K curated kinase records (~206K compounds × 507 targets) |
| `splits/{random,scaffold,target}_split.json` | Train/val/test split index JSONs (seed=42) |
| `benchmark_v1/multi_seed_aggregated.csv` | Multi-seed aggregated metrics (mean ± SD per model × split) |
| `benchmark_v1/per_seed_metrics.csv` | All per-seed metrics |
| `benchmark_v1/per_target/per_target_<model>_<split>.csv` | Per-target metric breakdown (model × split × seed); the H3 cross-class correlation reads these |

## Not included (in-repo)

- `features/rdkit_descriptors.npz` (~80 MB) — exceeds GitHub's recommended file-size threshold. Available on the corresponding author's local filesystem and will be included in the eventual Zenodo deposit. Plan 3's analyses use the per-target metric CSVs rather than raw descriptor arrays, so this exclusion does not block the cross-class analysis.

## Cross-link

For the full kinase data pipeline + benchmark code, see [kinase-affinity-baselines](https://github.com/jmabbott40/kinase-affinity-baselines). The data card in that repo describes the original kinase dataset construction.
