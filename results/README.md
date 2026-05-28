# Plan 3 Results Inventory

This directory contains the tables, figures, and supplementary files produced
by the Plan 3 cross-class analysis. Each output is consumed by a specific
manuscript section.

**Plan 3 status:** results complete; manuscript drafting deferred to Plan 4.

## Main-text tables

| File | Paper section | Produced by | Notes |
|---|---|---|---|
| `tables/01_dataset_summary.csv` | §Methods – Datasets | `notebooks/07_cross_class_comparison.ipynb` | 2 rows × 7 cols. Side-by-side kinase vs GPCR-aminergic dataset statistics. |
| `tables/02_headline_rmse.csv` | §Results – Cross-class replication | `notebooks/07_cross_class_comparison.ipynb` | 42 rows (7 models × 3 splits × 2 classes). RMSE mean ± SD over 5 seeds + `rmse_str` formatted column for display. |
| `tables/03_hypothesis_outcomes.csv` | §Results – Hypothesis tests | `notebooks/07_cross_class_comparison.ipynb` | 4 rows (one per H1–H4) with rollup verdict counts. Companion CSVs `03_hypothesis_outcomes_{h1,h2,h3,h4}.csv` provide per-test details. |
| `tables/04_metric_correlations.csv` | §Results – Cross-class scaffold + pLDDT | `notebooks/06_plddt_analysis.ipynb` (T19 unifies T18) | 13 rows: 10 scaffold (5 metrics × 2 directions) + 1 pLDDT regression + 2 pLDDT distribution (KS, Welch). Distribution-comparison rows (`direction = "plddt_distribution"`, `metric ∈ {"ks_2sample", "welch_t"}`) repurpose `slope_kinase` as the test statistic (KS D for the 2-sample test, Welch t for the means t-test) and `slope_kinase_p` as the test p-value. Regression rows use `slope_kinase` in its standard sense. Distinguish by checking the `direction` column. |

## Companion / supplementary tables

| File | Use | Notes |
|---|---|---|
| `tables/03_hypothesis_outcomes_h1.csv` | H1 per-test details | `model_pair × class × split`, with effect size, Bonferroni-corrected p, and verdict. |
| `tables/03_hypothesis_outcomes_h2.csv` | H2 per-test details | `model × class × transition` with ratio and in-range bool. Includes a class × split interaction row. |
| `tables/03_hypothesis_outcomes_h3.csv` | H3 advantage values | `class × split` with mean ESM-2 advantage and Bonferroni-corrected p. |
| `tables/03_hypothesis_outcomes_h4.csv` | H4 per-pair flip rates | `model_pair × split` with per-class flip rates, diff, bootstrap CI. |
| `tables/04_metric_correlations_scaffold.csv` | Scaffold-only table 4 | Superseded by the unified `04_metric_correlations.csv`; kept for provenance. |
| `supplement/per_target_metrics_scaffold.csv` | Per-target scaffold diversity | 543 rows (36 GPCR + 507 kinase). Input to T18; also a paper supplementary table. |
| `supplement/structure_provenance.csv` | Plan 3 design spec 5.4 Tier 1 | 543 rows. Documents the AlphaFold source + binding-site residue count for each target. NaN pLDDT rows have an `error` column indicating the failure reason. |

## Main-text figures

| File | Paper section | Produced by | Notes |
|---|---|---|---|
| `figures/figure1_design_overview.png` | §Methods – Benchmark design | `notebooks/07_cross_class_comparison.ipynb` | Schematic showing classes → splits → models pipeline. Working figure; T-21 leaves polish for manuscript phase. |
| `figures/figure2_headline_replication.png` | §Results – Cross-class replication | `notebooks/07_cross_class_comparison.ipynb` | 2-panel grouped-bar of per-model RMSE × split, per class. |
| `figures/figure3_scaffold_degradation.png` | §Results – Scaffold-diversity correlation | `notebooks/05_scaffold_diversity.ipynb` | 2 rows × 5 cols (directions × metrics) scatter + per-class regression lines. |
| `figures/figure4_plddt_advantage.png` | §Results – pLDDT-stratified ESM-2 advantage | `notebooks/06_plddt_analysis.ipynb` | Single scatter, both classes overlaid, regression lines, interaction p in title. |
| `figures/figure5_hypothesis_summary.png` | §Discussion – Hypothesis outcome summary | `notebooks/07_cross_class_comparison.ipynb` | Grouped-bar count of significant vs non-significant tests per hypothesis. |

## Key findings (one-line each)

1. **Scaffold-diversity slopes differ across classes** for `scaffold_entropy`, `largest_cluster_fraction`, and `mean_tanimoto` (interaction p ≤ 0.0164). Kinases show flat slopes; GPCRs show steeper degradation responses on cluster_fraction and mean_tanimoto.

2. **ESM-2 advantage × pLDDT interaction is significant** (p = 0.031; Plan 3 T19/T20 Part B). GPCRs show positive slope (ESM-2 helps less at higher pLDDT — perhaps overconfident structure); kinases show flat slope around zero.

3. **H3 Part A vs Part B distinction:** the per-table RMSE-level class × split interaction (Part A) is NOT significant (p = 0.318), while the per-target advantage × pLDDT interaction (Part B) IS significant (p = 0.031). These are *different statistical questions*: the table-level test asks whether the *average* ESM-2-vs-MLP gap depends on class × split; the per-target test asks whether the *individual-target* ESM-2 advantage tracks pLDDT differently across classes. Both are reported; the manuscript should discuss the difference.

## Caveats & limitations

### Kinase per-seed RF/XGB/EN/MLP gap

The kinase reference data hosted at `data/kinase_reference/benchmark_v1/per_seed_metrics.csv` includes only the three deep-model per-seed RMSE values (`esm_fp_mlp`, `fusion`, `gnn`); the other four models (`random_forest`, `xgboost`, `elasticnet`, `mlp`) have only mean + SD in `multi_seed_aggregated.csv`. Consequently:

- **H1 (RF vs deep)** runs on GPCR data only — 6 GPCR tests (model_pair × split) rather than 12. The Bonferroni correction still divides by the planned 12 per pre-registration.
- **H4 (single-seed flip rate)** is similarly GPCR-only — 6 GPCR rows out of 12 planned.
- **H3 Part A** is similarly affected (the kinase MLP per-seed isn't available).

**Impact:** within-GPCR conclusions are robust. The cross-class direction for H1 and H4 was not assessed.

### Kinase pLDDT coverage at 53%

Of 507 kinase targets, KLIFS has binding-site annotations for ~270 (rate-limited fetches during T14 capped at 267 successful). The remaining ~240 are either KLIFS-coverage gaps (~50 known pseudokinases or recently-named kinases) or transient API failures that cached empty results. H3 cross-class regression has n = 203 kinase + 36 GPCR = 239, sufficient statistical power.

### GPCR target-split is only 5 targets

The GPCR target-split holds out one of six aminergic subfamilies (DRD/HTR/ADR/HRH/CHRM/TAAR). With only 5 test targets, the scaffold→target degradation regression on the GPCR side has minimal power. T18 and T20 both surface n_per_class to make this visible.

### Metric pivot from RNS to mean binding-site pLDDT

The original spec referenced Prabakaran-Bromberg Residue Neighborhood Significance (RNS). Investigation in P3-T1 through P3-T6 revealed that the published Prabakaran-Bromberg paper concerns LM-embedding evaluation, not binding-site residue significance. Two implementation attempts (raw column entropy, JSD vs Swiss-Prot background) failed the validation gate (Spearman ρ = -0.524 and -0.476 against ConSurf reference). The team pivoted to mean binding-site pLDDT (AlphaFold's per-residue model confidence averaged over binding-site residues) as the per-target structural-confidence metric. This is the published AlphaFold quantity, no fragile rank-correlation validation needed, and biologically meaningful for the H3 question (does ESM-2's contribution correlate with regions where AlphaFold is also confident?). T16 + T17 (PDB-vs-AlphaFold sensitivity, structure-source decision branch) are MOOT post-pivot because pLDDT is an AlphaFold-specific concept; we use AlphaFold uniformly. See the T23 completion summary for the full pivot narrative.

## File lineage

```
Plan 1 (kinase preprint v1) → kinase reference data committed in P3-T9
Plan 2 (GPCR data + benchmark) → GPCR per-seed JSONs + curated parquet + splits
T11-T13 (library)             → scaffold_diversity + fit_degradation_regression
T14 + T15 (library + script)  → per_target_plddt.csv + hypothesis_tests
T12 (script)                  → per_target_metrics_scaffold.csv
T18 (notebook 05)             → figure3 + 04_metric_correlations_scaffold.csv
T19 (notebook 06)             → figure4 + 04_metric_correlations.csv (unified) + structure_provenance.csv
T20 (notebook 07)             → figures 1, 2, 5 + tables 01, 02, 03 (+ companions)
T21 (this file)               → README.md
```
