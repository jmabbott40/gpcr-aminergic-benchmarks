"""The aminergic GPCR TargetClassConfig builder.

`build_aminergic_config()` is a function rather than a module-level constant
because the explicit_target_ids and subfamily_map require the gene -> ChEMBL ID
resolution that happens at fetch time (Task 7), not at import time.
"""
from __future__ import annotations

from target_affinity_ml.data.target_class_config import TargetClassConfig
from gpcr_aminegric_benchmarks.target_lists import get_gene_to_family


def build_aminergic_config(resolved_ids: dict[str, str]) -> TargetClassConfig:
    """Construct the aminergic GPCR TargetClassConfig.

    Parameters
    ----------
    resolved_ids : dict[str, str]
        Mapping of HGNC gene symbol -> resolved ChEMBL target ID,
        produced by target_lists.resolve_chembl_ids() during data ingestion.

    Returns
    -------
    TargetClassConfig
        Configured for the aminergic class via the explicit-target-list path,
        with subfamily_map giving each ChEMBL ID's receptor family
        (dopamine, serotonin, adrenergic, histamine, muscarinic).
    """
    gene_to_family = get_gene_to_family()
    subfamily_map = {
        chembl_id: gene_to_family[gene]
        for gene, chembl_id in resolved_ids.items()
    }
    return TargetClassConfig(
        class_name="gpcr_aminergic",
        raw_filename_stem="chembl_gpcr_aminergic",
        explicit_target_ids=list(resolved_ids.values()),
        subfamily_map=subfamily_map,
    )
