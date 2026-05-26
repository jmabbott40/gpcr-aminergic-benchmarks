"""Aminergic Class A GPCR target list for Phase 1 audit.

Source: GPCRdb (https://gpcrdb.org/), filtered to Class A aminergic
receptors only (excludes 5-HT3 ionotropic receptors).
"""
from typing import Optional

AMINERGIC_TARGETS_BY_FAMILY = {
    "dopamine": ["DRD1", "DRD2", "DRD3", "DRD4", "DRD5"],
    "serotonin": [
        "HTR1A", "HTR1B", "HTR1D", "HTR1E", "HTR1F",
        "HTR2A", "HTR2B", "HTR2C",
        "HTR4", "HTR5A", "HTR6", "HTR7",
    ],
    "adrenergic": [
        "ADRA1A", "ADRA1B", "ADRA1D",
        "ADRA2A", "ADRA2B", "ADRA2C",
        "ADRB1", "ADRB2", "ADRB3",
    ],
    "histamine": ["HRH1", "HRH2", "HRH3", "HRH4"],
    "muscarinic": ["CHRM1", "CHRM2", "CHRM3", "CHRM4", "CHRM5"],
    "trace_amine": ["TAAR1"],
}


def get_all_gene_symbols(include_taar: bool = True) -> list[str]:
    """Return flat list of all aminergic gene symbols."""
    targets = []
    for family, members in AMINERGIC_TARGETS_BY_FAMILY.items():
        if family == "trace_amine" and not include_taar:
            continue
        targets.extend(members)
    return targets


def get_gene_to_family() -> dict[str, str]:
    """Return mapping of gene_symbol -> family name."""
    mapping = {}
    for family, members in AMINERGIC_TARGETS_BY_FAMILY.items():
        for gene in members:
            mapping[gene] = family
    return mapping


def resolve_chembl_ids(gene_symbols: Optional[list[str]] = None) -> dict[str, str]:
    """Resolve gene symbols to ChEMBL target IDs via API.

    Returns dict: gene_symbol -> chembl_id. Failed resolutions reported to stderr.
    Requires `chembl_webresource_client`.
    """
    import sys
    from chembl_webresource_client.new_client import new_client

    if gene_symbols is None:
        gene_symbols = get_all_gene_symbols(include_taar=True)

    target_client = new_client.target
    resolved = {}
    failed = []

    for gene in gene_symbols:
        # Each per-gene API call is wrapped in try/except so a transient
        # network failure or single bad response doesn't crash the entire
        # audit (which would lose all preceding successful resolutions).
        try:
            results = target_client.filter(
                target_components__target_component_synonyms__component_synonym=gene,
                target_type="SINGLE PROTEIN",
                organism="Homo sapiens",
            ).only(["target_chembl_id", "pref_name", "target_components"])

            chembl_id = None
            for result in results:
                for component in result.get("target_components", []):
                    synonyms = component.get("target_component_synonyms", [])
                    gene_match = any(
                        s.get("component_synonym") == gene
                        and s.get("syn_type") == "GENE_SYMBOL"
                        for s in synonyms
                    )
                    if gene_match:
                        chembl_id = result["target_chembl_id"]
                        break
                if chembl_id:
                    break

            if chembl_id:
                resolved[gene] = chembl_id
            else:
                failed.append(gene)
                print(f"WARN: could not resolve {gene} to a ChEMBL ID",
                      file=sys.stderr)
        except Exception as e:
            # API error (network timeout, 5xx, deprecated field, etc.)
            failed.append(gene)
            print(f"WARN: API error resolving {gene}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue

    if failed:
        print(f"\nResolution: {len(resolved)} succeeded, {len(failed)} failed",
              file=sys.stderr)
        print(f"Failed: {failed}", file=sys.stderr)

    return resolved
