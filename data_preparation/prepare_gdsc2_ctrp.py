"""Build the Topic 1 supplied-data table from the Partin et al. cross-study benchmark.

Development cohort: GDSCv2. External cohort: CTRPv2 (different study, compounds
disjoint from development). Endpoint: growth inhibition, AUC < AUC_CUTOFF.
The cutoff is fixed before any model is fitted and applied to both studies.
Feature selection uses no outcome information.

Usage: python prepare_gdsc2_ctrp.py RAW_DIR OUT_DIR   (OUT_DIR must not exist)
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from topic1_cell_context.chemistry import featurize

AUC_CUTOFF = 0.7
N_CONTEXTS = 120
N_GENES = 200
SEED = 20261104
import os

# Defaults reproduce the primary analysis; override to build replication cohorts (e.g. DEV_STUDY=CTRPv2 EXT_STUDY=GDSCv2).
DEV, EXT = os.environ.get("DEV_STUDY", "GDSCv2"), os.environ.get("EXT_STUDY", "CTRPv2")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_parent(smiles):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    parent = rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(molecule))
    try:
        Chem.SanitizeMol(parent)
    except Exception:
        return None
    return Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)


def main(raw, out):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True)
    x, y = raw / "x_data", raw / "y_data"
    response = pd.read_csv(y / "response.tsv", sep="\t", usecols=["source", "improve_sample_id", "improve_chem_id", "auc"])
    response = response[response.source.isin([DEV, EXT])]

    smiles = pd.read_csv(x / "drug_SMILES.tsv", sep="\t").dropna().drop_duplicates("improve_chem_id")
    smiles = smiles.rename(columns={"improve_chem_id": "compound_id", "canSMILES": "smiles"})
    smiles = smiles[smiles.compound_id.isin(response.improve_chem_id)].copy()
    smiles["parent"] = smiles.smiles.map(canonical_parent)
    invalid = int(smiles.parent.isna().sum())
    smiles = smiles.dropna(subset=["parent"])
    # One representative ID per standardized structure; duplicate IDs' responses are median-aggregated.
    representative = smiles.sort_values("compound_id").groupby("parent").compound_id.first()
    id_map = smiles.set_index("compound_id").parent.map(representative)
    merged_ids = int((id_map.index != id_map.values).sum())
    smiles = smiles[smiles.compound_id.isin(set(representative))]
    response = response[response.improve_chem_id.isin(id_map.index)].copy()
    response["improve_chem_id"] = response.improve_chem_id.map(id_map)
    response = response.groupby(["source", "improve_chem_id", "improve_sample_id"], as_index=False).auc.median()
    chem = featurize(smiles[["compound_id", "smiles"]].reset_index(drop=True))

    rppa = pd.read_csv(x / "cancer_RPPA.tsv", sep="\t", index_col=0)
    expression = pd.read_csv(x / "cancer_gene_expression.tsv", sep="\t", skiprows=[1, 2], index_col=0, low_memory=False)
    expression = expression.apply(pd.to_numeric, errors="coerce")
    lines = sorted(set(rppa.index) & set(expression.index) & set(response.improve_sample_id))
    rng = np.random.default_rng(SEED)
    lines = sorted(rng.choice(lines, size=min(N_CONTEXTS, len(lines)), replace=False))

    # Cohort construction: structures must not cross cohorts; drop shared ones from external.
    dev_ids = set(response[response.source == DEV].improve_chem_id)
    keyed = chem.set_index("compound_id").structure_key
    dev_structures = {keyed[c] for c in dev_ids if c in keyed.index}
    response = response[response.improve_sample_id.isin(lines) & response.improve_chem_id.isin(keyed.index)]
    is_ext = response.source == EXT
    shared = response.improve_chem_id.map(keyed).isin(dev_structures) | response.improve_chem_id.isin(dev_ids)
    response = response[~(is_ext & shared)]

    # Outcome-free feature selection: top-variance genes in development contexts; phospho-sites only.
    genes = expression.loc[lines].var().nlargest(N_GENES).index
    context = expression.loc[lines, genes].add_prefix("context__")
    phospho = [c for c in rppa.columns if re.search(r"_p[STY]\d", c)]
    ptm = rppa.loc[lines, phospho].add_prefix("ptm__")
    ptm = ptm.loc[:, ptm.notna().any()]
    ptm.columns = [re.sub(r"[^0-9A-Za-z_]", "_", c) for c in ptm.columns]
    mol = chem.loc[:, ~chem.columns.str.startswith("mol__")].copy()
    bits = chem.filter(like="mol__")
    dev_compounds = chem.compound_id.isin(response[response.source == DEV].improve_chem_id)
    bits = bits.loc[:, bits[dev_compounds.values].any()]
    mol = pd.concat([mol, bits], axis=1)

    table = response.rename(columns={"improve_chem_id": "compound_id", "improve_sample_id": "context_id"})
    table["label"] = (table.auc < AUC_CUTOFF).astype(int)
    table["study_id"] = table.source
    table["split_role"] = np.where(table.source == DEV, "development", "external")
    table["endpoint"] = f"growth_inhibition_auc_lt_{AUC_CUTOFF}"
    table["data_origin"] = "experimental"
    table["sample_id"] = table.source + "|" + table.compound_id + "|" + table.context_id
    table = table.merge(mol, on="compound_id").merge(context, left_on="context_id", right_index=True)
    table = table.merge(ptm, left_on="context_id", right_index=True, how="left")
    table = table.drop(columns=["source", "auc", "smiles", "canonical_smiles"], errors="ignore")
    first = [
        "sample_id", "compound_id", "structure_key", "scaffold_id", "cluster_id", "context_id",
        "study_id", "split_role", "endpoint", "label", "data_origin",
    ]
    table = table[first + [c for c in table.columns if c not in first]]
    table.to_csv(out / "compound_context.csv", index=False)

    summary = {
        "auc_cutoff": AUC_CUTOFF, "n_contexts": len(lines), "n_genes": N_GENES, "seed": SEED,
        "invalid_smiles_dropped": invalid, "duplicate_structure_ids_merged": merged_ids,
        "invalid_smiles_dropped": invalid, "duplicate_structure_ids_merged": merged_ids,
        "n_mol_bits": int(bits.shape[1]), "n_ptm": int(ptm.shape[1]),
        "cohorts": {
            role: {
                "rows": int(len(g)), "compounds": int(g.compound_id.nunique()),
                "contexts": int(g.context_id.nunique()), "prevalence": round(float(g.label.mean()), 4),
            }
            for role, g in table.groupby("split_role")
        },
        "source_sha256": {
            name: sha256(raw / rel)
            for name, rel in {
                "response": "y_data/response.tsv", "drug_SMILES": "x_data/drug_SMILES.tsv",
                "gene_expression": "x_data/cancer_gene_expression.tsv", "RPPA": "x_data/cancer_RPPA.tsv",
            }.items()
        },
        "table_sha256": sha256(out / "compound_context.csv"),
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
