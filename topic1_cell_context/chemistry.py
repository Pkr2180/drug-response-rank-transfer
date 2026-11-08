"""Optional RDKit adapter: standardized parent structures, fingerprints, scaffolds.

Chemistry decisions must be reviewed for the endpoint. Parent standardization
does not model metabolites, dose, stereochemical mixtures or bioavailability.
"""

import argparse

import pandas as pd

from research_common.validation import frame_contract, identifiers, real


def featurize(frame, similarity_cutoff=0.6):
    frame_contract(frame, ["compound_id", "smiles"])
    identifiers(frame, ["compound_id", "smiles"])
    real(similarity_cutoff, "similarity_cutoff", minimum=1e-12, maximum=1.0, upper_open=True)
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit.ML.Cluster import Butina
    except ImportError as exc:
        raise ImportError('Install the chemistry extra: python -m pip install -e ".[chemistry]"') from exc
    if not {"compound_id", "smiles"} <= set(frame) or frame.compound_id.duplicated().any() or frame.empty:
        raise ValueError("Unique compound_id and nonempty smiles required")
    if not 0 < similarity_cutoff < 1:
        raise ValueError("similarity_cutoff must be between 0 and 1")
    frame = frame.sort_values("compound_id").reset_index(drop=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints, rows = [], []
    for record in frame.to_dict("records"):
        molecule = Chem.MolFromSmiles(record["smiles"])
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES for {record['compound_id']}")
        molecule = rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(molecule))
        try:
            Chem.SanitizeMol(molecule)
        except Exception as exc:
            raise ValueError(f"Cannot sanitize standardized parent of {record['compound_id']}") from exc
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        fp = generator.GetFingerprint(molecule)
        fingerprints.append(fp)
        row = {
            "compound_id": record["compound_id"],
            "canonical_smiles": canonical,
            "structure_key": canonical,
            "scaffold_id": scaffold or "ACYCLIC",
        }
        row.update({f"mol__morgan_{i}": int(v) for i, v in enumerate(fp)})
        rows.append(row)
    distances = []
    for i in range(1, len(fingerprints)):
        distances.extend(1 - v for v in DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i]))
    clusters = Butina.ClusterData(distances, len(fingerprints), 1 - similarity_cutoff, isDistData=True)
    for cluster_id, members in enumerate(clusters):
        for i in members:
            rows[i]["cluster_id"] = f"BUTINA_{cluster_id}"
    result = pd.DataFrame(rows)
    if result.structure_key.duplicated().any():
        raise ValueError(
            "Multiple compound IDs standardize to the same structure; reconcile before splitting"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--similarity-cutoff", type=float, default=0.6)
    args = parser.parse_args()
    from pathlib import Path

    if Path(args.output).exists():
        raise FileExistsError(args.output)
    from research_common.io import read_table

    featurize(read_table(args.input), args.similarity_cutoff).to_csv(args.output, index=False, mode="x")


if __name__ == "__main__":
    main()
