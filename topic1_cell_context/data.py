"""Input contract and explicitly synthetic compound/context demonstration."""

import numpy as np
import pandas as pd
from scipy.special import expit

from research_common.validation import frame_contract, identifiers, numeric

REQUIRED = [
    "sample_id",
    "compound_id",
    "structure_key",
    "scaffold_id",
    "cluster_id",
    "context_id",
    "study_id",
    "split_role",
    "endpoint",
    "label",
    "data_origin",
]
MODALITIES = ["mol", "context", "ptm"]


def validate(frame, allow_synthetic=False):
    frame_contract(frame, REQUIRED)
    identifiers(frame, [c for c in REQUIRED if c != "label"])
    features = [c for c in frame if c.startswith(tuple(p + "__" for p in MODALITIES))]
    frame = numeric(frame, ["label", *features], allow_nan=True)
    if frame[REQUIRED].isna().any().any() or frame[REQUIRED].astype(str).eq("").any().any():
        raise ValueError("Required metadata and labels must not be missing or blank")
    if frame.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique")
    if frame.duplicated(["compound_id", "context_id", "endpoint"]).any():
        raise ValueError("Aggregate assay replicates first; repeated compound/context/endpoint")
    if not frame.label.isin([0, 1]).all() or frame.endpoint.nunique() != 1:
        raise ValueError("One binary toxicity endpoint per run is required")
    if set(frame.split_role) != {"development", "external"}:
        raise ValueError("split_role must contain development and external, and no other values")
    if not set(frame.data_origin) <= {"synthetic", "experimental"}:
        raise ValueError("data_origin must be synthetic or experimental")
    if not allow_synthetic and (frame.data_origin == "synthetic").any():
        raise ValueError("Synthetic data require explicit demonstration mode")
    if frame.data_origin.nunique() != 1:
        raise ValueError("Do not mix synthetic and experimental observations")
    for identifier in ["structure_key", "scaffold_id", "cluster_id"]:
        if frame.groupby("compound_id")[identifier].nunique().max() != 1:
            raise ValueError(f"Each compound must have one {identifier}")
    if frame.groupby("structure_key").compound_id.nunique().max() != 1:
        raise ValueError("Duplicate structure under multiple compound IDs")
    for prefix in MODALITIES:
        cols = [c for c in frame if c.startswith(prefix + "__")]
        if not cols:
            raise ValueError(f"Missing {prefix}__ feature block")
        array = frame[cols].to_numpy(dtype=float)
        if np.isinf(array).any() or np.isnan(array).all():
            raise ValueError(f"Invalid or wholly absent {prefix} block")
    dev, ext = frame[frame.split_role == "development"], frame[frame.split_role == "external"]
    for name, subset in [("development", dev), ("external", ext)]:
        if subset.label.nunique() != 2 or subset.compound_id.nunique() < 8:
            raise ValueError(f"{name} needs two classes and at least eight compounds")
    for col in ["compound_id", "structure_key", "study_id"]:
        if set(dev[col]) & set(ext[col]):
            raise ValueError(f"Development/external overlap in {col}")
    for prefix in MODALITIES:
        cols = [c for c in features if c.startswith(prefix + "__")]
        if dev[cols].isna().all().all():
            raise ValueError(f"Development cohort has no observed {prefix} features")
    return (
        dev.sort_values("sample_id").reset_index(drop=True),
        ext.sort_values("sample_id").reset_index(drop=True),
    )


def synthetic_compounds(seed=17, n_development=150, n_external=60):
    """No actual compounds, patients, PTM sites, scaffolds or toxicity measurements."""
    rng = np.random.default_rng(seed)
    rows = []
    for role, count, offset in [("development", n_development, 0), ("external", n_external, 10000)]:
        for j in range(count):
            compound = j + offset
            mol = rng.normal(size=5)
            if role == "external":
                mol += 0.4
            for context in range(2):
                cell = rng.normal(size=3) + context * 0.4
                ptm = rng.normal(size=2)
                logit = 0.9 * mol[0] - 0.6 * mol[1] + 0.7 * cell[0] + 0.5 * ptm[0] - 0.5
                if role == "external":
                    logit += 0.3 * cell[1]
                row = {
                    "sample_id": f"SIM_{compound}_{context}",
                    "compound_id": f"SIM_C{compound}",
                    "structure_key": f"SYNTHETIC_STRUCTURE_{compound}",
                    "scaffold_id": f"SIM_S{compound // 5}",
                    "cluster_id": f"SIM_K{compound // 10}",
                    "context_id": f"SIM_CONTEXT_{context}",
                    "study_id": f"SIM_{role}",
                    "split_role": role,
                    "endpoint": "SYNTHETIC_TOXICITY",
                    "label": int(rng.random() < expit(logit)),
                    "data_origin": "synthetic",
                }
                row.update({f"mol__{i}": v for i, v in enumerate(mol)})
                row.update({f"context__{i}": v for i, v in enumerate(cell)})
                row.update({f"ptm__{i}": v for i, v in enumerate(ptm)})
                rows.append(row)
    return pd.DataFrame(rows)
