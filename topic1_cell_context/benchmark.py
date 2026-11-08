"""Audit internal protocols against an independent, never-trained-on cohort.

All models, seeds and protocols are fixed before inspecting external performance.
The same fitted model supplies each internal/external comparison, avoiding a
training-size mismatch. Repeated splits are dependent, not independent studies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research_common.metrics import (
    binary_scores,
    cluster_mean_interval,
    rank_agreement,
    selected_indices,
    selection_regret,
    unit_losses,
)
from research_common.models import LateFusion, feature_columns, fit_checked, logistic_model
from research_common.randomness import derive_seed
from research_common.splits import assert_disjoint, group_split
from research_common.validation import integer, mapping, unique_choices

from .data import validate

PROTOCOLS = {"random_compound": "compound_id", "scaffold": "scaffold_id", "chemical_cluster": "cluster_id"}
SPECS = {
    "molecular_only": ["mol"],
    "context_only": ["context"],
    "early_fusion": ["mol", "context", "ptm"],
    "without_ptm": ["mol", "context"],
    "late_fusion": ["mol", "context", "ptm"],
}


def validate_config(config):
    mapping(config)
    unknown = set(config) - {"seeds", "data_seed", "protocols", "models", "bootstrap_draws"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if not isinstance(config.get("seeds"), list) or not config["seeds"]:
        raise ValueError("Nonempty unique seeds required")
    for seed in config["seeds"]:
        integer(seed, "seed", maximum=2**32 - 1)
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be unique")
    integer(config.get("data_seed", 17), "data_seed", maximum=2**32 - 1)
    unique_choices(config.get("protocols"), "protocols", PROTOCOLS)
    unique_choices(config.get("models"), "models", SPECS, minimum=2)
    integer(config.get("bootstrap_draws"), "bootstrap_draws", minimum=50)


def run_benchmark(frame, config, allow_synthetic=False, progress=None):
    validate_config(config)
    development, external = validate(frame, allow_synthetic)
    rows, predictions, splits, weights = [], [], [], []
    # External outcomes are used only after the complete predeclared prediction pass.
    frozen_predictions = []
    for protocol in config["protocols"]:
        group = PROTOCOLS[protocol]
        for seed in config["seeds"]:
            if progress:
                progress(f"Topic 1: {protocol}, seed {seed}")
            training_pool, internal = group_split(development, group, seed, 0.25)
            train, calibration = group_split(training_pool, group, derive_seed(seed, "calibration"), 0.25)
            for left, right in [(train, internal), (train, calibration), (calibration, internal)]:
                assert_disjoint(left, right, ["compound_id", group])
            for partition, subset in [("train", train), ("calibration", calibration), ("internal", internal)]:
                for record in subset[["sample_id", "compound_id", "scaffold_id", "cluster_id"]].to_dict(
                    "records"
                ):
                    splits.append({"protocol": protocol, "seed": seed, "partition": partition, **record})
            for model_name in config["models"]:
                prefixes = SPECS[model_name]
                if model_name == "late_fusion":
                    model = LateFusion(prefixes, seed).fit(train, calibration)
                    p_internal, p_external = model.predict(internal), model.predict(external)
                    weights.extend(
                        {"protocol": protocol, "seed": seed, "modality": m, "weight": w}
                        for m, w in zip(prefixes, model.weights)
                    )
                else:
                    columns = feature_columns(train, prefixes)
                    model = logistic_model(seed)
                    fit_checked(model, train[columns], train.label)
                    p_internal = model.predict_proba(internal[columns])[:, 1]
                    p_external = model.predict_proba(external[columns])[:, 1]
                frozen_predictions.append((protocol, seed, model_name, internal, p_internal, p_external))
    for protocol, seed, model_name, internal, pi, pe in frozen_predictions:
        row = {"protocol": protocol, "seed": seed, "model": model_name}
        for prefix, subset, p in [("internal", internal, pi), ("external", external, pe)]:
            scores = binary_scores(subset.label, p)
            row.update({prefix + "_" + k: v for k, v in scores.items()})
            losses = (subset.label.to_numpy() - p) ** 2
            interval_group = PROTOCOLS[protocol]
            n_blocks = subset[interval_group].nunique()
            if n_blocks >= 2:
                mean, low, high = cluster_mean_interval(
                    losses,
                    subset.compound_id,
                    seed,
                    config["bootstrap_draws"],
                    resample_groups=subset[interval_group],
                )
            else:
                mean, low, high = float(unit_losses(losses, subset.compound_id).mean()), np.nan, np.nan
            row.update(
                {
                    prefix + "_compound_brier": mean,
                    prefix + "_ci_low": low,
                    prefix + "_ci_high": high,
                    prefix + "_n_compounds": subset.compound_id.nunique(),
                    prefix + "_n_resampling_blocks": n_blocks,
                    prefix + "_interval_status": "available" if n_blocks >= 2 else "insufficient_blocks",
                    "resampling_unit": interval_group,
                }
            )
            predictions.extend(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "model": model_name,
                    "partition": prefix,
                    "sample_id": sid,
                    "compound_id": cid,
                    "label": int(y),
                    "probability": float(prob),
                }
                for sid, cid, y, prob in zip(subset.sample_id, subset.compound_id, subset.label, p)
            )
        row["brier_gap"] = row["external_compound_brier"] - row["internal_compound_brier"]
        row["absolute_brier_gap"] = abs(row["brier_gap"])
        row["ap_optimism"] = row["internal_average_precision"] - row["external_average_precision"]
        rows.append(row)
    scores = pd.DataFrame(rows)
    audit = []
    for (protocol, seed), subset in scores.groupby(["protocol", "seed"]):
        internal, external_score = subset.internal_compound_brier, subset.external_compound_brier
        audit.append(
            {
                "protocol": protocol,
                "seed": seed,
                "rank_agreement": rank_agreement(internal, external_score),
                "selection_regret": selection_regret(internal, external_score),
                "mean_absolute_brier_gap": subset.absolute_brier_gap.mean(),
                "selection_tie_count": len(selected_indices(internal)),
                "internally_selected_model": "|".join(sorted(subset.iloc[selected_indices(internal)].model)),
                "externally_best_model_descriptive_only": "|".join(
                    sorted(subset.iloc[selected_indices(external_score)].model)
                ),
            }
        )
    return {
        "scores": scores,
        "meta_evaluation": pd.DataFrame(audit),
        "predictions": pd.DataFrame(predictions),
        "split_assignments": pd.DataFrame(splits),
        "fusion_weights": pd.DataFrame(weights, columns=["protocol", "seed", "modality", "weight"]),
    }
