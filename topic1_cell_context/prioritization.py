"""Transparent exploratory drug ranking, distinct from toxicity validation."""

import numpy as np

from research_common.validation import frame_contract, identifiers, numeric


def pareto_candidates(frame):
    """Return nondominated candidates: maximize relevance, minimize toxicity.

    relevance_score is an externally supplied prioritization proxy, not efficacy.
    No arbitrary scalar tradeoff is introduced. All candidates must share one
    comparable toxicity endpoint, context and exposure definition.
    """
    needed = {"compound_id", "relevance_score", "toxicity_probability", "endpoint", "context_id", "exposure"}
    frame_contract(frame, needed)
    identifiers(frame, ["compound_id", "endpoint", "context_id", "exposure"])
    frame = numeric(frame, ["relevance_score", "toxicity_probability"])
    if not needed <= set(frame):
        raise ValueError(f"Missing columns: {sorted(needed - set(frame))}")
    if frame.empty or frame.compound_id.duplicated().any():
        raise ValueError("Unique compounds and at least one candidate required")
    if any(frame[c].nunique() != 1 for c in ["endpoint", "context_id", "exposure"]):
        raise ValueError("Do not compare different endpoints, contexts or exposures on one frontier")
    scores = frame[["relevance_score", "toxicity_probability"]].to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Both scores must be finite and in [0,1]")
    flags = []
    for relevance, toxicity in scores:
        dominates = (
            (scores[:, 0] >= relevance)
            & (scores[:, 1] <= toxicity)
            & ((scores[:, 0] > relevance) | (scores[:, 1] < toxicity))
        )
        flags.append(not dominates.any())
    result = frame.copy()
    result["pareto_candidate"] = flags
    result["interpretation"] = "EXPLORATORY PRIORITIZATION; NOT VALIDATED EFFICACY OR SAFETY"
    return result
