"""Explicit estimands, cluster bootstrap intervals and tie-aware model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .validation import integer


def paired_vectors(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("Expected nonempty one-dimensional vectors of equal length")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Vectors must contain finite values")
    return a, b


def binary_scores(y, probability):
    y, p = paired_vectors(y, probability)
    if not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0,1]")
    both = len(np.unique(y)) == 2
    return {
        "n": len(y),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if both else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if both else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def unit_losses(values, groups):
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    if values.ndim != 1 or groups.ndim != 1 or values.shape != groups.shape or not values.size:
        raise ValueError("Finite losses and matching nonempty group vectors required")
    if not np.isfinite(values).all() or pd.isna(groups).any():
        raise ValueError("Losses and group identifiers must not be missing or nonfinite")
    return pd.DataFrame({"group": groups, "value": values}).groupby("group", sort=True).value.mean()


def cluster_mean_interval(values, groups, seed=0, draws=500, *, resample_groups=None):
    """Equal-unit mean with a 95% percentile cluster bootstrap interval.

    groups identifies equally weighted units (compounds or simulated donors).
    Optional resample_groups identifies a coarser dependency block; units must
    nest in blocks. Resampling preserves the equal-unit estimand via a ratio of
    unit-loss sums to unit counts. Intervals condition on the fitted predictor.
    """
    integer(draws, "draws", minimum=50)
    integer(seed, "seed", maximum=2**32 - 1)
    means = unit_losses(values, groups)
    if resample_groups is None:
        sums, counts = means.to_numpy(), np.ones(len(means))
    else:
        blocks = np.asarray(resample_groups)
        if blocks.ndim != 1 or len(blocks) != len(groups) or pd.isna(blocks).any():
            raise ValueError("Matching nonmissing resample_groups required")
        assignments = pd.DataFrame({"group": groups, "block": blocks})
        if assignments.groupby("group").block.nunique().gt(1).any():
            raise ValueError("Each evaluation unit must nest in one resampling block")
        by_unit = assignments.drop_duplicates("group").set_index("group").join(means)
        stats = by_unit.groupby("block").value.agg(["sum", "count"])
        sums, counts = stats["sum"].to_numpy(), stats["count"].to_numpy()
    if len(sums) < 2:
        raise ValueError("At least two independent resampling groups needed for an interval")
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    batch = max(1, min(256, 1_000_000 // len(sums)))
    for start in range(0, draws, batch):
        stop = min(draws, start + batch)
        index = rng.integers(0, len(sums), size=(stop - start, len(sums)))
        boot[start:stop] = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(means.mean()), float(low), float(high)


def rank_agreement(a, b):
    a, b = paired_vectors(a, b)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def selected_indices(estimated, higher_is_better=False):
    estimated, _ = paired_vectors(estimated, estimated)
    optimum = estimated.max() if higher_is_better else estimated.min()
    return np.flatnonzero(estimated == optimum)


def selection_regret(estimated, reference, higher_is_better=False):
    """Expected regret with uniform selection among exact internal-score ties."""
    estimated, reference = paired_vectors(estimated, reference)
    selected = reference[selected_indices(estimated, higher_is_better)].mean()
    return float(reference.max() - selected if higher_is_better else selected - reference.min())


def binomial_interval(successes, n, z=1.959963984540054):
    """Wilson interval, including nondegenerate bounds for all/zero successes."""
    integer(n, "n", minimum=1)
    integer(successes, "successes", maximum=n)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)
