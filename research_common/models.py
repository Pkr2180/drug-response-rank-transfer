"""Deliberately interpretable baselines and validation-weighted late fusion.

These are original small reference implementations, not reproductions of the
RankMap, MIXTAP or MolMMP architectures.
"""

import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .validation import unique_choices


def fit_checked(model, x, y, **kwargs):
    """Do not silently report scores from an unconverged fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, **kwargs)
    return model


def logistic_model(seed=0):
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
        ]
    )


def feature_columns(frame, prefixes):
    unique_choices(prefixes, "feature prefixes")
    for prefix in prefixes:
        if not any(c.startswith(prefix + "__") for c in frame.columns):
            raise ValueError(f"No features for {prefix}")
    return sorted(c for c in frame.columns if any(c.startswith(p + "__") for p in prefixes))


class LateFusion:
    """Convex mixture of modality probabilities; weights use development calibration only."""

    def __init__(self, prefixes, seed=0):
        self.prefixes, self.seed = prefixes, seed

    def fit(self, train, calibration):
        self.models = []
        for prefix in self.prefixes:
            cols = feature_columns(train, [prefix])
            model = logistic_model(self.seed)
            fit_checked(model, train[cols], train.label)
            self.models.append((cols, model))
        probabilities = self._matrix(calibration)
        y = calibration.label.to_numpy()
        unit_counts = calibration.groupby("compound_id").compound_id.transform("size").to_numpy()
        sample_weight = 1 / unit_counts
        k = probabilities.shape[1]

        def loss(weights):
            return np.average((y - probabilities @ weights) ** 2, weights=sample_weight)

        result = minimize(
            loss,
            np.full(k, 1 / k),
            method="SLSQP",
            bounds=[(0, 1)] * k,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success:
            raise RuntimeError(f"Fusion fit failed: {result.message}")
        self.weights = np.maximum(result.x, 0)
        self.weights /= self.weights.sum()
        self.fitted_ = True
        return self

    def _matrix(self, frame):
        return np.column_stack([m.predict_proba(frame[c])[:, 1] for c, m in self.models])

    def predict(self, frame):
        if not getattr(self, "fitted_", False):
            raise NotFittedError("Fit LateFusion on separate training and calibration sets first")
        return np.clip(self._matrix(frame) @ self.weights, 0, 1)
