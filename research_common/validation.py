"""Explicit contracts shared by the public APIs and command-line tools."""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside allowed range [{minimum}, {maximum}]")
    return int(value)


def real(value, name, minimum=0.0, maximum=None, upper_open=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and (value >= maximum if upper_open else value > maximum)):
        raise ValueError(f"{name} outside allowed range")
    return float(value)


def mapping(value, name="config"):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")


def unique_choices(values, name, allowed=None, minimum=1):
    if not isinstance(values, (list, tuple)) or len(values) < minimum:
        raise ValueError(f"{name} needs at least {minimum} entries")
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{name} entries must be nonblank strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} entries must be unique; duplicate entries found")
    if allowed is not None and not set(values) <= set(allowed):
        raise ValueError(f"Unknown {name}: {sorted(set(values) - set(allowed))}")


def frame_contract(frame, columns, *, allow_empty=False):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError("Expected a DataFrame with unique column names")
    if any(not isinstance(c, str) for c in frame.columns):
        raise ValueError("Column names must be strings")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    if frame.empty and not allow_empty:
        raise ValueError("Input table must not be empty")


def identifiers(frame, columns):
    for name in columns:
        values = frame[name]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} must not contain missing or blank identifiers")
        if (values.astype(str) != values.astype(str).str.strip()).any():
            raise ValueError(f"{name} contains surrounding whitespace")


def numeric(frame, columns, *, allow_nan=False):
    result = frame.copy()
    try:
        for name in columns:
            result[name] = pd.to_numeric(result[name], errors="raise")
        values = result[list(columns)].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Expected numeric values in {list(columns)}") from exc
    if np.isinf(values).any() or (not allow_nan and np.isnan(values).any()):
        raise ValueError(f"Nonfinite values in {list(columns)}")
    return result
