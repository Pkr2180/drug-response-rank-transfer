"""Single reproducible group splits; never search over held-out outcomes."""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .validation import frame_contract, identifiers, integer, real


def group_split(frame, group_column, seed, test_size=0.25, label_column="label"):
    frame_contract(frame, [group_column, label_column])
    identifiers(frame, [group_column])
    integer(seed, "seed", maximum=2**32 - 1)
    real(test_size, "test_size", minimum=np.finfo(float).eps, maximum=1.0, upper_open=True)
    if not frame[label_column].isin([0, 1]).all() or frame[group_column].nunique() < 2:
        raise ValueError("Binary labels and at least two split groups required")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(frame, groups=frame[group_column]))
    if frame.iloc[train][label_column].nunique() < 2:
        raise ValueError(f"Training split for {group_column} has one class; revise the prespecified design")
    return frame.iloc[train].copy(), frame.iloc[test].copy()


def assert_disjoint(a, b, columns):
    for column in columns:
        frame_contract(a, [column])
        frame_contract(b, [column])
        identifiers(a, [column])
        identifiers(b, [column])
        common = set(a[column]) & set(b[column])
        if common:
            raise ValueError(f"Leakage: {len(common)} overlapping {column} values")
