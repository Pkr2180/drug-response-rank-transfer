"""Stable named random streams, independent of loop ordering and Python hash()."""

import hashlib
import json

import numpy as np

from .validation import integer


def derive_seed(seed, *labels):
    integer(seed, "seed", maximum=2**32 - 1)
    payload = json.dumps([int(seed), *labels], ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int(np.frombuffer(digest[:4], dtype="<u4")[0])
