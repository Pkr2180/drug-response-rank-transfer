"""Fail-closed output creation and portable run manifests."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def read_config(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    with Path(path).open(encoding="utf-8-sig") as handle:
        result = json.load(handle, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def read_table(path):
    """Preserve identifier strings (including leading zeros and the literal NA)."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or len(set(header)) != len(header) or any(not h.strip() for h in header):
        raise ValueError("CSV requires unique nonblank column names")
    ids = {
        c: str
        for c in header
        if c.endswith("_id")
        or c
        in {
            "source",
            "target",
            "structure_key",
            "split_role",
            "endpoint",
            "data_origin",
            "evidence_tier",
            "smiles",
            "exposure",
        }
    }
    return pd.read_csv(path, dtype=ids, keep_default_na=False, encoding="utf-8-sig")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def manifest(output, config, inputs, status, extra=None):
    root = Path(__file__).resolve().parents[1]
    code = {}
    for folder in ["research_common", "topic1_cell_context", "topic2_uncertainty"]:
        for p in sorted((root / folder).rglob("*")):
            if p.suffix in {".py", ".json"}:
                code[str(p.relative_to(root)).replace("\\", "/")] = sha256(p)
    if (root / "pyproject.toml").is_file():
        code["pyproject.toml"] = sha256(root / "pyproject.toml")
    versions = {}
    for package in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pytest",
        "rdkit",
        "joblib",
        "threadpoolctl",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    record = {
        "manifest_version": 2,
        "software_version": "0.2.0",
        "command": sys.argv,
        "status": status,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "config": config,
        "input_sha256": {str(p): sha256(p) for p in inputs},
        "code_sha256": code,
        "notes": extra or {},
    }
    write_json(Path(output) / "run_manifest.json", record)
    return record


def write_table(path, table):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        table.to_csv(handle, index=False, float_format="%.17g")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def managed_run(output, config, inputs, status, provenance=None):
    """Exclusive run directory, frozen input/code hashes, atomic final status."""
    output = new_output(output)
    record = manifest(output, config, inputs, "RUNNING_" + status, provenance)
    record["started_at_utc"] = record["recorded_at_utc"]
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        try:
            yield output
            root = Path(__file__).resolve().parents[1]
            if any(sha256(p) != digest for p, digest in record["input_sha256"].items()):
                raise RuntimeError("Input files changed during the run")
            if any(sha256(root / p) != digest for p, digest in record["code_sha256"].items()):
                raise RuntimeError("Source files changed during the run")
            record["status"] = "COMPLETED_" + status
        except BaseException as exc:
            record["status"] = "FAILED_" + status
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            (output / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            write_json(
                output / "warnings.json",
                [{"category": w.category.__name__, "message": str(w.message)} for w in observed],
            )
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            record["elapsed_seconds"] = time.monotonic() - started
            record["output_sha256"] = {
                p.name: sha256(p)
                for p in sorted(output.iterdir())
                if p.is_file() and p.name != "run_manifest.json"
            }
            write_json(output / "run_manifest.json", record)
