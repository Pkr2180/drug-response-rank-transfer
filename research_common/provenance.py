"""Require recorded provenance; assertions do not certify source truthfulness."""

from .validation import mapping


def validate_provenance(record, workflow="external"):
    mapping(record, "provenance")
    if workflow not in {"external", "plasmode"}:
        raise ValueError("Unknown provenance workflow")
    required = [
        "dataset_sources",
        "license_review",
        "endpoint_definition",
        "feature_construction",
        "independent_unit",
    ]
    required += ["external_cohort_definition"] if workflow == "external" else ["source_population_definition"]
    for key in required:
        value = record.get(key)
        if not value or (
            isinstance(value, str)
            and (not value.strip() or any(t in value.upper() for t in ["TODO", "REPLACE"]))
        ):
            raise ValueError(f"Complete provenance field: {key}")
        if key != "dataset_sources" and not isinstance(value, str):
            raise ValueError(f"Provenance {key} must be descriptive text")
    if not isinstance(record["dataset_sources"], list):
        raise ValueError("dataset_sources must be a nonempty list of objects")
    for source in record["dataset_sources"]:
        mapping(source, "dataset source")
        if not all(source.get(k) for k in ["accession_or_url", "version_or_retrieval_date", "sha256"]):
            raise ValueError("Each dataset needs source, version/date and SHA256")
        checksum = source["sha256"]
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum.lower())
        ):
            raise ValueError("Dataset SHA256 must be a 64-character hex digest")
    if record.get("outcome_derived_features") is not False:
        raise ValueError("Outcome-derived features prohibited; declare outcome_derived_features=false")
    if workflow == "external" and record.get("external_used_for_tuning") is not False:
        raise ValueError("Declare external_used_for_tuning=false before running the audit")
