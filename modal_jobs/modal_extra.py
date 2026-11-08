"""Additional replication analyses on Modal (account yadalam1980).

Topic 1: (a) 40 further split seeds on the primary table; (b) replication cohorts: reverse direction CTRPv2->GDSCv2 and GDSCv2->GDSCv1.
Topic 2: (a) false-negative and donor-effect sweeps under the interaction and PTM-dominant mechanisms; (b) +100 replicates of the linear sweeps (new seed).
Deploy + spawn with modal_jobs/submit_extra.py.
"""
import json
import os
import subprocess

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.2.5", "pandas==2.3.1", "scipy==1.15.2", "scikit-learn==1.8.0", "matplotlib==3.10.1")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(f"{ROOT}/research_common", "/app/research_common", ignore=["__pycache__"])
    .add_local_dir(f"{ROOT}/topic1_cell_context", "/app/topic1_cell_context", ignore=["__pycache__", "results"])
    .add_local_dir(f"{ROOT}/topic2_uncertainty", "/app/topic2_uncertainty", ignore=["__pycache__", "results"])
)
vol = modal.Volume.from_name("bioinfo-results", create_if_missing=True)
app = modal.App("bioinfo-extra", image=image)


def _cli(cmd):
    rc = subprocess.run(cmd, cwd="/app", env={**os.environ, "PYTHONPATH": "/app"}).returncode
    vol.commit()
    if rc:
        raise RuntimeError(f"workflow exited {rc}")


@app.function(volumes={"/data": vol}, cpu=4, memory=32768, timeout=8 * 3600)
def t1_run(table: str, provenance: str, out_name: str, seeds: list | None = None):
    vol.reload()  # pick up files uploaded after this container started
    assert os.path.exists(table), f"missing input {table}"
    cfg = json.load(open("/app/topic1_cell_context/configs/research.json"))
    if seeds:
        cfg["seeds"] = seeds
    os.makedirs("/tmp/cfg", exist_ok=True)
    path = f"/tmp/cfg/{out_name}.json"
    json.dump(cfg, open(path, "w"))
    out = f"/data/extra_t1/{out_name}"
    _cli(["python", "-m", "topic1_cell_context.cli", "run", "--config", path, "--input", table, "--provenance", provenance, "--output", out])
    return out


def _scen(name, mech="linear", ann=0.0, fn=0.0, ptm=0.0, donor=0.5):
    return {"name": name, "mechanism": mech, "annotation_error": ann, "false_negative_rate": fn, "ptm_missingness": ptm, "donor_effect": donor}


BASE = {"seed": 29, "replicates": 100, "n_donors": 40, "reference_donors": 300, "rows_per_donor": 40, "bootstrap_draws": 2000,
        "protocols": ["random_row", "held_out_donor"], "models": ["unweighted_logistic", "confidence_weighted_logistic", "expression_only", "random_forest"],
        "save_unit_scores": False}
FN = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DN = [0.5, 1.0, 1.5, 2.0, 3.0]
T2 = {
    "fn_interaction": {"scenarios": [_scen(f"fn_{x:.1f}", "interaction", fn=x) for x in FN]},
    "fn_ptm_dominant": {"scenarios": [_scen(f"fn_{x:.1f}", "ptm_dominant", fn=x) for x in FN]},
    "donor_interaction": {"scenarios": [_scen(f"donor_{x:.1f}", "interaction", donor=x) for x in DN]},
    "donor_ptm_dominant": {"scenarios": [_scen(f"donor_{x:.1f}", "ptm_dominant", donor=x) for x in DN]},
    "fn_linear_b": {"seed": 31, "scenarios": [_scen(f"fn_{x:.1f}", fn=x) for x in FN]},
    "donor_linear_b": {"seed": 31, "scenarios": [_scen(f"donor_{x:.1f}", donor=x) for x in DN]},
}


def add_skill(scores, scenarios, n_ref, rows_per_donor):
    from topic2_uncertainty.simulation import simulate

    by_name = {s["name"]: s for s in scenarios}
    pi = {}
    for (sc, rep, rs), _ in scores.groupby(["scenario", "replicate", "reference_seed"]):
        pi[(sc, rep)] = float(simulate(int(rs), by_name[sc], n_ref, rows_per_donor, None, prefix="ORACLE").truth_probability.mean())
    s = scores.copy()
    s["reference_mean_risk"] = [pi[(a, b)] for a, b in zip(s.scenario, s.replicate)]
    pr = s.reference_mean_risk
    s["skill_reference"] = 1 - s.reference_expected_brier / (pr * (1 - pr))
    po = s.observed_prevalence
    s["skill_estimate"] = 1 - s.estimated_brier / (po * (1 - po))
    s["skill_bias"] = s.skill_estimate - s.skill_reference
    return s


@app.function(volumes={"/data": vol}, cpu=2, memory=8192, timeout=12 * 3600)
def t2_run(name: str):
    import pandas as pd

    cfg = {**BASE, **T2[name]}
    os.makedirs("/tmp/cfg", exist_ok=True)
    path = f"/tmp/cfg/{name}.json"
    json.dump(cfg, open(path, "w"))
    out = f"/data/extra_t2/{name}"
    _cli(["python", "-m", "topic2_uncertainty.cli", "demo", "--config", path, "--output", out])
    add_skill(pd.read_csv(f"{out}/scores.csv"), cfg["scenarios"], cfg["reference_donors"], cfg["rows_per_donor"]).to_csv(f"{out}/scores_with_skill.csv", index=False)
    vol.commit()
    return out
