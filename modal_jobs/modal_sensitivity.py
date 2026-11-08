"""Sensitivity and prevalence-adjusted analyses on Modal. Select project with env TOPIC=1|2.

Deploy + spawn with modal_jobs/submit_sensitivity.py (spawned calls survive the client exiting).
Topic 1 needs /inputs/{compound_context.csv,provenance.json,auc_lookup.csv} and /context_real_gdsc2_ctrp_r1 on the volume.
Topic 2 needs /uncertainty_research_sim_r1 on the volume.
"""
import json
import os
import subprocess

import modal

TOPIC = os.environ.get("TOPIC", "1")
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
app = modal.App(f"bioinfo-sens-topic{TOPIC}", image=image)
ENV = {"PYTHONPATH": "/app"}


def _cli(cmd):
    rc = subprocess.run(cmd, cwd="/app", env={**os.environ, **ENV}).returncode
    vol.commit()
    if rc:
        raise RuntimeError(f"workflow exited {rc}")


# ----------------------------------------------------------------- Topic 1
@app.function(volumes={"/data": vol}, cpu=4, memory=32768, timeout=6 * 3600)
def t1_cutoff(cutoff: float):
    import pandas as pd

    tag = f"{cutoff:.2f}"
    tab = pd.read_csv("/data/inputs/compound_context.csv")
    auc = pd.read_csv("/data/inputs/auc_lookup.csv")
    tab = tab.merge(auc, on="sample_id", how="left", validate="one_to_one")
    assert tab.auc.notna().all()
    tab["label"] = (tab.auc < cutoff).astype(int)
    tab["endpoint"] = f"growth_inhibition_auc_lt_{tag}"
    prev = tab.groupby("split_role").label.mean().round(4).to_dict()
    tab = tab.drop(columns="auc")
    os.makedirs("/tmp/in", exist_ok=True)
    table_path, prov_path = f"/tmp/in/table_{tag}.csv", f"/tmp/in/prov_{tag}.json"
    tab.to_csv(table_path, index=False)
    prov = json.load(open("/data/inputs/provenance.json"))
    prov["endpoint_definition"] = (
        f"SENSITIVITY ANALYSIS: label 1 if median benchmark-fitted dose-response AUC < {tag}; same rows and features as the "
        f"primary analysis (cutoff 0.7). Cutoff applied identically to both studies; prevalence {prev}. "
    )
    prov.pop("prepared_table_sha256", None)
    json.dump(prov, open(prov_path, "w"), indent=2)
    out = f"/data/sens_t1/cutoff_{tag}"
    _cli(["python", "-m", "topic1_cell_context.cli", "run", "--config", "/app/topic1_cell_context/configs/research.json",
          "--input", table_path, "--provenance", prov_path, "--output", out])
    return {"cutoff": cutoff, "prevalence": prev, "out": out}


@app.function(volumes={"/data": vol}, cpu=4, memory=48000, timeout=6 * 3600)
def t1_prevalence():
    """Prevalence-adjusted re-analysis of the primary run (predictions are not refit)."""
    import numpy as np
    import pandas as pd
    from scipy.optimize import brentq
    from scipy.special import expit, logit

    from research_common.metrics import rank_agreement, selection_regret

    src = "/data/context_real_gdsc2_ctrp_r1"
    out = "/data/sens_t1/prevalence"
    os.makedirs(out, exist_ok=True)
    scores = pd.read_csv(f"{src}/scores.csv")
    pred = pd.read_csv(f"{src}/predictions.csv", dtype={"protocol": "category", "model": "category", "partition": "category"})
    print("partitions:", pred.partition.unique().tolist(), flush=True)

    def ec(y, p, comp):
        d = pd.DataFrame({"c": comp, "e": (y - p) ** 2})
        return float(d.groupby("c", observed=True).e.mean().mean())

    rows = []
    for (protocol, seed, model, part), g in pred.groupby(["protocol", "seed", "model", "partition"], observed=True):
        if part not in ("internal", "external"):
            continue
        y, p, comp = g.label.to_numpy(float), np.clip(g.probability.to_numpy(float), 1e-9, 1 - 1e-9), g.compound_id.to_numpy()
        pi = float(y.mean())
        brier = ec(y, p, comp)
        ref = ec(y, np.full_like(p, pi), comp)
        lp = logit(p)
        delta = brentq(lambda d: expit(lp + d).mean() - pi, -30, 30)
        rows.append({"protocol": protocol, "seed": seed, "model": model, "partition": part, "prevalence": pi,
                     "mean_prediction": float(p.mean()), "brier_ec": brier, "climatology_brier_ec": ref,
                     "brier_skill": 1 - brier / ref, "intercept_shift": delta,
                     "brier_recalibrated_ec": ec(y, expit(lp + delta), comp)})
    long = pd.DataFrame(rows)
    long.to_csv(f"{out}/adjusted_metrics_long.csv", index=False)

    # validate against the stored scores: which column is the equal-compound Brier?
    chk = long[long.partition == "external"].merge(scores, on=["protocol", "seed", "model"])
    chk_info = {"max_abs_diff_vs_external_brier": float((chk.brier_ec - chk.external_brier).abs().max()),
                "max_abs_diff_vs_external_compound_brier": float((chk.brier_ec - chk.external_compound_brier).abs().max())}
    json.dump(chk_info, open(f"{out}/validation_against_scores.json", "w"), indent=2)
    print(chk_info, flush=True)

    wide = long.pivot_table(index=["protocol", "seed", "model"], columns="partition",
                            values=["brier_ec", "brier_skill", "brier_recalibrated_ec", "prevalence"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    wide = wide.merge(scores[["protocol", "seed", "model", "internal_roc_auc", "external_roc_auc"]], on=["protocol", "seed", "model"])
    wide.to_csv(f"{out}/adjusted_metrics_wide.csv", index=False)

    metrics = {"brier": ("brier_ec_internal", "brier_ec_external", False),
               "brier_skill": ("brier_skill_internal", "brier_skill_external", True),
               "brier_recalibrated": ("brier_recalibrated_ec_internal", "brier_recalibrated_ec_external", False),
               "auroc": ("internal_roc_auc", "external_roc_auc", True)}
    meta = []
    for (protocol, seed), g in wide.groupby(["protocol", "seed"]):
        for name, (ci, ce, hib) in metrics.items():
            a, b = g[ci].to_numpy(), g[ce].to_numpy()
            meta.append({"protocol": protocol, "seed": seed, "metric": name, "mean_abs_gap": float(np.abs(a - b).mean()),
                         "mean_signed_gap_ext_minus_int": float((b - a).mean()),
                         "rank_agreement": rank_agreement(a, b), "selection_regret": selection_regret(a, b, higher_is_better=hib),
                         "internal_best": "|".join(g.model[a == (a.max() if hib else a.min())]),
                         "external_best": "|".join(g.model[b == (b.max() if hib else b.min())])})
    pd.DataFrame(meta).to_csv(f"{out}/meta_evaluation_by_metric.csv", index=False)
    vol.commit()
    return chk_info


# ----------------------------------------------------------------- Topic 2
def _scen(name, mech="linear", ann=0.0, fn=0.0, ptm=0.0, donor=0.5):
    return {"name": name, "mechanism": mech, "annotation_error": ann, "false_negative_rate": fn,
            "ptm_missingness": ptm, "donor_effect": donor}


T2_BASE = {"seed": 29, "replicates": 100, "n_donors": 40, "reference_donors": 300, "rows_per_donor": 40,
           "bootstrap_draws": 2000, "protocols": ["random_row", "held_out_donor"],
           "models": ["unweighted_logistic", "confidence_weighted_logistic", "expression_only", "random_forest"],
           "save_unit_scores": False}
T2_CONFIGS = {
    "fn_sweep": {"scenarios": [_scen(f"fn_{x:.1f}", fn=x) for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]},
    "donor_sweep": {"scenarios": [_scen(f"donor_{x:.1f}", donor=x) for x in [0.5, 1.0, 1.5, 2.0, 3.0]]},
    "annotation_sweep": {"scenarios": [_scen(f"ann_{x:.1f}", ann=x) for x in [0.0, 0.1, 0.2, 0.3, 0.4]]},
    **{f"nsize_{n}": {"n_donors": n, "scenarios": [_scen("clean_linear"), _scen("donor_only", donor=2.0)]} for n in [20, 80, 160]},
}


def add_skill(scores, scenarios, n_ref, rows_per_donor):
    """Prevalence-adjusted skill: 1 - Brier / (pi(1-pi)), using the true reference mean risk for the reference
    and the observed test prevalence for the estimate."""
    from topic2_uncertainty.simulation import simulate

    by_name = {s["name"]: s for s in scenarios}
    pi = {}
    for (sc, rep, rs), _ in scores.groupby(["scenario", "replicate", "reference_seed"]):
        ref = simulate(int(rs), by_name[sc], n_ref, rows_per_donor, None, prefix="ORACLE")
        pi[(sc, rep)] = float(ref.truth_probability.mean())
    s = scores.copy()
    s["reference_mean_risk"] = [pi[(a, b)] for a, b in zip(s.scenario, s.replicate)]
    pr = s.reference_mean_risk
    s["skill_reference"] = 1 - s.reference_expected_brier / (pr * (1 - pr))
    po = s.observed_prevalence
    s["skill_estimate"] = 1 - s.estimated_brier / (po * (1 - po))
    s["skill_bias"] = s.skill_estimate - s.skill_reference
    s["prevalence_gap_observed_minus_true"] = po - pr
    return s


@app.function(volumes={"/data": vol}, cpu=2, memory=8192, timeout=12 * 3600)
def t2_sweep(name: str):
    from research_common.io import read_config  # noqa: F401  (ensures package import path works)

    cfg = {**T2_BASE, **T2_CONFIGS[name]}
    os.makedirs("/tmp/cfg", exist_ok=True)
    path = f"/tmp/cfg/{name}.json"
    json.dump(cfg, open(path, "w"), indent=2)
    out = f"/data/sens_t2/{name}"
    _cli(["python", "-m", "topic2_uncertainty.cli", "demo", "--config", path, "--output", out])
    import pandas as pd

    scores = pd.read_csv(f"{out}/scores.csv")
    add_skill(scores, cfg["scenarios"], cfg["reference_donors"], cfg["rows_per_donor"]).to_csv(f"{out}/scores_with_skill.csv", index=False)
    vol.commit()
    return out


@app.function(volumes={"/data": vol}, cpu=2, memory=8192, timeout=6 * 3600)
def t2_prevalence():
    """Skill-score re-analysis of the primary 200-replicate run."""
    import pandas as pd

    src = "/data/uncertainty_research_sim_r1"
    cfg = json.load(open("/app/topic2_uncertainty/configs/research.json"))
    scores = pd.read_csv(f"{src}/scores.csv")
    out = "/data/sens_t2/prevalence_primary"
    os.makedirs(out, exist_ok=True)
    add_skill(scores, cfg["scenarios"], cfg["reference_donors"], cfg["rows_per_donor"]).to_csv(f"{out}/scores_with_skill.csv", index=False)
    vol.commit()
    return out
