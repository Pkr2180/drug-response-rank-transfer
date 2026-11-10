"""Paper 1: find splits in which two models are numerically identical (late fusion collapsed onto one block), so that
Spearman rank agreement orders them by floating-point rounding. Compares the stored rank agreement with a tie-aware value
(losses rounded to 12 decimals) for every run reported in the manuscript and writes
sensitivity/tables/tableS_numerical_ties.csv (per split) and tableS_numerical_ties_summary.csv (per run and design).
Usage (from bioinformatics_method_projects/): python modal_jobs/tie_check_t1.py"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "topic1_cell_context/results/modal_research_r1/sensitivity/tables"
RUNS = {"primary": ROOT / "topic1_cell_context/results/modal_research_r1",
        **{f"seeds_{k}": ROOT / f"outputs/extra_t1/seeds_{k}" for k in "ABCD"},
        **{f"cutoff_{c}": ROOT / f"outputs/sens_t1/sens_t1/cutoff_{c}" for c in ("0.50", "0.60", "0.80", "0.90")},
        "rev_ctrp_gdsc2": ROOT / "outputs/extra_t1/rev_ctrp_gdsc2", "gdsc2_gdsc1": ROOT / "outputs/extra_t1/gdsc2_gdsc1"}
rows, summ = [], []
for run, d in RUNS.items():
    s, m = pd.read_csv(d / "scores.csv"), pd.read_csv(d / "meta_evaluation.csv")
    for (p, sd), g in s.groupby(["protocol", "seed"]):
        a, b = g.internal_compound_brier.round(12).to_numpy(), g.external_compound_brier.round(12).to_numpy()
        tie_aware = spearmanr(a, b).statistic
        stored = m[(m.protocol == p) & (m.seed == sd)].rank_agreement.iloc[0]
        rows.append({"run": run, "protocol": p, "seed": sd, "stored": stored, "tie_aware": tie_aware, "diff": tie_aware - stored})
t = pd.DataFrame(rows)
t.to_csv(OUT / "tableS_numerical_ties.csv", index=False)
sm = t.groupby(["run", "protocol"]).agg(stored_mean=("stored", "mean"), tie_aware_mean=("tie_aware", "mean"), splits=("seed", "size"),
                                        affected=("diff", lambda x: int((x.abs() > 1e-9).sum()))).reset_index()
sm["mean_shift"] = sm.tie_aware_mean - sm.stored_mean
sm.to_csv(OUT / "tableS_numerical_ties_summary.csv", index=False)
aff = t[t["diff"].abs() > 1e-9]
print(f"{len(aff)} of {len(t)} splits affected; max split change {aff['diff'].abs().max():.3f}; max design-mean change {sm.mean_shift.abs().max():.4f}")
print(aff.round(3).to_string())
