"""Topic 1: re-score the external cohort separately on cell lines that were / were not present in the development cohort.
No model is refitted: stored external predictions of the primary run are subset by cell line.
Usage: python modal_jobs/analyze_cellline_novel.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research_common.metrics import rank_agreement, selection_regret

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/modal_t1/context_real_gdsc2_ctrp_r1"
OUT = ROOT / "topic1_cell_context/results/modal_research_r1/sensitivity"
TABLE = "F:/bioinfo_data/topic1_prepared/compound_context.csv"
OUT.mkdir(parents=True, exist_ok=True)

tab = pd.read_csv(TABLE, usecols=["study_id", "context_id", "compound_id", "sample_id"])
dev_lines = set(tab[tab.study_id == "GDSCv2"].context_id)
ext = tab[tab.study_id == "CTRPv2"]
line_of = dict(zip(ext.sample_id, ext.context_id))
novel_flag = {s: (c not in dev_lines) for s, c in line_of.items()}
summary = {
    "development_lines": len(dev_lines), "external_lines": ext.context_id.nunique(),
    "external_lines_shared_with_development": int(ext.context_id.isin(dev_lines).groupby(ext.context_id).first().sum()),
    "external_rows": len(ext), "external_rows_novel_lines": int(sum(novel_flag.values())),
}
novel_rows = ext[~ext.context_id.isin(dev_lines)]
summary.update({"novel_line_count": int(novel_rows.context_id.nunique()), "novel_line_compounds": int(novel_rows.compound_id.nunique()),
                "shared_line_rows": int(len(ext) - len(novel_rows))})

pred = pd.read_csv(SRC / "predictions.csv", usecols=["protocol", "seed", "model", "partition", "sample_id", "compound_id", "label", "probability"],
                   dtype={"protocol": "category", "model": "category", "partition": "category", "compound_id": "category"})
pred = pred[pred.partition == "external"].copy()
pred["novel"] = pred.sample_id.map(novel_flag)
assert pred.novel.notna().all()
pred["err"] = (pred.label - pred.probability) ** 2


def ec(g):
    return float(g.groupby("compound_id", observed=True).err.mean().mean())


rows = []
for (protocol, seed, model), g in pred.groupby(["protocol", "seed", "model"], observed=True):
    for name, sub in [("all", g), ("shared_lines", g[~g.novel]), ("novel_lines", g[g.novel])]:
        auc = roc_auc_score(sub.label, sub.probability) if sub.label.nunique() == 2 else np.nan
        pi = float(sub.label.mean())
        ref = float(((sub.label - pi) ** 2).groupby(sub.compound_id, observed=True).mean().mean())
        rows.append({"protocol": protocol, "seed": seed, "model": model, "subset": name, "n": len(sub), "compounds": sub.compound_id.nunique(),
                     "prevalence": pi, "brier_ec": ec(sub), "climatology_ec": ref, "auroc": auc})
long = pd.DataFrame(rows)
long["brier_skill"] = 1 - long.brier_ec / long.climatology_ec
long.to_csv(OUT / "cellline_subset_scores_long.csv", index=False)

scores = pd.read_csv(SRC / "scores.csv")
meta = []
for (protocol, seed), g in long[long.subset != "all"].groupby(["protocol", "seed"]):
    internal = scores[(scores.protocol == protocol) & (scores.seed == seed)].set_index("model").internal_compound_brier
    for name, h in g.groupby("subset"):
        h = h.set_index("model").loc[internal.index]
        a, b = internal.to_numpy(), h.brier_ec.to_numpy()
        meta.append({"protocol": protocol, "seed": seed, "external_subset": name, "mean_abs_gap": float(np.abs(a - b).mean()),
                     "rank_agreement": rank_agreement(a, b), "selection_regret": selection_regret(a, b)})
meta = pd.DataFrame(meta)
meta.to_csv(OUT / "cellline_subset_meta_evaluation.csv", index=False)

by_model = long.groupby(["subset", "model"])[["prevalence", "brier_ec", "climatology_ec", "brier_skill", "auroc"]].mean().round(4)
by_model.to_csv(OUT / "tables/tableS5_cellline_subset_by_model.csv")
agg = meta.groupby(["external_subset", "protocol"]).agg(mean_abs_gap=("mean_abs_gap", "mean"), rank_agreement_mean=("rank_agreement", "mean"),
                                                       rank_agreement_sd=("rank_agreement", "std"), selection_regret_mean=("selection_regret", "mean"),
                                                       zero_regret_splits=("selection_regret", lambda x: int((x == 0).sum()))).round(4)
agg.to_csv(OUT / "tables/tableS6_cellline_subset_meta_evaluation.csv")
json.dump(summary, open(OUT / "cellline_overlap_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print(by_model.to_string())
print(agg.to_string())
