"""Build manuscript tables/figures from Modal results. Usage: python modal_jobs/make_report.py"""
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
COL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]


def save(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def ci(s):
    """Mean and 95% half-width across seeds (descriptive; splits share observations)."""
    s = s.dropna()
    h = 1.96 * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
    return s.mean(), h


def topic1():
    src = ROOT / "outputs/modal_t1/context_real_gdsc2_ctrp_r1"
    out = ROOT / "topic1_cell_context/results/modal_research_r1"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    for f in ["scores.csv", "meta_evaluation.csv", "fusion_weights.csv", "run_manifest.json",
              "interpretation.json", "warnings.json", "external_validity.png", "external_validity.svg"]:
        shutil.copy2(src / f, out / f)
    s = pd.read_csv(src / "scores.csv")
    m = pd.read_csv(src / "meta_evaluation.csv")
    w = pd.read_csv(src / "fusion_weights.csv")
    models = ["molecular_only", "context_only", "early_fusion", "without_ptm", "late_fusion"]
    prots = ["random_compound", "scaffold", "chemical_cluster"]

    rows = []
    for (p, mo), g in s.groupby(["protocol", "model"]):
        r = {"protocol": p, "model": mo}
        for k in ["internal_compound_brier", "external_compound_brier", "internal_roc_auc", "external_roc_auc",
                  "internal_average_precision", "external_average_precision", "absolute_brier_gap"]:
            mu, h = ci(g[k])
            r[k] = f"{mu:.3f} ± {h:.3f}"
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "tables/table1_performance_mean_ci_over_seeds.csv", index=False)

    rw = s.groupby(["protocol", "model"])[["internal_brier", "external_brier", "internal_compound_brier", "external_compound_brier"]].mean().round(4)
    rw.rename(columns={"internal_brier": "internal_row_weighted_brier", "external_brier": "external_row_weighted_brier"}).to_csv(out / "tables/tableS_row_weighted_vs_equal_compound_brier.csv")
    t2 = m.groupby("protocol").agg(
        rank_agreement_mean=("rank_agreement", "mean"), rank_agreement_sd=("rank_agreement", "std"),
        selection_regret_mean=("selection_regret", "mean"), selection_regret_max=("selection_regret", "max"),
        zero_regret_splits=("selection_regret", lambda x: int((x == 0).sum())),
        abs_brier_gap_mean=("mean_absolute_brier_gap", "mean"), splits=("seed", "size")).reindex(prots)
    t2.round(4).to_csv(out / "tables/table2_meta_evaluation.csv")

    piv = s.pivot_table(index=["protocol", "seed"], columns="model", values="external_compound_brier")
    rows = []
    for p in prots:
        for mo in models[:-1]:
            d = piv.loc[p][mo] - piv.loc[p]["late_fusion"]
            mu, h = ci(d)
            rows.append({"protocol": p, "comparator": mo, "mean_external_brier_minus_late_fusion": round(mu, 4),
                         "ci95_halfwidth": round(h, 4), "splits_late_fusion_better": int((d > 0).sum()), "n": len(d)})
    t3 = pd.DataFrame(rows)
    t3.to_csv(out / "tables/table3_paired_external_brier.csv", index=False)

    sel = m.groupby(["protocol", "internally_selected_model"]).size().rename("n_splits").reset_index()
    sel.to_csv(out / "tables/table4_internal_selection_frequency.csv", index=False)
    w.groupby(["protocol", "modality"]).weight.agg(["mean", "std"]).round(3).to_csv(out / "tables/table5_fusion_weights.csv")

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for a, p in zip(ax, prots):
        for i, mo in enumerate(models):
            g = s[(s.protocol == p) & (s.model == mo)]
            a.scatter(g.internal_compound_brier, g.external_compound_brier, s=14, color=COL[i], alpha=.8, label=mo.replace("_", " "))
        a.plot([.08, .2], [.08, .2], "k:", lw=.8)
        a.set_title(p.replace("_", " "))
        a.set_xlabel("Internal Brier loss")
    ax[0].set_ylabel("External Brier loss (CTRPv2)")
    ax[2].legend(frameon=False, fontsize=7)
    save(fig, out / "figures/fig1_internal_vs_external_brier")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    wd = .25
    for j, p in enumerate(prots):
        sub = [s[(s.protocol == p) & (s.model == mo)].external_compound_brier for mo in models]
        ax.bar(np.arange(5) + (j - 1) * wd, [x.mean() for x in sub], wd, yerr=[ci(x)[1] for x in sub],
               color=COL[j], label=p.replace("_", " "), capsize=2)
    ax.set_xticks(range(5))
    ax.set_xticklabels([x.replace("_", "\n") for x in models])
    ax.set_ylim(.08, .11)
    ax.set_ylabel("External Brier loss")
    ax.legend(frameon=False, fontsize=7)
    save(fig, out / "figures/fig2_external_brier_by_model")

    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for a, k, lab in zip(ax, ["rank_agreement", "selection_regret"],
                         ["Rank agreement (internal vs external)", "Selection regret (Brier)"]):
        data = [m[m.protocol == p][k] for p in prots]
        a.boxplot(data, widths=.5, showfliers=False)
        for i, d in enumerate(data):
            a.scatter(np.random.default_rng(1).normal(i + 1, .05, len(d)), d, s=12, color=COL[i], zorder=3)
        a.set_xticks([1, 2, 3])
        a.set_xticklabels([p.replace("_", "\n") for p in prots])
        a.set_ylabel(lab)
    ax[0].axhline(0, color="k", lw=.6, ls=":")
    save(fig, out / "figures/fig3_meta_evaluation")

    fig, ax = plt.subplots(figsize=(5, 3))
    ww = w.groupby(["protocol", "modality"]).weight.mean().unstack().reindex(prots)
    ww.plot.bar(stacked=True, ax=ax, color=COL[:3], rot=0)
    ax.set_ylabel("Mean late-fusion weight")
    ax.legend(frameon=False, fontsize=7)
    ax.set_xticklabels([p.replace("_", "\n") for p in prots])
    save(fig, out / "figures/fig4_fusion_weights")
    print(t2.round(3), t3, ww.round(3), sel.pivot(index="internally_selected_model", columns="protocol", values="n_splits"), sep="\n")


def topic2():
    src = ROOT / "outputs/modal_t2/uncertainty_research_sim_r1"
    out = ROOT / "topic2_uncertainty/results/modal_research_r1"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    for f in ["summary.csv", "paired_comparisons.csv", "meta_evaluation.csv", "scores.csv", "run_manifest.json",
              "interpretation.json", "warnings.json", "simulation_audit.png", "simulation_audit.svg"]:
        shutil.copy2(src / f, out / f)
    sm = pd.read_csv(src / "summary.csv")
    me = pd.read_csv(src / "meta_evaluation.csv")
    pc = pd.read_csv(src / "paired_comparisons.csv")
    sc = ["clean_linear", "annotation_only", "negative_label_only", "missing_ptm_only", "donor_only", "combined_nonlinear"]
    prots = ["random_row", "held_out_donor"]
    models = sorted(sm.model.unique())
    idx = pd.MultiIndex.from_product([sc, prots])

    t1 = sm.copy()
    t1["containment"] = t1.apply(lambda r: f"{r.reference_containment_rate:.3f} [{r.reference_containment_low:.3f}, {r.reference_containment_high:.3f}]", axis=1)
    t1[["scenario", "protocol", "model", "replicates", "mean_bias", "bias_mcse", "mean_absolute_error", "rmse", "containment"]].round(5).to_csv(out / "tables/table1_estimator_accuracy.csv", index=False)
    t2 = sm.groupby(["scenario", "protocol"]).agg(mean_bias=("mean_bias", "mean"), mae=("mean_absolute_error", "mean"),
                                                   rmse=("rmse", "mean"), containment=("reference_containment_rate", "mean")).round(4).reindex(idx)
    t2.to_csv(out / "tables/table2_scenario_protocol_summary.csv")
    t3 = me.groupby(["scenario", "protocol"]).agg(rank_agreement=("rank_agreement", "mean"), selection_regret=("selection_regret", "mean"),
                                                   best_selected_probability=("selected_best_probability", "mean")).round(4).reindex(idx)
    t3.to_csv(out / "tables/table3_meta_evaluation.csv")
    pc["z"] = pc.mean_absolute_error_difference_a_minus_b / pc.paired_mcse
    pc.round(5).to_csv(out / "tables/table4_paired_model_comparisons.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for a, p in zip(ax, prots):
        for i, mo in enumerate(models):
            g = sm[(sm.protocol == p) & (sm.model == mo)].set_index("scenario").reindex(sc)
            a.errorbar(np.arange(6) + (i - 1.5) * .12, g.mean_bias, 1.96 * g.bias_mcse, fmt="o", ms=3.5,
                       color=COL[i], label=mo.replace("_", " "), capsize=1.5)
        a.axhline(0, color="k", lw=.6, ls=":")
        a.set_xticks(range(6))
        a.set_xticklabels([x.replace("_", "\n") for x in sc], fontsize=6.5)
        a.set_title(p.replace("_", " "))
    ax[0].set_ylabel("Mean bias of Brier estimate (est - reference)")
    ax[1].legend(frameon=False, fontsize=6.5)
    save(fig, out / "figures/fig1_bias_by_scenario")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for j, p in enumerate(prots):
        g = sm[sm.protocol == p].groupby("scenario").reference_containment_rate.mean().reindex(sc)
        ax.plot(np.arange(6) + (j - .5) * .1, g, "o-", color=COL[j], label=p.replace("_", " "), ms=4)
    ax.axhline(.95, color="k", ls=":", lw=.7)
    ax.set_ylim(.7, 1)
    ax.set_xticks(range(6))
    ax.set_xticklabels([x.replace("_", "\n") for x in sc], fontsize=7)
    ax.set_ylabel("Interval containment of reference risk")
    ax.legend(frameon=False, fontsize=7)
    save(fig, out / "figures/fig2_containment")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for a, k, lab in zip(ax, ["rank_agreement", "selection_regret"], ["Mean rank agreement", "Mean selection regret"]):
        for j, p in enumerate(prots):
            g = me[me.protocol == p].groupby("scenario")[k].mean().reindex(sc)
            a.plot(range(6), g, "o-", color=COL[j], label=p.replace("_", " "), ms=4)
        a.set_xticks(range(6))
        a.set_xticklabels([x.replace("_", "\n") for x in sc], fontsize=6.5)
        a.set_ylabel(lab)
    ax[0].legend(frameon=False, fontsize=7)
    save(fig, out / "figures/fig3_meta_evaluation")

    print(t2, t3, sep="\n")
    print(sm.groupby("protocol")[["mean_bias", "mean_absolute_error", "reference_containment_rate"]].mean().round(4))
    print(pc.reindex(pc.z.abs().sort_values(ascending=False).index).head(6)[["scenario", "protocol", "model_a", "model_b", "mean_absolute_error_difference_a_minus_b", "z"]])


if __name__ == "__main__":
    topic1()
    topic2()
