"""Tables and figures for the sensitivity / prevalence-adjusted analyses.
Usage: python modal_jobs/make_sensitivity_report.py [1|2|all]"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
PROTS1 = ["random_compound", "scaffold", "chemical_cluster"]
MODELS1 = ["molecular_only", "context_only", "early_fusion", "without_ptm", "late_fusion"]


def save(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def topic1():
    b = ROOT / "outputs/sens_t1/sens_t1"
    out = ROOT / "topic1_cell_context/results/modal_research_r1/sensitivity"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    cutoffs = ["0.50", "0.60", "0.70", "0.80", "0.90"]
    rows, ext = [], []
    for c in cutoffs:
        me = pd.read_csv(b / f"cutoff_{c}/meta_evaluation.csv")
        sc = pd.read_csv(b / f"cutoff_{c}/scores.csv")
        for p in PROTS1:
            g, s = me[me.protocol == p], sc[sc.protocol == p]
            best = s.groupby("model").external_compound_brier.mean()
            rows.append({"auc_cutoff": float(c), "protocol": p, "internal_prevalence": s.internal_prevalence.mean(),
                         "external_prevalence": s.external_prevalence.mean(), "rank_agreement_mean": g.rank_agreement.mean(),
                         "rank_agreement_sd": g.rank_agreement.std(), "selection_regret_mean": g.selection_regret.mean(),
                         "zero_regret_splits": int((g.selection_regret == 0).sum()), "abs_brier_gap_mean": g.mean_absolute_brier_gap.mean(),
                         "best_external_model_by_mean_brier": best.idxmin(), "late_fusion_external_brier": best["late_fusion"],
                         "external_auroc_mean": s.external_roc_auc.mean()})
        for mo, g in sc.groupby("model"):
            ext.append({"auc_cutoff": float(c), "model": mo, "external_compound_brier": g.external_compound_brier.mean(), "external_auroc": g.external_roc_auc.mean(),
                        "external_prevalence": g.external_prevalence.mean()})
    t1 = pd.DataFrame(rows)
    t1.round(4).to_csv(out / "tables/tableS1_cutoff_sensitivity.csv", index=False)
    ext = pd.DataFrame(ext)
    ext.round(4).to_csv(out / "tables/tableS2_external_by_model_and_cutoff.csv", index=False)

    m = pd.read_csv(b / "prevalence/meta_evaluation_by_metric.csv")
    t3 = m.groupby(["metric", "protocol"]).agg(mean_abs_gap=("mean_abs_gap", "mean"), mean_signed_gap_ext_minus_int=("mean_signed_gap_ext_minus_int", "mean"),
        rank_agreement_mean=("rank_agreement", "mean"), rank_agreement_sd=("rank_agreement", "std"), selection_regret_mean=("selection_regret", "mean"),
        zero_regret_splits=("selection_regret", lambda x: int((x == 0).sum()))).round(4)
    t3.to_csv(out / "tables/tableS3_prevalence_adjusted_meta_evaluation.csv")
    w = pd.read_csv(b / "prevalence/adjusted_metrics_wide.csv")
    t4 = w.groupby("model")[["brier_ec_internal", "brier_ec_external", "brier_skill_internal", "brier_skill_external",
                             "brier_recalibrated_ec_internal", "brier_recalibrated_ec_external", "internal_roc_auc", "external_roc_auc"]].mean().reindex(MODELS1).round(4)
    t4.to_csv(out / "tables/tableS4_skill_and_recalibration_by_model.csv")
    for f in ["adjusted_metrics_long.csv", "meta_evaluation_by_metric.csv", "validation_against_scores.json"]:
        (out / f).write_bytes((b / "prevalence" / f).read_bytes())

    # Fig S1: cutoff sensitivity
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    x = np.array([float(c) for c in cutoffs])
    for j, p in enumerate(PROTS1):
        g = t1[t1.protocol == p]
        ax[0].errorbar(x + (j - 1) * .008, g.rank_agreement_mean, g.rank_agreement_sd, fmt="o-", color=COL[j], capsize=2, label=p.replace("_", " "), ms=4)
        ax[1].plot(x, g.selection_regret_mean, "o-", color=COL[j], ms=4)
    ax[0].axhline(0, color="k", ls=":", lw=.6)
    ax[0].set_ylabel("Rank agreement (mean ± SD, 10 splits)")
    ax[1].set_ylabel("Mean selection regret (Brier)")
    for mo, c in zip(MODELS1, COL):
        g = ext[ext.model == mo]
        ax[2].plot(g.auc_cutoff, g.external_compound_brier, "o-", color=c, ms=4, label=mo.replace("_", " "))
    ax[2].set_ylabel("Mean external Brier loss")
    for a in ax:
        a.set_xlabel("AUC cutoff defining a positive")
        a.set_xticks(x)
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    ax[2].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.set_size_inches(12, 3.4)
    fig.subplots_adjust(wspace=0.38)
    save(fig, out / "figures/figS1_cutoff_sensitivity")

    # Fig S2: skill & recalibration
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    xs = np.arange(5)
    ax[0].bar(xs - .2, t4.brier_skill_internal, .4, color=COL[0], label="internal")
    ax[0].bar(xs + .2, t4.brier_skill_external, .4, color=COL[1], label="external")
    ax[0].axhline(0, color="k", lw=.7)
    ax[0].set_ylabel("Brier skill vs constant forecast")
    ax[1].bar(xs - .2, t4.brier_ec_external, .4, color=COL[1], label="as predicted")
    ax[1].bar(xs + .2, t4.brier_recalibrated_ec_external, .4, color=COL[2], label="intercept-recalibrated")
    long = pd.read_csv(b / "prevalence/adjusted_metrics_long.csv")
    clim = long[long.partition == "external"].climatology_brier_ec.mean()
    ax[1].axhline(clim, color="k", ls=":", lw=.8)
    ax[1].text(-0.45, clim - .0035, "constant forecast", ha="left", fontsize=6.5)
    ax[1].set_ylabel("External equal-compound Brier loss")
    ax[1].set_ylim(.07, .12)
    for a in ax:
        a.set_xticks(xs)
        a.set_xticklabels([m.replace("_", "\n") for m in MODELS1], fontsize=7)
        a.legend(frameon=False, fontsize=7)
    save(fig, out / "figures/figS2_skill_and_recalibration")

    # Fig S3: rank agreement by metric
    fig, ax = plt.subplots(figsize=(6, 3.2))
    metrics = ["brier", "brier_recalibrated", "auroc"]
    for j, p in enumerate(PROTS1):
        v = [m[(m.metric == k) & (m.protocol == p)].rank_agreement.mean() for k in metrics]
        e = [m[(m.metric == k) & (m.protocol == p)].rank_agreement.std() for k in metrics]
        ax.bar(np.arange(3) + (j - 1) * .25, v, .25, yerr=e, color=COL[j], capsize=2, label=p.replace("_", " "))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Brier", "Brier after\nintercept recalibration", "AUROC"])
    ax.set_ylabel("Rank agreement (mean ± SD)")
    ax.legend(frameon=False, fontsize=7)
    ax.axhline(0, color="k", lw=.6)
    save(fig, out / "figures/figS3_rank_agreement_by_metric")
    lp = out / "cellline_subset_scores_long.csv"
    if lp.exists():
        L = pd.read_csv(lp)
        M = pd.read_csv(out / "cellline_subset_meta_evaluation.csv")
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
        for j, (sub, lab) in enumerate([("shared_lines", "cell lines also in development"), ("novel_lines", "cell lines not in development")]):
            v = L[L.subset == sub].groupby("model").brier_skill.mean().reindex(MODELS1)
            ax[0].bar(np.arange(5) + (j - .5) * .38, v, .38, color=COL[j], label=lab)
            r = M[M.external_subset == sub].groupby("protocol").rank_agreement.agg(["mean", "std"]).reindex(PROTS1)
            ax[1].bar(np.arange(3) + (j - .5) * .38, r["mean"], .38, yerr=r["std"], capsize=2, color=COL[j], label=lab)
        ax[0].axhline(0, color="k", lw=.7)
        ax[0].set_xticks(range(5))
        ax[0].set_xticklabels([m.replace("_", " ") for m in MODELS1], fontsize=7)
        ax[0].set_ylabel("External Brier skill vs constant forecast")
        ax[1].set_xticks(range(3))
        ax[1].set_xticklabels([p.replace("_", " ") for p in PROTS1])
        ax[1].set_ylabel("Rank agreement (mean ± SD, 10 splits)")
        ax[1].axhline(0, color="k", lw=.6)
        ax[0].legend(frameon=False, fontsize=7, loc="lower left")
        fig.tight_layout()
        save(fig, out / "figures/figS9_cellline_subsets")
    print(t1.round(3).to_string())
    print(t3.to_string())
    print(t4.to_string())


def topic2():
    b = ROOT / "outputs/sens_t2/sens_t2"
    out = ROOT / "topic2_uncertainty/results/modal_research_r1/sensitivity"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    parts = {}
    for n in ["fn_sweep", "donor_sweep", "annotation_sweep", "nsize_20", "nsize_80", "nsize_160"]:
        parts[n] = pd.read_csv(b / n / "scores_with_skill.csv")
        extra = ROOT / "outputs/extra_t2/extra_t2" / {"fn_sweep": "fn_linear_b", "donor_sweep": "donor_linear_b"}.get(n, "_none_") / "scores_with_skill.csv"
        if extra.exists():  # pool two independent 100-replicate runs (seeds 29 and 31) into 200 replicates
            e = pd.read_csv(extra)
            e["replicate"] = e.replicate + 100
            parts[n] = pd.concat([parts[n], e], ignore_index=True)
    prim = pd.read_csv(b / "prevalence_primary/scores_with_skill.csv")

    def summ(s):
        return s.groupby(["scenario", "protocol", "model"]).agg(
            replicates=("replicate", "nunique"), mean_bias=("bias", "mean"), mae=("absolute_error", "mean"), rmse=("bias", lambda x: float(np.sqrt((x ** 2).mean()))),
            containment=("covers_reference", "mean"), true_mean_risk=("reference_mean_risk", "mean"), observed_prevalence=("observed_prevalence", "mean"),
            skill_reference=("skill_reference", "mean"), skill_estimate=("skill_estimate", "mean"), skill_bias=("skill_bias", "mean")).reset_index()

    for n, s in parts.items():
        summ(s).round(5).to_csv(out / f"tables/tableS_{n}.csv", index=False)
    ps = summ(prim)
    ps.round(5).to_csv(out / "tables/tableS_primary_skill.csv", index=False)

    def meta(s):
        # rank agreement / regret per replicate from reference vs estimated Brier across the 4 models
        from scipy.stats import spearmanr
        rows = []
        for (sc, rep, pr), g in s.groupby(["scenario", "replicate", "protocol"]):
            a, r = g.estimated_brier.to_numpy(), g.reference_expected_brier.to_numpy()
            best = a == a.min()
            rows.append({"scenario": sc, "protocol": pr, "rank": spearmanr(a, r).statistic, "regret": r[best].mean() - r.min(), "hit": float(r[best].mean() == r.min()) if best.sum() == 1 else float(np.mean(r[best] == r.min()))})
        return pd.DataFrame(rows).groupby(["scenario", "protocol"]).mean(numeric_only=True).reset_index()

    for n, s in parts.items():
        meta(s).round(4).to_csv(out / f"tables/tableS_{n}_meta.csv", index=False)

    # Fig: fn sweep and donor sweep, bias and containment (average over unweighted models separately)
    def sweep_fig(name, xs_fn, xlabel, fname):
        s = parts[name]
        s = s.assign(x=s.scenario.map(xs_fn))
        mt = meta(s).assign(x=lambda d: d.scenario.map(xs_fn))
        fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.6))
        ax = ax.ravel()
        for j, pr in enumerate(["random_row", "held_out_donor"]):
            for i, mo in enumerate(sorted(s.model.unique())):
                g = s[(s.protocol == pr) & (s.model == mo)].groupby("x").bias.mean()
                ax[j].plot(g.index, g.values, "o-", color=COL[i], ms=4, label=mo.replace("_", " "))
            ax[j].axhline(0, color="k", lw=.6, ls=":")
            ax[j].set_title(pr.replace("_", " "))
            ax[j].set_ylabel("Mean bias (estimate - reference)")
        ax[1].legend(frameon=False, fontsize=6.5)
        for i, pr in enumerate(["random_row", "held_out_donor"]):
            g = s[s.protocol == pr].groupby("x").covers_reference.mean()
            ax[2].plot(g.index, g.values, "o-", color=COL[i], ms=4, label=pr.replace("_", " "))
            h = mt[mt.protocol == pr].sort_values("x")
            ax[3].plot(h.x, h["rank"], "o-", color=COL[i], ms=4, label=pr.replace("_", " "))
        ax[2].axhline(.95, color="k", lw=.6, ls=":")
        ax[2].set_ylim(0, 1)
        ax[2].set_ylabel("Containment of reference risk (all models)")
        ax[3].axhline(0, color="k", lw=.6, ls=":")
        ax[3].set_ylabel("Mean rank agreement")
        ax[2].legend(frameon=False, fontsize=7)
        for a_ in ax:
            a_.set_xlabel(xlabel)
        fig.tight_layout()
        save(fig, out / f"figures/{fname}")

    sweep_fig("fn_sweep", lambda n: float(n.split("_")[1]), "False-negative rate", "figS4_false_negative_sweep")
    sweep_fig("donor_sweep", lambda n: float(n.split("_")[1]), "Donor effect size", "figS5_donor_effect_sweep")
    sweep_fig("annotation_sweep", lambda n: float(n.split("_")[1]), "Annotation error", "figS6_annotation_error_sweep")

    # n-size
    nrows = []
    for n, d in [(20, "nsize_20"), (80, "nsize_80"), (160, "nsize_160")]:
        s = parts[d].assign(n_donors=n)
        nrows.append(s)
    main = prim[prim.scenario.isin(["clean_linear", "donor_only"])].assign(n_donors=40)
    ns = pd.concat(nrows + [main])
    nt = ns.groupby(["scenario", "protocol", "n_donors"]).agg(mean_bias=("bias", "mean"), mae=("absolute_error", "mean"), containment=("covers_reference", "mean"),
        test_donors=("n_test_donors", "mean"), blocks=("n_test_resampling_blocks", "mean")).round(4).reset_index()
    nt.to_csv(out / "tables/tableS_donor_count_effect.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for a, sc in zip(ax, ["clean_linear", "donor_only"]):
        for j, pr in enumerate(["random_row", "held_out_donor"]):
            g = nt[(nt.scenario == sc) & (nt.protocol == pr)].sort_values("n_donors")
            a.plot(g.n_donors, g.containment, "o-", color=COL[j], label=pr.replace("_", " "))
        a.axhline(.95, color="k", ls=":", lw=.6)
        a.set_xscale("log")
        a.minorticks_off()
        a.set_xticks([20, 40, 80, 160])
        a.set_xticklabels([20, 40, 80, 160])
        a.set_xlabel("Simulated development donors")
        a.set_title(sc.replace("_", " "))
    ax[0].set_ylabel("Containment of reference risk")
    ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_ylim(.5, 1)
    save(fig, out / "figures/figS7_donor_count_containment")

    # Skill figure on primary run
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    sc_order = ["clean_linear", "annotation_only", "negative_label_only", "missing_ptm_only", "donor_only", "combined_nonlinear"]
    for j, pr in enumerate(["random_row", "held_out_donor"]):
        g = ps[ps.protocol == pr].groupby("scenario")[["mean_bias", "skill_bias"]].mean().reindex(sc_order)
        ax[0].plot(range(6), g.mean_bias, "o-", color=COL[j], ms=4, label=pr.replace("_", " "))
        ax[1].plot(range(6), g.skill_bias, "o-", color=COL[j], ms=4)
    for a, t in zip(ax, ["Brier bias (est - reference)", "Brier skill bias (est - reference)"]):
        a.axhline(0, color="k", ls=":", lw=.6)
        a.set_xticks(range(6))
        a.set_xticklabels([x.replace("_", "\n") for x in sc_order], fontsize=6.5)
        a.set_ylabel(t)
    ax[0].legend(frameon=False, fontsize=7)
    save(fig, out / "figures/figS8_skill_vs_brier_bias")
    pr = ps.groupby("scenario")[["true_mean_risk", "observed_prevalence"]].mean().reindex(sc_order).round(3)
    pr.to_csv(out / "tables/tableS_true_vs_observed_prevalence.csv")
    print(pr)
    print(ps.groupby(["scenario", "protocol"])[["mean_bias", "skill_bias", "containment"]].mean().round(3).to_string())
    print(nt.to_string())
    for n in ["fn_sweep", "donor_sweep", "annotation_sweep"]:
        print(n)
        print(summ(parts[n]).groupby(["scenario", "protocol"])[["mean_bias", "containment", "skill_bias"]].mean().round(3).to_string())
        print(meta(parts[n]).round(3).to_string())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("1", "all"):
        topic1()
    if which in ("2", "all"):
        topic2()
