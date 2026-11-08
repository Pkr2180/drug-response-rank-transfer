"""Aggregate the extra Modal runs (yadalam1980): Topic 1 (50 seeds, replication cohorts) and Topic 2 (mechanism sweeps, 200-replicate linear sweeps).
Usage: python modal_jobs/analyze_extra.py t1 | t2   (after downloading extra_t1 / extra_t2 files into outputs/extra_t1, outputs/extra_t2)"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
COL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
PROT = ["random_compound", "scaffold", "chemical_cluster"]
MODELS = ["molecular_only", "context_only", "early_fusion", "without_ptm", "late_fusion"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(20261105)


def save(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def boot_ci(x, stat=np.mean, n=10000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    b = np.array([stat(rng.choice(x, len(x))) for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_diff(a, b, n=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)])
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ------------------------------------------------------------------ Topic 1
def topic1():
    prim = ROOT / "topic1_cell_context/results/modal_research_r1"
    ex = ROOT / "outputs/extra_t1"
    out = prim / "sensitivity"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # ---- 50 seeds
    if all((ex / f"seeds_{k}/meta_evaluation.csv").exists() for k in "ABCD"):
        me = pd.concat([pd.read_csv(prim / "meta_evaluation.csv")] + [pd.read_csv(ex / f"seeds_{k}/meta_evaluation.csv") for k in "ABCD"])
        sc = pd.concat([pd.read_csv(prim / "scores.csv")] + [pd.read_csv(ex / f"seeds_{k}/scores.csv") for k in "ABCD"])
        assert me.seed.nunique() == 50, me.seed.nunique()
        rows = []
        for p in PROT:
            g = me[me.protocol == p]
            lo, hi = boot_ci(g.rank_agreement)
            rl, rh = boot_ci(g.selection_regret)
            rows.append({"design": p, "seeds": len(g), "rank_agreement_mean": g.rank_agreement.mean(), "rank_agreement_sd": g.rank_agreement.std(), "rank_ci_low": lo, "rank_ci_high": hi,
                         "regret_mean": g.selection_regret.mean(), "regret_ci_low": rl, "regret_ci_high": rh, "zero_regret_share": float((g.selection_regret == 0).mean()),
                         "abs_gap_mean": g.mean_absolute_brier_gap.mean()})
        t7 = pd.DataFrame(rows)
        t7.round(4).to_csv(out / "tables/tableS7_seeds50_meta.csv", index=False)
        rows = []
        for a, b in [("scaffold", "random_compound"), ("chemical_cluster", "random_compound"), ("chemical_cluster", "scaffold")]:
            for metric in ["rank_agreement", "selection_regret"]:
                d, lo, hi = boot_diff(me[me.protocol == a][metric], me[me.protocol == b][metric])
                rows.append({"contrast": f"{a} minus {b}", "metric": metric, "difference": d, "ci_low": lo, "ci_high": hi})
        t8 = pd.DataFrame(rows)
        t8.round(4).to_csv(out / "tables/tableS8_design_contrasts_50seeds.csv", index=False)
        ext = sc.groupby(["protocol", "seed", "model"]).external_compound_brier.mean().unstack()
        lf_best = (ext.idxmin(axis=1) == "late_fusion").groupby(level=0).mean()
        t9 = sc.groupby(["protocol", "model"]).agg(external_brier=("external_compound_brier", "mean"), external_auroc=("external_roc_auc", "mean")).round(4)
        t9["late_fusion_best_share"] = [lf_best[p] if m == "late_fusion" else np.nan for p, m in t9.index]
        t9.to_csv(out / "tables/tableS9_seeds50_models.csv")
        me.to_csv(out / "meta_evaluation_50seeds.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
        for a, k, lab in zip(ax, ["rank_agreement", "selection_regret"], ["Rank agreement", "Selection regret"]):
            data = [me[me.protocol == p][k] for p in PROT]
            a.boxplot(data, widths=.5, showfliers=False)
            for i, d in enumerate(data):
                a.scatter(np.random.default_rng(2).normal(i + 1, .06, len(d)), d, s=9, color=COL[i], alpha=.7, zorder=3)
            a.set_xticks([1, 2, 3])
            a.set_xticklabels([p.replace("_", "\n") for p in PROT])
            a.set_ylabel(lab + " (50 seeds)")
        ax[0].axhline(0, color="k", lw=.6, ls=":")
        fig.tight_layout()
        save(fig, out / "figures/figS10_seeds50_meta")
        print(t7.round(3).to_string())
        print(t8.round(3).to_string())
        print(t9.to_string())

    # ---- replication cohorts
    reps = {"GDSCv2 to CTRPv2 (primary)": (prim / "meta_evaluation.csv", prim / "scores.csv", None),
            "CTRPv2 to GDSCv2 (reverse)": (ex / "rev_ctrp_gdsc2/meta_evaluation.csv", ex / "rev_ctrp_gdsc2/scores.csv", ex / "rev_ctrp_gdsc2/predictions.csv"),
            "GDSCv2 to GDSCv1": (ex / "gdsc2_gdsc1/meta_evaluation.csv", ex / "gdsc2_gdsc1/scores.csv", ex / "gdsc2_gdsc1/predictions.csv")}
    rows, skills, mrows = [], [], []
    for name, (mp, sp, pp) in reps.items():
        m, s = pd.read_csv(mp), pd.read_csv(sp)
        for p in PROT:
            g = m[m.protocol == p]
            e = s[s.protocol == p].groupby("model").external_compound_brier.mean()
            rows.append({"cohort": name, "design": p, "external_prevalence": s[s.protocol == p].external_prevalence.mean(), "internal_prevalence": s[s.protocol == p].internal_prevalence.mean(),
                         "rank_agreement_mean": g.rank_agreement.mean(), "rank_agreement_sd": g.rank_agreement.std(), "regret_mean": g.selection_regret.mean(),
                         "zero_regret_splits": int((g.selection_regret == 0).sum()), "abs_gap_mean": g.mean_absolute_brier_gap.mean(), "best_external_model": e.idxmin(), "late_fusion_external_brier": e["late_fusion"],
                         "external_auroc_late_fusion": s[(s.protocol == p) & (s.model == "late_fusion")].external_roc_auc.mean()})
        if pp is not None and pp.exists():
            pr = pd.read_csv(pp, usecols=["protocol", "seed", "model", "partition", "compound_id", "label", "probability"], dtype={"protocol": "category", "model": "category", "partition": "category", "compound_id": "category"})
            pr["err"] = (pr.label - pr.probability) ** 2
            for (pro, seed, mo, part), g in pr.groupby(["protocol", "seed", "model", "partition"], observed=True):
                if part not in ("internal", "external"):
                    continue
                pi = g.label.mean()
                ref = ((g.label - pi) ** 2).groupby(g.compound_id, observed=True).mean().mean()
                b = g.groupby("compound_id", observed=True).err.mean().mean()
                skills.append({"cohort": name, "protocol": pro, "seed": seed, "model": mo, "partition": part, "brier": b, "climatology": ref, "skill": 1 - b / ref})
    t10 = pd.DataFrame(rows)
    t10.round(4).to_csv(out / "tables/tableS10_replication_cohorts.csv", index=False)
    if skills:
        sk = pd.DataFrame(skills)
        sk.to_csv(out / "replication_skill_long.csv", index=False)
        t11 = sk.groupby(["cohort", "model", "partition"]).skill.mean().unstack().round(3)
        t11.to_csv(out / "tables/tableS11_replication_skill.csv")
        print(t11.to_string())
    print(t10.round(3).to_string())
    # figure: rank agreement by cohort and design
    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = list(reps)
    for j, p in enumerate(PROT):
        v = [t10[(t10.cohort == n) & (t10.design == p)].rank_agreement_mean.iloc[0] for n in names]
        e = [t10[(t10.cohort == n) & (t10.design == p)].rank_agreement_sd.iloc[0] for n in names]
        ax.bar(np.arange(3) + (j - 1) * .26, v, .26, yerr=e, capsize=2, color=COL[j], label=p.replace("_", " "))
    ax.set_xticks(range(3))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7.5)
    ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("Rank agreement (mean ± SD, 10 splits)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, out / "figures/figS11_replication_cohorts")


# ------------------------------------------------------------------ Topic 2
def topic2():
    prim = ROOT / "topic2_uncertainty/results/modal_research_r1"
    sens = ROOT / "outputs/sens_t2/sens_t2"
    ex = ROOT / "outputs/extra_t2/extra_t2"
    out = prim / "sensitivity"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    def load(path, offset=0):
        s = pd.read_csv(path)
        s["replicate"] = s.replicate + offset
        return s

    sets = {
        ("linear", "fn"): pd.concat([load(sens / "fn_sweep/scores_with_skill.csv"), load(ex / "fn_linear_b/scores_with_skill.csv", 100)]),
        ("linear", "donor"): pd.concat([load(sens / "donor_sweep/scores_with_skill.csv"), load(ex / "donor_linear_b/scores_with_skill.csv", 100)]),
        ("interaction", "fn"): load(ex / "fn_interaction/scores_with_skill.csv"),
        ("ptm_dominant", "fn"): load(ex / "fn_ptm_dominant/scores_with_skill.csv"),
        ("interaction", "donor"): load(ex / "donor_interaction/scores_with_skill.csv"),
        ("ptm_dominant", "donor"): load(ex / "donor_ptm_dominant/scores_with_skill.csv"),
    }
    rows = []
    for (mech, sweep), s in sets.items():
        s = s.assign(level=s.scenario.str.split("_").str[-1].astype(float))
        for (lv, pr), g in s.groupby(["level", "protocol"]):
            ranks, regs, hits = [], [], []
            for (_, _), h in g.groupby(["scenario", "replicate"]):
                est, ref = h.estimated_brier.to_numpy(), h.reference_expected_brier.to_numpy()
                ranks.append(spearmanr(est, ref).statistic)
                sel = est == est.min()
                regs.append(ref[sel].mean() - ref.min())
                hits.append(float(np.mean(ref[sel] == ref.min())))
            rows.append({"mechanism": mech, "sweep": sweep, "level": lv, "protocol": pr, "replicates": g.replicate.nunique(), "mean_bias": g.bias.mean(), "containment": g.covers_reference.mean(),
                         "skill_bias": g.skill_bias.mean(), "rank_agreement": np.nanmean(ranks), "regret": np.mean(regs), "p_best_chosen": np.mean(hits)})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(out / "tables/tableS12_mechanism_sweeps.csv", index=False)
    print(t.round(3).to_string())
    for sweep, xl in [("fn", "False-negative rate"), ("donor", "Donor-effect SD")]:
        fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
        ax = ax.ravel()
        for j, (metric, lab) in enumerate([("rank_agreement", "Rank agreement"), ("containment", "Containment of reference risk")]):
            for k, pr in enumerate(["random_row", "held_out_donor"]):
                a = ax[j * 2 + k]
                for i, mech in enumerate(["linear", "interaction", "ptm_dominant"]):
                    g = t[(t.sweep == sweep) & (t.mechanism == mech) & (t.protocol == pr)].sort_values("level")
                    a.plot(g.level, g[metric], "o-", color=COL[i], ms=4, label=mech.replace("_", " "))
                a.set_title(pr.replace("_", " "))
                a.set_xlabel(xl)
                a.set_ylabel(lab)
                if metric == "rank_agreement":
                    a.axhline(0, color="k", lw=.6, ls=":")
                else:
                    a.axhline(.95, color="k", lw=.6, ls=":")
                    a.set_ylim(0, 1)
        ax[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        save(fig, out / f"figures/figS{12 if sweep == 'fn' else 13}_mechanism_{sweep}_sweep")


if __name__ == "__main__":
    {"t1": topic1, "t2": topic2}[sys.argv[1]]()
