"""Publication figures for Paper 1 (drug-response rank transfer), redrawn from the stored result tables in one style.
Usage (from modal_jobs/): python figures_paper1.py     -> topic1_cell_context/results/modal_research_r1/figures_pub/"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import pubfig as pf
from pubfig import plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "topic1_cell_context/results/modal_research_r1"
S = R / "sensitivity"
OUT = R / "figures_pub"
PROT = ["random_compound", "scaffold", "chemical_cluster"]
MODELS = ["molecular_only", "context_only", "early_fusion", "without_ptm", "late_fusion"]
DODGE = {"random_compound": -0.22, "scaffold": 0.0, "chemical_cluster": 0.22}
XT = ["Random-\ncompound", "Scaffold", "Chemical-\ncluster"]

sc = pd.read_csv(R / "scores.csv")
me = pd.read_csv(R / "meta_evaluation.csv")
m50 = pd.read_csv(S / "meta_evaluation_50seeds.csv")
s1 = pd.read_csv(S / "tables/tableS1_cutoff_sensitivity.csv")
s3 = pd.read_csv(S / "tables/tableS3_prevalence_adjusted_meta_evaluation.csv")
s4 = pd.read_csv(S / "tables/tableS4_skill_and_recalibration_by_model.csv").set_index("model")
s5 = pd.read_csv(S / "tables/tableS5_cellline_subset_by_model.csv")
s7 = pd.read_csv(S / "tables/tableS7_seeds50_meta.csv").set_index("design")
s8 = pd.read_csv(S / "tables/tableS8_design_contrasts_50seeds.csv")
s10 = pd.read_csv(S / "tables/tableS10_replication_cohorts.csv")
s11 = pd.read_csv(S / "tables/tableS11_replication_skill.csv")
lng = pd.read_csv(S / "adjusted_metrics_long.csv")
cellsub = pd.read_csv(S / "cellline_subset_meta_evaluation.csv")
fw = pd.read_csv(R / "fusion_weights.csv")
prep = json.load(open("F:/bioinfo_data/topic1_prepared/prepare_summary.json"))
ov = json.load(open(S / "cellline_overlap_summary.json"))
clim_e = lng[lng.partition == "external"].climatology_brier_ec.mean()
DH = pf.design_handles(PROT, pf.DESIGN_C, pf.DESIGN_L, pf.DESIGN_M)
made = {}


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def model_axis(ax, labels=True):
    ax.set_yticks(np.arange(len(MODELS)), [pf.MODEL_L[m] for m in MODELS] if labels else [])
    ax.set_ylim(-0.6, len(MODELS) - 0.2)
    ax.grid(True, axis="x"), ax.grid(False, axis="y")


def const_line(ax, x):
    ax.axvline(x, color=pf.INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(x, len(MODELS) - 0.42, " constant forecast", fontsize=6, color=pf.INK2, ha="left", va="center")


# ------------------------------------------------------------------ Fig 1 study design
def fig1():
    dev, ext = prep["cohorts"]["development"], prep["cohorts"]["external"]
    fig = plt.figure(figsize=(pf.DOUBLE, 3.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    top, bot, h = 0.65, 0.28, 0.30
    pf.box(ax, (0.01, top), 0.21, h, f"Development cohort\nGDSCv2\n{dev['rows']:,} observations\n{dev['compounds']} compounds, {dev['contexts']} cell lines\nPositive: AUC < 0.7 ({dev['prevalence']*100:.1f}%)")
    pf.box(ax, (0.01, bot), 0.21, h, f"External cohort\nCTRPv2\n{ext['rows']:,} observations\n{ext['compounds']} compounds not in development\n{ov['external_lines_shared_with_development']} of {ov['external_lines']} cell lines shared")
    pf.box(ax, (0.27, top), 0.21, h, "Internal holdout design\nRandom-compound\nScaffold (Murcko)\nChemical-cluster (Butina)\n25% of groups; 10 + 40 seeds", fc="#eaf1fb")
    pf.box(ax, (0.53, bot), 0.20, top + h - bot, "Five prespecified models\nMolecular-only\nContext-only\nEarly fusion (all blocks)\nNo-PTM\nLate fusion\n\nL2-regularised logistic\nregression; fusion weights\nfitted on a calibration\npartition")
    pf.box(ax, (0.78, top), 0.21, h, "Internal assessment\nEqual-compound Brier loss\non the internal holdout\n→ internally selected model")
    pf.box(ax, (0.78, bot), 0.21, h, "External assessment\nEqual-compound Brier loss\non CTRPv2\n→ externally best model")
    pf.box(ax, (0.27, 0.03), 0.72, 0.17, "Meta-evaluation of each design\nInternal–external gap  ·  rank agreement (Spearman correlation of model losses)  ·  selection regret", fc="#fdf0ea")
    pf.box(ax, (0.01, 0.03), 0.21, 0.17, "Sensitivity analyses\ncutoff 0.5–0.9 · prevalence · cell-line\noverlap · 2 replication cohort pairs", fc="#eef8f3", fs=6)
    pf.arrow(ax, (0.22, top + h / 2), (0.27, top + h / 2))
    pf.arrow(ax, (0.48, top + h / 2), (0.53, top + h / 2))
    pf.arrow(ax, (0.73, top + h / 2), (0.78, top + h / 2))
    pf.arrow(ax, (0.22, bot + h / 2), (0.53, bot + h / 2))
    pf.arrow(ax, (0.73, bot + h / 2), (0.78, bot + h / 2))
    pf.arrow(ax, (0.885, top), (0.885, bot + h), style="<|-|>")
    pf.arrow(ax, (0.885, bot), (0.885, 0.20))
    made["fig1"] = pf.save(fig, OUT / "Fig1_study_design")


# ------------------------------------------------------------------ Fig 2 primary analysis
def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(pf.DOUBLE, 2.75), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.45})
    for p in PROT:
        g = sc[sc.protocol == p]
        a.scatter(g.internal_compound_brier, g.external_compound_brier, s=9, marker=pf.DESIGN_M[p], facecolor=pf.DESIGN_C[p], edgecolor="white", lw=0.3, alpha=0.9, zorder=3)
    xlo, xhi = sc.internal_compound_brier.min() - 0.006, sc.internal_compound_brier.max() + 0.006
    ylo, yhi = 0.078, 0.122
    a.plot([ylo, yhi], [ylo, yhi], color=pf.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    a.text(yhi - 0.001, yhi - 0.0005, "equality ", fontsize=6, color=pf.INK2, ha="right", va="top")
    a.set_xlim(xlo, xhi), a.set_ylim(ylo, yhi)
    a.set_xlabel("Internal equal-compound Brier loss"), a.set_ylabel("External equal-compound Brier loss")
    a.grid(True, axis="both")
    pf.legend_below(a, DH, ncol=3, dy=-0.2)
    pf.panel(a, "A", "Internal versus external loss", x=-0.22)
    y = np.arange(len(MODELS))
    for p in PROT:
        g = sc[sc.protocol == p].groupby("model").external_compound_brier
        b.errorbar(g.mean().reindex(MODELS), y + DODGE[p], xerr=g.apply(ci95).reindex(MODELS), fmt=pf.DESIGN_M[p], color=pf.DESIGN_C[p], ms=4, lw=1, zorder=3)
    model_axis(b)
    const_line(b, clim_e)
    b.set_xlabel("External Brier loss (mean ± 95% CI, 10 seeds)")
    a.get_legend().remove()
    fig.legend(handles=DH, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    pf.panel(b, "B", "External loss by model", x=-0.36)
    made["fig2"] = pf.save(fig, OUT / "Fig2_primary_analysis")


# ------------------------------------------------------------------ Fig 3 meta-evaluation, 10 vs 50 seeds, and design contrasts
def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(pf.DOUBLE, 2.7), gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.45})
    rng = np.random.default_rng(1)
    for ax, col, lab, ci_lo, ci_hi, key in [(axs[0], "rank_agreement", "Rank agreement (Spearman)", "rank_ci_low", "rank_ci_high", "rank_agreement_mean"),
                                            (axs[1], "selection_regret", "Selection regret (Brier units)", "regret_ci_low", "regret_ci_high", "regret_mean")]:
        for i, p in enumerate(PROT):
            v = m50[m50.protocol == p][col].to_numpy()
            ax.scatter(i - 0.12 + rng.uniform(-0.14, 0.14, len(v)), v, s=4, color=pf.MUTED, alpha=0.55, lw=0, zorder=2)
            mu = s7.loc[p, key]
            ax.errorbar(i + 0.25, mu, yerr=[[mu - s7.loc[p, ci_lo]], [s7.loc[p, ci_hi] - mu]], fmt=pf.DESIGN_M[p], color=pf.DESIGN_C[p], ms=5, lw=1.2, zorder=4)
            ax.scatter(i + 0.25, me[me.protocol == p][col].mean(), s=22, marker=pf.DESIGN_M[p], facecolor="white", edgecolor=pf.DESIGN_C[p], lw=0.9, zorder=5)
        ax.set_xticks(range(3), XT), ax.set_xlim(-0.5, 2.5), ax.set_ylabel(lab)
    pf.refline(axs[0], y=0)
    pf.legend_below(axs[0], [Line2D([], [], color=pf.MUTED, marker="o", lw=0, ms=2.5, label="single seed (50 per design)"),
                             Line2D([], [], color=pf.INK2, marker="o", lw=1, ms=4, label="mean and 95% CI, 50 seeds"),
                             Line2D([], [], color=pf.INK2, marker="o", lw=0, ms=4.5, mfc="white", label="mean, 10 prespecified seeds")], ncol=3, dy=-0.27, bbox_transform=axs[0].transAxes,
                    bbox_to_anchor=(1.75, -0.27))
    pf.panel(axs[0], "A", "Rank agreement")
    pf.panel(axs[1], "B", "Selection regret")
    c = s8[s8.metric == "rank_agreement"].reset_index(drop=True)
    lab = {"scaffold minus random_compound": "Scaffold − random-compound", "chemical_cluster minus random_compound": "Cluster − random-compound",
           "chemical_cluster minus scaffold": "Cluster − scaffold"}
    yy = np.arange(len(c))[::-1].astype(float)
    ax = axs[2]
    ax.errorbar(c.difference, yy, xerr=[c.difference - c.ci_low, c.ci_high - c.difference], fmt="o", color=pf.INK, ms=4, lw=1.1)
    for yv, (_, r) in zip(yy, c.iterrows()):
        ax.text(-0.29, yv + 0.22, lab[r.contrast], fontsize=6.3, color=pf.INK, ha="left", va="bottom")
    ax.axvline(0, color=pf.MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_yticks([]), ax.set_ylim(-0.5, len(c) - 0.2), ax.set_xlim(-0.3, 0.36)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x"), ax.grid(False, axis="y")
    ax.set_xlabel("Difference in rank agreement\n(95% CI, 50 seeds)")
    pf.panel(ax, "C", "Design contrasts", x=-0.06)
    made["fig3"] = pf.save(fig, OUT / "Fig3_meta_evaluation_seeds")


# ------------------------------------------------------------------ Fig 4 endpoint cutoff
def fig4():
    fig, axs = plt.subplots(1, 3, figsize=(pf.DOUBLE, 2.55), gridspec_kw={"wspace": 0.45})
    cuts = sorted(s1.auc_cutoff.unique())
    off = {"random_compound": -0.012, "scaffold": 0.0, "chemical_cluster": 0.012}
    for p in PROT:
        g = s1[s1.protocol == p].set_index("auc_cutoff").reindex(cuts)
        axs[0].errorbar(np.array(cuts) + off[p], g.rank_agreement_mean, yerr=g.rank_agreement_sd, color=pf.DESIGN_C[p], marker=pf.DESIGN_M[p], ms=4, lw=1, elinewidth=0.6)
        axs[1].plot(cuts, g.selection_regret_mean, color=pf.DESIGN_C[p], marker=pf.DESIGN_M[p], ms=4, lw=1)
    pf.refline(axs[0], y=0)
    axs[0].set_ylabel("Rank agreement (mean ± SD, 10 seeds)")
    axs[1].set_ylabel("Mean selection regret (Brier units)")
    prev = s1.groupby("auc_cutoff")[["internal_prevalence", "external_prevalence"]].mean().reindex(cuts) * 100
    axs[2].plot(cuts, prev.internal_prevalence, color=pf.INK, marker="o", ms=4, lw=1, label="internal holdouts")
    axs[2].plot(cuts, prev.external_prevalence, color=pf.MUTED, marker="s", ms=4, lw=1, ls="--", label="external cohort")
    axs[2].set_ylabel("Positive observations (%)")
    for ax in axs:
        ax.set_xticks(cuts), ax.set_xlabel("AUC cutoff defining a positive")
        ax.axvspan(0.68, 0.72, color=pf.GRID, lw=0, zorder=0)
    pf.legend_below(axs[1], DH, ncol=3, dy=-0.24)
    axs[2].legend(loc="upper left")
    pf.panel(axs[0], "A", "Rank agreement")
    pf.panel(axs[1], "B", "Selection regret")
    pf.panel(axs[2], "C", "Prevalence")
    made["fig4"] = pf.save(fig, OUT / "Fig4_endpoint_cutoff")


# ------------------------------------------------------------------ Fig 5 prevalence-adjusted analyses
def fig5():
    fig = plt.figure(figsize=(pf.DOUBLE, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.12)
    a = fig.add_subplot(gs[0])
    b = fig.add_subplot(gs[1], sharey=a)
    c = fig.add_subplot(gs[2])
    fig.subplots_adjust(left=0.13)
    y = np.arange(len(MODELS))
    a.scatter(s4.brier_skill_internal.reindex(MODELS), y + 0.12, marker="o", s=18, facecolor="white", edgecolor=pf.INK, lw=0.9, label="internal", zorder=3)
    a.scatter(s4.brier_skill_external.reindex(MODELS), y - 0.12, marker="o", s=18, color=pf.INK, label="external", zorder=3)
    a.axvline(0, color=pf.MUTED, lw=0.7, ls=(0, (3, 2)))
    model_axis(a)
    a.set_xlabel("Brier skill vs constant forecast")
    pf.legend_below(a, ncol=2, dy=-0.2)
    pf.panel(a, "A", "Skill", x=-0.62)
    b.scatter(s4.brier_ec_external.reindex(MODELS), y + 0.12, marker="o", s=18, color=pf.INK, label="as predicted", zorder=3)
    b.scatter(s4.brier_recalibrated_ec_external.reindex(MODELS), y - 0.12, marker="D", s=15, facecolor="white", edgecolor=pf.INK, lw=0.9, label="recalibrated", zorder=3)
    const_line(b, clim_e)
    plt.setp(b.get_yticklabels(), visible=False)
    b.tick_params(axis="y", length=0)
    b.grid(True, axis="x"), b.grid(False, axis="y")
    b.set_xlabel("External Brier loss")
    pf.legend_below(b, ncol=2, dy=-0.2)
    pf.panel(b, "B", "Intercept recalibration", x=-0.05)
    metrics = ["brier", "brier_skill", "brier_recalibrated", "auroc"]
    ml = ["Brier", "Skill", "Recal.\nBrier", "AUROC"]
    x = np.arange(len(metrics))
    m = s3.set_index(["metric", "protocol"])
    for p in PROT:
        c.errorbar(x + DODGE[p], [m.loc[(k, p)].rank_agreement_mean for k in metrics], yerr=[m.loc[(k, p)].rank_agreement_sd for k in metrics],
                   fmt=pf.DESIGN_M[p], color=pf.DESIGN_C[p], ms=4, lw=0.8)
    pf.refline(c, y=0)
    c.set_xticks(x, ml), c.set_ylabel("Rank agreement (mean ± SD)")
    pf.legend_below(c, DH, ncol=2, dy=-0.2, fontsize=6)
    pf.panel(c, "C", "Agreement by metric", x=-0.3)
    pos = c.get_position()
    c.set_position([pos.x0 + 0.07, pos.y0, pos.width - 0.07, pos.height])
    made["fig5"] = pf.save(fig, OUT / "Fig5_prevalence_adjusted")


# ------------------------------------------------------------------ Fig 6 cell-line overlap
def fig6():
    fig, axs = plt.subplots(1, 3, figsize=(pf.DOUBLE, 2.7), gridspec_kw={"width_ratios": [1.15, 1, 1], "wspace": 0.5})
    y = np.arange(len(MODELS))
    t = s5.set_index(["subset", "model"]).brier_skill
    axs[0].scatter(t.loc["shared_lines"].reindex(MODELS), y + 0.12, s=18, color=pf.INK, label=f"shared ({ov['external_lines_shared_with_development']} lines)", zorder=3)
    axs[0].scatter(t.loc["novel_lines"].reindex(MODELS), y - 0.12, s=18, marker="s", facecolor="white", edgecolor=pf.INK, lw=0.9, label=f"new ({ov['novel_line_count']} lines)", zorder=3)
    axs[0].axvline(0, color=pf.MUTED, lw=0.7, ls=(0, (3, 2)))
    model_axis(axs[0])
    axs[0].set_xlabel("External Brier skill")
    pf.legend_below(axs[0], ncol=2, dy=-0.2)
    pf.panel(axs[0], "A", "Skill by cell-line subset", x=-0.55)
    for ax, col, lab in [(axs[1], "rank_agreement", "Rank agreement (mean ± SD)"), (axs[2], "selection_regret", "Selection regret (mean ± 95% CI)")]:
        for j, (sub, mk) in enumerate([("shared_lines", "o"), ("novel_lines", "s")]):
            for i, p in enumerate(PROT):
                v = cellsub[(cellsub.protocol == p) & (cellsub.external_subset == sub)][col]
                err = v.std(ddof=1) if col == "rank_agreement" else ci95(v)
                ax.errorbar(i + (-0.13 if j == 0 else 0.13), v.mean(), yerr=err, fmt=mk, color=pf.DESIGN_C[p], mfc="white" if j else pf.DESIGN_C[p], ms=4.5, lw=0.9)
        ax.set_xticks(range(3), XT), ax.set_ylabel(lab)
    pf.refline(axs[1], y=0)
    sub_h = [Line2D([], [], color=pf.INK2, marker="o", lw=0, ms=4, label="shared lines"), Line2D([], [], color=pf.INK2, marker="s", mfc="white", lw=0, ms=4, label="new lines")]
    pf.legend_below(axs[1], sub_h, ncol=2, dy=-0.27, bbox_to_anchor=(1.3, -0.27))
    pf.panel(axs[1], "B", "Rank agreement")
    pf.panel(axs[2], "C", "Selection regret")
    made["fig6"] = pf.save(fig, OUT / "Fig6_cellline_overlap")


# ------------------------------------------------------------------ Fig 7 replication
def fig7():
    fig, (a, b) = plt.subplots(1, 2, figsize=(pf.DOUBLE, 2.8), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.5})
    cohorts = ["GDSCv2 to CTRPv2 (primary)", "CTRPv2 to GDSCv2 (reverse)", "GDSCv2 to GDSCv1"]
    cl = ["GDSCv2→\nCTRPv2\n(primary)", "CTRPv2→\nGDSCv2\n(reverse)", "GDSCv2→\nGDSCv1"]
    t = s10.set_index(["cohort", "design"])
    for p in PROT:
        a.errorbar(np.arange(3) + DODGE[p], [t.loc[(c, p)].rank_agreement_mean for c in cohorts], yerr=[t.loc[(c, p)].rank_agreement_sd for c in cohorts],
                   fmt=pf.DESIGN_M[p], color=pf.DESIGN_C[p], ms=4.5, lw=0.9)
    pf.refline(a, y=0)
    a.set_xticks(range(3), cl), a.set_ylabel("Rank agreement (mean ± SD, 10 seeds)")
    pf.legend_below(a, DH, ncol=3, dy=-0.32)
    pf.panel(a, "A", "Rank agreement by cohort pair")
    k = s11.set_index(["cohort", "model"]).external
    y = np.arange(len(MODELS))
    b.scatter(s4.brier_skill_external.reindex(MODELS), y + 0.18, s=16, marker="o", color=pf.INK, label="GDSCv2→CTRPv2", zorder=3)
    b.scatter(k.loc[cohorts[1]].reindex(MODELS), y, s=16, marker="s", facecolor="white", edgecolor=pf.INK, lw=0.9, label="CTRPv2→GDSCv2", zorder=3)
    b.scatter(k.loc[cohorts[2]].reindex(MODELS), y - 0.18, s=16, marker="^", color=pf.MUTED, label="GDSCv2→GDSCv1", zorder=3)
    b.axvline(0, color=pf.MUTED, lw=0.7, ls=(0, (3, 2)))
    model_axis(b)
    b.set_xlabel("External Brier skill vs constant forecast")
    pf.legend_below(b, ncol=3, dy=-0.2, fontsize=6)
    pf.panel(b, "B", "External skill by model", x=-0.36)
    made["fig7"] = pf.save(fig, OUT / "Fig7_replication")


# ------------------------------------------------------------------ Fig S1 fusion weights
def figS1():
    fig, ax = plt.subplots(figsize=(pf.SINGLE * 1.3, 2.5))
    blocks = [("mol", "Molecular"), ("context", "Cell context"), ("ptm", "Phospho-protein")]
    rng = np.random.default_rng(2)
    for j, (bk, _) in enumerate(blocks):
        for p in PROT:
            v = fw[(fw.protocol == p) & (fw.modality == bk)].weight
            x0 = j + DODGE[p]
            ax.scatter(x0 + rng.uniform(-0.05, 0.05, len(v)), v, s=4, color=pf.DESIGN_C[p], alpha=0.45, lw=0)
            ax.errorbar(x0, v.mean(), yerr=v.std(ddof=1), fmt=pf.DESIGN_M[p], color=pf.DESIGN_C[p], ms=4.5, lw=1, zorder=4)
    ax.set_xticks(range(3), [b for _, b in blocks]), ax.set_ylabel("Late-fusion block weight"), ax.set_ylim(-0.05, 1.05)
    pf.legend_below(ax, DH, ncol=3, dy=-0.13)
    made["figS1"] = pf.save(fig, OUT / "FigS1_fusion_weights")


for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, figS1]:
    f()
for k, v in made.items():
    print(k, v.relative_to(R))
