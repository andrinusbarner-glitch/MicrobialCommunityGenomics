"""Figures for the written report (two-column A4), drawn only from tables in results/.

    python report_figures.py            -> results/figures/report/fig{1..5}_*.pdf and .png

Fig 1  habitat filtering     run_summary.tsv, edge_fate.tsv
Fig 2  what remains          env_network_phylo.tsv, context_pair_classes.tsv
Fig 3  TM7 positive control  tm7_benchmark_summary.tsv
Fig 4  replication           study_replication_summary.tsv, shortlist_summary.tsv, shortlist_final.tsv
Fig 5  metabolic models      gem_scores.tsv
Prints every number shown, so the text can be checked against it.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ENVS = ["Human-associated", "Animal-associated", "Non-saline", "Saline"]
ENV_SHORT = {"Human-associated": "Human", "Animal-associated": "Animal", "Non-saline": "Non-saline", "Saline": "Saline"}
MODELS = ["global", "env", "env_rich", "env_study_rich", "hab_study_rich"]
MODEL_LBL = ["none", "+ environment", "+ richness", "+ study", "+ sub-habitat"]
POS, NEG, GREY, DARK, ACC = "#2166ac", "#b2182b", "#9e9e9e", "#3a3a3a", "#e08214"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "pdf.fonttype": 42})
W1, W2 = 3.4, 7.1      # one column / full width, inches


def panel(ax, letter, x=-0.14):
    ax.text(x, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")


COL = False   # True: one-column figures with panels stacked (A above B), saved as *_col


def mk(ratios=(1, 1), h_wide=2.4, h_col=4.6, **kw):
    if COL:
        return plt.subplots(2, 1, figsize=(W1, h_col), **kw)
    return plt.subplots(1, 2, figsize=(W2, h_wide), gridspec_kw={"width_ratios": list(ratios)}, **kw)


def save(fig, out, name):
    if COL:
        name += "_col"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig1(R, out):
    rs = pd.read_csv(f"{R}/run_summary.tsv", sep="\t").set_index("model").loc[MODELS]
    ef = pd.read_csv(f"{R}/edge_fate.tsv", sep="\t").set_index("comparison").loc["global->hab_study_rich"]
    fig, (a, b) = mk((1.15, 1), 2.4, 4.9)
    x = np.arange(len(MODELS))
    a.plot(x, rs.sig_pos_pairs / 1e6, "o-", color=POS, label="positive")
    a.plot(x, rs.sig_neg_pairs / 1e6, "o-", color=NEG, label="negative")
    a.set_xticks(x, MODEL_LBL, rotation=25, ha="right"); a.set_ylabel("significant pairs (millions)")
    a.set_xlabel("controls in the null model (cumulative)"); a.legend(frameon=False)
    a.set_title("Significant associations per null model", loc="left"); panel(a, "A")
    rows = []
    for sgn, col in [("pos", POS), ("neg", NEG)]:
        tot = ef[f"{sgn}_kept"] + ef[f"{sgn}_lost"] + ef[f"{sgn}_flipped"]
        rows.append((sgn, ef[f"{sgn}_kept"] / tot, ef[f"{sgn}_flipped"] / tot, ef[f"{sgn}_lost"] / tot, tot))
    for i, (sgn, k, f, l, tot) in enumerate(rows):
        b.barh(i, k, color=DARK); b.barh(i, f, left=k, color=ACC); b.barh(i, l, left=k + f, color="#cfd4da")
        b.text(1.01, i, f"{(f + l):.0%} lost\nor flipped", va="center", fontsize=7)
        print(f"fig1 {sgn}: n={tot:,.0f} kept {k:.3f} flipped {f:.3f} lost {l:.3f} lost+flipped {f + l:.3f}")
    b.set_yticks([0, 1], [f"positive\n(n = {rows[0][4] / 1e6:.1f} M)", f"negative\n(n = {rows[1][4] / 1e6:.1f} M)"])
    b.invert_yaxis(); b.set_xlim(0, 1); b.set_xlabel("share of pairs significant without controls")
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (DARK, ACC, "#cfd4da")],
             labels=["kept", "flipped sign", "lost"], frameon=False, ncol=3, loc="upper center",
             bbox_to_anchor=(0.5, -0.32))
    b.set_title("Fate under all controls (none → main model)", loc="left"); panel(b, "B")
    print("fig1 pos/neg ratio", rs.pos_neg_ratio.round(2).to_dict())
    fig.tight_layout(); save(fig, out, "fig1_habitat")


def fig2(R, out):
    ph = pd.read_csv(f"{R}/env_network_phylo.tsv", sep="\t")
    ph = ph[ph.variant == "controlled"]
    cx = pd.read_csv(f"{R}/context_pair_classes.tsv", sep="\t").set_index("level")
    fig, (a, b) = mk((1.2, 1), 2.3, 4.9)
    y = np.arange(len(ENVS))
    for kind, col, off, lbl in [("positive network edges", POS, -0.12, "positive partners"),
                                ("negative pairs (r <= -min_r)", NEG, 0.12, "negative partners")]:
        d = ph[ph.pairs == kind].set_index("group").loc[ENVS]
        a.scatter(d.ratio_config, y + off, color=col, s=22, label=lbl, zorder=3)
        for e, r_, s_ in zip(ENVS, d.ratio_config, d.ses_config):
            print(f"fig2 {kind[:8]} {e}: ratio {r_:.3f} SES {s_:.0f}")
    a.axvline(1, color=GREY, lw=0.8, ls="--")
    a.set_yticks(y, [ENV_SHORT[e] for e in ENVS]); a.invert_yaxis()
    a.set_xlabel("patristic distance, observed / degree-preserving null")
    a.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.3))
    a.set_title("Relatedness of partners (main model)", loc="left")
    panel(a, "A")
    lv = [("category", "environments"), ("sub-habitat", "sub-habitats")]
    for i, (k, lbl) in enumerate(lv):
        r = cx.loc[k]; g, s_, v = r.frac_general, r.frac_context_specific, r.frac_sign_reversing
        b.barh(i, s_, color=GREY); b.barh(i, g, left=s_, color=POS); b.barh(i, v, left=s_ + g, color=NEG)
        b.text(s_ / 2, i, f"{s_:.0%}", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")
        print(f"fig2 context {k}: n={r.pairs_robust_pos_testable_in_ge2:,.0f} specific {s_:.4f} general {g:.4f} reversing {v:.4f}")
    b.set_yticks([0, 1], [f"{lbl}\n(n = {cx.loc[k].pairs_robust_pos_testable_in_ge2 / 1e3:.0f} k)" for k, lbl in lv])
    b.invert_yaxis(); b.set_xlim(0, 1); b.set_xlabel("share of robust positive pairs testable in ≥ 2 contexts")
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, POS, NEG)],
             labels=["specific to one", "shared", "reversed"], frameon=False, ncol=3, loc="upper center",
             bbox_to_anchor=(0.5, -0.3))
    b.set_title("Are associations shared between contexts?", loc="left"); panel(b, "B")
    fig.tight_layout(); save(fig, out, "fig2_remaining")


def fig3(R, out):
    T = pd.read_csv(f"{R}/tm7_benchmark_summary.tsv", sep="\t")
    fig, (a, b) = mk((1, 1.15), 2.5, 5.4)
    d = T[(T.group == "all TM7") & (T.metric == "top1_tier1")].set_index("model").loc[MODELS]
    x = np.arange(len(MODELS))
    a.fill_between(x, d.control_lo, d.control_hi, color=GREY, alpha=0.25, lw=0, label="matched controls, 95% range")
    a.plot(x, d.control_mean, "o-", color=GREY, ms=4, label="matched controls, mean")
    a.plot(x, d.tm7, "o-", color=POS, ms=5, label="TM7 (n = 41)")
    a.set_xticks(x, MODEL_LBL, rotation=25, ha="right"); a.set_ylabel("top partner is an Actinomycetota")
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    a.legend(frameon=False, loc="upper left"); a.set_title("Host phylum recovered only after controls", loc="left")
    panel(a, "A", x=-0.2)
    for m, r in d.iterrows():
        print(f"fig3 top1 {m}: TM7 {r.tm7:.3f} ctrl {r.control_mean:.3f} [{r.control_lo:.3f},{r.control_hi:.3f}] p {r.p:.4f}")
    mets = [("top1_tier1", "top partner:\nhost phylum"), ("top10_tier3", "top 10 partners:\ndocumented host genus")]
    groups = [("all TM7", "all"), ("Human-associated", "Human"), ("Non-saline", "Non-sal." if COL else "Non-saline")]
    M = T[T.model == "hab_study_rich"].set_index(["group", "metric"])
    w = 0.13; xs = []
    for j, (met, mlbl) in enumerate(mets):
        for i, (g, glbl) in enumerate(groups):
            r = M.loc[(g, met)]; xc = j + (i - 1) * 0.3
            b.bar(xc - w / 2, r.tm7, w, color=POS); b.bar(xc + w / 2, r.control_mean, w, color=GREY)
            b.errorbar(xc + w / 2, r.control_mean, yerr=[[r.control_mean - r.control_lo], [r.control_hi - r.control_mean]],
                       color=DARK, lw=0.8, capsize=2)
            star = "***" if r.p < 0.001 else "**" if r.p < 0.01 else "*" if r.p < 0.05 else "n.s."
            b.text(xc, max(r.tm7, r.control_hi) + 0.02, star, ha="center", fontsize=6.5)
            b.text(xc, -0.03, glbl, ha="center", va="top", fontsize=6.5, transform=b.get_xaxis_transform())
            print(f"fig3 main {met} {g}: TM7 {r.tm7:.3f} ctrl {r.control_mean:.3f} [{r.control_lo:.3f},{r.control_hi:.3f}] p {r.p:.4f}")
    b.set_xticks([0, 1], mets[0][1:] + mets[1][1:]); b.tick_params(axis="x", pad=14, length=0)
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0)); b.set_ylabel("share of TM7 / controls")
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (POS, GREY)], labels=["TM7", "matched controls (mean, 95% range)"],
             frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.3))
    b.set_ylim(0, 0.75)
    b.set_title("Main model, by environment and evidence tier", loc="left"); panel(b, "B")
    fig.tight_layout(); save(fig, out, "fig3_tm7")


def fig4(R, out):
    sr = pd.read_csv(f"{R}/study_replication_summary.tsv", sep="\t")
    sl = pd.read_csv(f"{R}/shortlist_summary.tsv", sep="\t")
    F = pd.read_csv(f"{R}/shortlist_final.tsv", sep="\t"); F = F[F.n_folds == 2]
    fig, (a, b) = mk((1.1, 1), 2.4, 4.9)
    g = ["all"] + ENVS; x = np.arange(len(g)); w = 0.36
    for k, (s, col) in enumerate([("network edges", POS), ("random same-env pairs", GREY)]):
        v = sr[sr.set == s].set_index("group").loc[g].pos_ge2
        a.bar(x + (k - 0.5) * w, v, w, color=col, label="main-model edges" if k == 0 else "random same-environment pairs")
        print(f"fig4 studyrep {s}:", v.round(3).to_dict())
    a.set_xticks(x, ["all"] + [ENV_SHORT[e] for e in ENVS]); a.set_ylim(0, 1.1)
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    a.set_ylabel("positive in ≥ 2 independent studies")
    a.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    a.set_title("Replication across independent studies", loc="left"); panel(a, "A")
    sets = [("shortlist (D1-D5)", "shortlist criteria", POS), ("D1 only (significant, r >= 0.2)", "significant only", "#92c5de"),
            ("random same-env, different order", "random pairs", GREY)]
    for k, (s, lbl, col) in enumerate(sets):
        v = sl[sl.set == s].set_index("fold").loc[["A", "B"]].share_validated_bh5
        b.bar(np.arange(2) + (k - 1) * 0.26, v, 0.26, color=col, label=lbl)
        print(f"fig4 heldout {s}:", v.round(3).to_dict())
    b.set_xticks([0, 1], ["fold A", "fold B"]); b.set_ylim(0, 1.1)
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    b.set_ylabel("confirmed in held-out studies (BH 5%)"); b.legend(frameon=False, loc="upper center", ncol=3,
                                                                    bbox_to_anchor=(0.5, -0.12))
    b.set_title("Held-out confirmation: selected vs baselines", loc="left"); panel(b, "B")
    env = F.S_dom_env.value_counts()
    print(f"fig4 both folds: {len(F)} pairs; by S environment:", env.to_dict(), "non-marine:", int((F.S_dom_env != "Saline").sum()))
    tm7 = F[(F.S_phylum == "Patescibacteria") | (F.H_phylum == "Patescibacteria")]
    print(f"fig4 TM7 pairs in shortlist: {len(tm7)}; partner Actinomycetota: {int((tm7.H_phylum == 'Actinomycetota').sum())}")
    fig.tight_layout(); save(fig, out, "fig4_replication")


def fig5(R, out):
    G = pd.read_csv(f"{R}/gem_scores.tsv", sep="\t"); G = G[G["medium"] == "minimal"].copy()
    G["lbl"] = G.pair.str.replace("Unknown ", "").str.replace(r" sp\d+", " sp.", regex=True).str.replace("+", " + ", regex=False)
    order = ["positive control", "one-way", "succinate", "H2", "methanogen", "TM7"]
    G["o"] = G.group.map({k: i for i, k in enumerate(order)}); G = G.sort_values(["o", "lbl"]).reset_index(drop=True)
    if COL:   # short labels so the plot keeps its width in one column
        G["lbl"] = [" + ".join(p.replace("Unknown ", "").split()[0].split("_")[0] for p in s.split("+"))
                    for s in G.pair]
    fig, axes = mk((1, 1), 2.7, 5.2, sharey=True)
    y = np.arange(len(G))
    for ax, m, title, letter in [(axes[0], "mip", "Interaction potential (MIP)", "A"),
                                 (axes[1], "mro", "Resource overlap (MRO)", "B")]:
        for i, r in G.iterrows():
            if r[f"{m}_ctrl_n"] > 0 and pd.notna(r[f"{m}_ctrl_min"]):
                ax.plot([r[f"{m}_ctrl_min"], r[f"{m}_ctrl_max"]], [i, i], color=GREY, lw=3, alpha=0.5, solid_capstyle="round")
                ax.scatter(r[f"{m}_ctrl_mean"], i, color=GREY, s=14, zorder=3)
            if pd.notna(r[m]):
                ax.scatter(r[m], i, color=ACC if r.group == "positive control" else POS, s=30, zorder=4,
                           marker="D" if r[f"{m}_above_all_ctrl"] else "o")
            else:
                ax.text(1.01, i, "no score", va="center", fontsize=6.5, color=DARK, transform=ax.get_yaxis_transform())
        ax.set_title(f"$\\bf{{{letter}}}$   {title}", loc="right" if COL else "left")
        ax.set_xlabel(f"{m.upper()}, minimal anaerobic medium")
        n_t = int(G[f"{m}_testable"].sum()); n_a = int(G[f"{m}_above_all_ctrl"].sum())
        print(f"fig5 {m}: testable {n_t}, above all controls {n_a}")
    for ax in (axes if COL else axes[:1]):
        ax.set_yticks(y, [r.lbl if COL else f"{r.lbl}  [{r.group}]" for r in G.itertuples()], fontsize=6.5)
        ax.tick_params(axis="y", labelleft=True)
    axes[0].invert_yaxis()
    h = [plt.Line2D([], [], marker="o", ls="", color=POS, label="linked pair"),
         plt.Line2D([], [], marker="o", ls="", color=ACC, label="positive control"),
         plt.Line2D([], [], marker="D", ls="", color=POS, label="above all its controls"),
         plt.Line2D([], [], color=GREY, lw=3, alpha=0.5, label="random partners (2–3): range, mean")]
    fig.legend(handles=h, frameon=False, ncol=2 if COL else 4, loc="lower center",
               bbox_to_anchor=(0.5, -0.06) if COL else (0.6, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, out, "fig5_gem")
    print(G[["lbl", "group", "mip", "mip_ctrl_mean", "mip_ctrl_max", "mro", "mro_ctrl_mean", "mro_ctrl_max"]].to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--layout", choices=["wide", "column", "both"], default="both",
                    help="wide: full-width figures; column: one-column figures (*_col)")
    a = ap.parse_args(argv)
    od = os.path.join(a.out, "figures", "report"); os.makedirs(od, exist_ok=True)
    global COL
    for col in {"wide": [False], "column": [True], "both": [False, True]}[a.layout]:
        COL = col
        for f in (fig1, fig2, fig3, fig4, fig5):
            f(a.out, od)


if __name__ == "__main__":
    main()
