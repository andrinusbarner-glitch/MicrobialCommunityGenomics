"""
p3_viz.py - figures for the streamlined P3 pipeline (p3_pipeline.py).

It only READS the pipeline's outputs; clusters are never recomputed here, so
every figure matches the tables in results/.

    import p3_pipeline as p, p3_viz as v
    R = v.load_results("results")                 # tables written by the pipeline
    pa, samples, motus = p.load("data")           # only needed for the sample PCoA
    v.plot_environment_overview(pa, R)
    v.plot_richness_diagnostics(R)
    v.plot_edge_fate(R, "global->env_rich")
    net = v.network("results", R, "env_rich")    # adds a force layout
    v.plot_network(net, R, "env")

Or run  python make_figures.py  to write every figure to results/figures/.
"""
import json
import os

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.metrics import adjusted_mutual_info_score

from p3_pipeline import ENVS, MODELS, MAIN

ENV_COLORS = {"Human-associated": "#c8435a", "Animal-associated": "#d99a2b",
              "Non-saline": "#1f8a70", "Saline": "#2f5d9e"}
SHORT = ["Human", "Animal", "Non-sal.", "Saline"]
LABEL = {"global": "global (no strata)", "env": "environment",
         "env_rich": "environment x richness"}


# ---------------------------------------------------------------- loading
def load_results(out):
    out = out.rstrip("/\\")
    motus = pd.read_csv(f"{out}/motus.tsv", sep="\t", index_col=0)
    R = {"out": out, "motus": motus, "motus_k": motus[motus.kept].copy(),
         "samples": pd.read_csv(f"{out}/samples.tsv", sep="\t", index_col=0),
         "summary": pd.read_csv(f"{out}/run_summary.tsv", sep="\t", index_col="model"),
         "fate": pd.read_csv(f"{out}/edge_fate.tsv", sep="\t", index_col=0),
         "rhist": pd.read_csv(f"{out}/effect_size_hist.tsv", sep="\t")}
    with open(f"{out}/diagnostics.json") as fh:
        R["diag"] = json.load(fh)
    return R


def clusters(R, model):
    return pd.read_csv(f"{R['out']}/clusters_{model}.tsv", sep="\t")


# ---------------------------------------------------------------- 1. environment
def sample_pcoa(pa, samples, n_per_env=900, seed=1, ax=None):
    """Balanced subsample -> Jaccard -> PERMANOVA-style R2 for environment + PCoA."""
    env = samples["environment"].reindex(pa.columns).values
    rng = np.random.default_rng(seed)
    sel = np.concatenate([rng.choice(np.where(env == e)[0], n_per_env, replace=False) for e in ENVS])
    S = pa.values[:, sel].T.astype(np.float32)
    inter = S @ S.T
    rs = S.sum(1)
    D = 1 - inter / np.maximum(rs[:, None] + rs[None, :] - inter, 1)
    g = env[sel]
    n = len(sel)
    sst = (D ** 2).sum() / (2 * n)
    ssw = sum((D[np.ix_(g == e, g == e)] ** 2).sum() / (2 * (g == e).sum()) for e in ENVS)
    R2 = 1 - ssw / sst
    J = np.eye(n) - 1 / n
    w, V = np.linalg.eigh(J @ (-0.5 * D ** 2) @ J)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    pc = V[:, :2] * np.sqrt(w[:2])
    ve = w[:2] / w[w > 0].sum()
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    for e in ENVS:
        m = g == e
        ax.scatter(pc[m, 0], pc[m, 1], s=3, color=ENV_COLORS[e], label=e, alpha=0.6, lw=0)
    ax.set_xlabel(f"PCoA 1 ({ve[0]:.1%})"); ax.set_ylabel(f"PCoA 2 ({ve[1]:.1%})")
    ax.set_title(f"Jaccard PCoA of samples, environment R2 = {R2:.2f}")
    ax.legend(frameon=False, fontsize=8, markerscale=4)
    return R2, ax


def plot_environment_overview(pa, R):
    m = R["motus_k"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    bins = np.linspace(0.25, 1, 31)
    bottom = np.zeros(len(bins) - 1)
    for e in ENVS:
        h, _ = np.histogram(m.loc[m.dom_env == e, "purity"], bins=bins)
        ax[0].bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge",
                  color=ENV_COLORS[e], label=e, edgecolor="white", linewidth=0.3)
        bottom += h
    ax[0].set_xlabel("habitat purity (max environment affinity)"); ax[0].set_ylabel("mOTUs")
    ax[0].set_title(f"median purity = {m.purity.median():.2f}"); ax[0].legend(frameon=False, fontsize=8)
    R2, _ = sample_pcoa(pa, R["samples"], ax=ax[1])
    ax[2].hist([m.loc[m.dom_env == e, "eta2_env"] for e in ENVS], bins=40, stacked=True,
               color=[ENV_COLORS[e] for e in ENVS], label=ENVS)
    ax[2].set_xlabel("eta² (environment)"); ax[2].set_ylabel("mOTUs")
    ax[2].set_title(f"median eta² = {m.eta2_env.median():.2f}")
    fig.tight_layout()
    return fig, R2


# ---------------------------------------------------------------- 2. richness / depth diagnostics
def plot_richness_diagnostics(R):
    """(a) richness vs read depth, (b) richness by environment with the quartile
    strata, (c) does a mOTU's number of significant partners just follow its prevalence?"""
    s, m = R["samples"], R["motus_k"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for e in ENVS:
        g = s[s.environment == e]
        rho = R["diag"]["richness_vs_reads_spearman"][e]
        ax[0].scatter(g.reads, g.richness, s=2, alpha=0.35, color=ENV_COLORS[e], lw=0,
                      label=f"{e} (rho {rho:.2f})")
    ax[0].set_xscale("log"); ax[0].set_xlabel("reads per sample"); ax[0].set_ylabel("mOTUs detected")
    ax[0].set_title("richness vs sequencing depth"); ax[0].legend(frameon=False, fontsize=7, markerscale=5)

    data = [s.loc[s.environment == e, "richness"] for e in ENVS]
    parts = ax[1].violinplot(data, showextrema=False)
    for pc, e in zip(parts["bodies"], ENVS):
        pc.set_facecolor(ENV_COLORS[e]); pc.set_alpha(0.6)
    for k, e in enumerate(ENVS):
        g = s[s.environment == e]
        for b in sorted(g.rich_bin.unique())[1:]:
            y = g.loc[g.rich_bin == b, "richness"].min()
            ax[1].plot([k + 0.75, k + 1.25], [y, y], color="k", lw=0.8)
    ax[1].set_xticks(range(1, 5), SHORT); ax[1].set_ylabel("mOTUs detected per sample")
    ax[1].set_title("richness strata (lines = quartile cuts)")

    for n, c in zip(MODELS, ["#9aa3ad", "#d99a2b", "#3b4fd9"]):
        rho = R["summary"].loc[n, "spearman_sigdegree_prevalence"]
        ax[2].scatter(m.prevalence, m[f"sigdegree_{n}"], s=1.5, alpha=0.3, color=c, lw=0,
                      label=f"{LABEL[n]} (rho {rho:.2f})")
    ax[2].set_xscale("log"); ax[2].set_xlabel("mOTU prevalence (samples)")
    ax[2].set_ylabel("significant positive partners")
    ax[2].set_title("are hubs just common mOTUs?")
    ax[2].legend(frameon=False, fontsize=7, markerscale=6)
    fig.tight_layout()
    return fig


def plot_effect_sizes(R, min_r=None):
    """Distribution of the effect size r over all significant pairs, per model."""
    h = R["rhist"]
    mid = (h.bin_lo + h.bin_hi) / 2
    min_r = min_r if min_r is not None else R["diag"]["params"]["min_r"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, c in zip(MODELS, ["#9aa3ad", "#d99a2b", "#3b4fd9"]):
        ax.plot(mid, h[n], color=c, label=LABEL[n], lw=1.4)
    ax.axvline(float(min_r), color="k", ls="--", lw=0.8, label=f"network cut-off r = {min_r}")
    ax.set_yscale("log"); ax.set_xlim(-0.5, 1)
    ax.set_xlabel("effect size r (within-stratum phi)"); ax.set_ylabel("significant pairs")
    ax.set_title("most significant pairs are very weak associations")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 3. what stratifying removes
def plot_edge_fate(R, comparisons=("global->env", "env->env_rich"), ax=None):
    """Stacked bars per comparison: kept / flipped / lost (right), new (left)."""
    F = R["fate"]
    cols = {"kept": "#4a5a6a", "flipped": "#d99a2b", "lost": "#c9cfd6", "new": "#1f8a70"}
    ax = ax or plt.subplots(figsize=(9, 1.2 + 1.1 * len(comparisons)))[1]
    ylab, y = [], 0
    for cmp in comparisons:
        for s in ["pos", "neg"]:
            left = 0
            for k in ["kept", "flipped", "lost"]:
                v = F.loc[cmp, f"{s}_{k}"] / 1e6
                ax.barh(y, v, left=left, color=cols[k], label=k if y == 0 else None)
                left += v
            ax.barh(y, -F.loc[cmp, f"{s}_new"] / 1e6, color=cols["new"],
                    label="new" if y == 0 else None)
            a_, b_ = cmp.split("->")
            ylab.append(f"{'positive' if s == 'pos' else 'negative'}: {a_} -> {b_}")
            y += 1
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(ylab)), ylab); ax.invert_yaxis()
    ax.set_xlabel("million significant pairs  (left of 0: significant only after stratifying)")
    ax.set_title("What happens to significant pairs when strata are added", pad=24)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower right", bbox_to_anchor=(1, 1.02))
    return ax


def plot_envpair_density(R, models=MODELS):
    fig, axes = plt.subplots(len(models), 2, figsize=(9, 3.9 * len(models)), squeeze=False)
    for r, n in enumerate(models):
        for c, (s, cmap) in enumerate([("pos", "Blues"), ("neg", "Reds")]):
            D = pd.read_csv(f"{R['out']}/envpair_density_{n}_{s}.tsv", sep="\t", index_col=0).values
            ax = axes[r, c]
            vmax = max(D.max(), 1e-6)
            ax.imshow(D, vmin=0, vmax=vmax, cmap=cmap)
            for (x, y), v in np.ndenumerate(D):
                ax.text(y, x, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 * vmax else "black")
            ax.set_xticks(range(4), SHORT, rotation=30); ax.set_yticks(range(4), SHORT)
            ax.set_title(f"{LABEL[n]}: {'positive' if s == 'pos' else 'negative'} pair density")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 4. networks
def _topk(a, b, w, M, k):
    src = np.concatenate([a, b]); dst = np.concatenate([b, a]); ww = np.concatenate([w, w])
    o = np.lexsort((-ww, src)); src, dst, ww = src[o], dst[o], ww[o]
    start = np.searchsorted(src, np.arange(M))
    keep = (np.arange(len(src)) - start[src]) < k
    x, y, ww = src[keep], dst[keep], ww[keep]
    lo, hi = np.minimum(x, y), np.maximum(x, y)
    _, u = np.unique(lo.astype(np.int64) * M + hi, return_index=True)
    return lo[u], hi[u], ww[u]


def fast_layout(H, init, iters=150, grid=48):
    """Fruchterman-Reingold with grid-approximated repulsion (fast for ~10k nodes)."""
    nl = list(H.nodes)
    X = np.array([init[n] for n in nl], float)
    X -= X.mean(0); X /= np.abs(X).max()
    A = nx.to_scipy_sparse_array(H, nodelist=nl, weight="weight", format="coo")
    data = A.data / A.data.max()
    N = len(nl); kopt = 0.6 / np.sqrt(N); temp = 0.05
    for _ in range(iters):
        lo = X.min(0); span = X.max(0) - lo + 1e-9
        cell = np.minimum(((X - lo) / span * grid).astype(int), grid - 1)
        cid = cell[:, 0] * grid + cell[:, 1]
        mass = np.bincount(cid, minlength=grid * grid).astype(float)
        nz = mass > 0
        Cc = np.stack([np.bincount(cid, X[:, 0], grid * grid)[nz],
                       np.bincount(cid, X[:, 1], grid * grid)[nz]], 1) / mass[nz, None]
        disp = np.zeros_like(X)
        for s in range(0, N, 2500):
            b = slice(s, min(N, s + 2500))
            d = X[b, None, :] - Cc[None]
            r2 = (d ** 2).sum(-1) + (0.25 * kopt) ** 2
            disp[b] += (d * (mass[nz][None] * kopt ** 2 / r2)[..., None]).sum(1)
        dd = X[A.row] - X[A.col]; dist = np.sqrt((dd ** 2).sum(1)) + 1e-9
        np.add.at(disp, A.row, -dd * ((dist ** 2 / kopt) * data / dist)[:, None])
        disp -= X * 0.02 * N * kopt
        L = np.sqrt((disp ** 2).sum(1)) + 1e-9
        X += disp / L[:, None] * np.minimum(L, temp)[:, None]
        temp *= 0.975
    return {n: X[k] for k, n in enumerate(nl)}


def network(out, R, model, k_vis=8, seed=0):
    """Read the pipeline's network + cluster labels and compute a layout.
    The drawing uses each node's k_vis strongest partners; clusters come from the
    pipeline (top-knn graph), they are not recomputed."""
    mk = R["motus_k"]
    M = len(mk)
    pos = pd.Series(np.arange(M), index=mk.index)
    E = pd.read_csv(f"{out}/network_edges_{model}.tsv.gz", sep="\t")
    a, b, w = pos[E.a].values, pos[E.b].values, E.r.values
    lab = mk[f"cluster_{model}"].fillna(-1).astype(int).values
    va, vb, vw = _topk(a, b, w, M, k_vis)
    H = nx.Graph()
    H.add_nodes_from(np.unique(np.concatenate([a, b])).tolist())
    H.add_weighted_edges_from(zip(va.tolist(), vb.tolist(), vw.astype(float).tolist()))
    # cluster-level layout first, then every node starts at its cluster's position
    ncl = lab.max() + 1
    la, lb = lab[a], lab[b]
    ok = (la >= 0) & (lb >= 0) & (la != lb)
    mw = (pd.DataFrame({"x": np.minimum(la[ok], lb[ok]), "y": np.maximum(la[ok], lb[ok]), "w": w[ok]})
          .groupby(["x", "y"]).w.sum())
    C = nx.Graph(); C.add_nodes_from(range(ncl))
    for (x, y), q in mw.items():
        C.add_edge(int(x), int(y), weight=float(np.log1p(q)))
    cp = nx.spring_layout(C, weight="weight", seed=seed, iterations=200, k=1.5 / np.sqrt(max(ncl, 1)))
    sizes = np.bincount(lab[lab >= 0], minlength=ncl)
    rng = np.random.default_rng(seed)
    init = {n: cp[lab[n]] + rng.normal(size=2) * 0.02 * np.sqrt(sizes[lab[n]] / sizes.max())
            for n in H.nodes}
    xy = np.full((M, 2), np.nan)
    for n, p in fast_layout(H, init).items():
        xy[n] = p
    return {"model": model, "H": H, "labels": lab, "xy": xy, "vis_edges": (va, vb)}


def plot_network(net, R, color_by="env", ax=None, s=2, title=None):
    """color_by: 'env' | 'cluster' | 'phylum' | any numeric column of motus (e.g. 'breadth')."""
    mk = R["motus_k"]
    xy = net["xy"]
    ax = ax or plt.subplots(figsize=(9, 9))[1]
    va, vb = net["vis_edges"]
    seg = np.stack([xy[va], xy[vb]], 1)
    seg = seg[~np.isnan(seg).any((1, 2))]
    ax.add_collection(LineCollection(seg[::3], colors="#8a8f98", linewidths=0.15, alpha=0.18))
    ok = ~np.isnan(xy[:, 0])
    if color_by == "env":
        c = mk.dom_env.map(ENV_COLORS).values
        for e in ENVS:
            ax.scatter([], [], color=ENV_COLORS[e], label=e, s=20)
        ax.scatter(xy[ok, 0], xy[ok, 1], c=c[ok], s=s, lw=0)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    elif color_by == "cluster":
        lab = net["labels"]
        ax.scatter(xy[ok, 0], xy[ok, 1], c=lab[ok] % 20, cmap="tab20", s=s, lw=0)
        for c in np.unique(lab[lab >= 0])[:25]:
            m = (lab == c) & ok
            if m.sum():
                ax.text(*np.median(xy[m], 0), str(c), fontsize=9, weight="bold", ha="center")
    elif color_by == "phylum":
        top = mk.phylum.value_counts().index[:9]
        ph = mk.phylum.where(mk.phylum.isin(top), "other").values
        pal = dict(zip(list(top) + ["other"], list(plt.cm.tab10.colors[:9]) + [(0.8, 0.8, 0.8)]))
        for p in pal:
            m = (ph == p) & ok
            ax.scatter(xy[m, 0], xy[m, 1], color=pal[p], s=s, lw=0, label=p)
        ax.legend(frameon=False, loc="upper left", fontsize=7, markerscale=4)
    else:
        sc = ax.scatter(xy[ok, 0], xy[ok, 1], c=mk[color_by].values[ok], cmap="viridis", s=s, lw=0)
        plt.colorbar(sc, ax=ax, shrink=0.5, label=color_by)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or f"{LABEL[net['model']]}: positive network, coloured by {color_by}")
    return ax


def plot_cluster_env_heatmap(R, model=MAIN, max_clusters=30, rank="family"):
    T = clusters(R, model).head(max_clusters)
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(T) + 1.6))
    im = ax.imshow(T[ENVS].values, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(4), SHORT)
    ax.set_yticks(range(len(T)), [f"{c}  (n={n}, {t} {f:.0%}, prev {p:.0f})" for c, n, t, f, p in
                                  zip(T.cluster, T.n_motus, T[f"top_{rank}"], T[f"top_{rank}_frac"],
                                      T.median_prevalence)], fontsize=8)
    for (r, c), v in np.ndenumerate(T[ENVS].values):
        if v > 0.05:
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 else "black")
    ax.set_title(f"{LABEL[model]}: environment profile of clusters")
    plt.colorbar(im, ax=ax, shrink=0.6, label="mean affinity")
    fig.tight_layout()
    return fig


def plot_partition_comparison(R, model_a="env", model_b=MAIN, max_clusters=20):
    mk = R["motus_k"]
    la = mk[f"cluster_{model_a}"].fillna(-1).astype(int).values
    lb = mk[f"cluster_{model_b}"].fillna(-1).astype(int).values
    m = (la >= 0) & (lb >= 0)
    ct = pd.crosstab(la[m], lb[m])
    ct = ct.loc[ct.index[:max_clusters], ct.columns[:max_clusters]]
    ctn = ct.div(ct.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(ctn.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(ct.shape[1]), ct.columns); ax.set_yticks(range(ct.shape[0]), ct.index)
    ax.set_xlabel(f"{LABEL[model_b]} cluster"); ax.set_ylabel(f"{LABEL[model_a]} cluster")
    ax.set_title(f"cluster correspondence (AMI = {adjusted_mutual_info_score(la[m], lb[m]):.2f}, row-normalised)")
    fig.tight_layout()
    return fig


def export_gephi(net, R, path):
    """GraphML (drawn edges, attributes, layout) for Gephi / Cytoscape."""
    mk = R["motus_k"]
    H = net["H"].copy()
    attrs = mk[["dom_env", "purity", "breadth", "eta2_env", "prevalence", "phylum", "family", "genus"]]
    for n in H.nodes:
        r = attrs.iloc[n]
        H.nodes[n].update({"label": mk.index[n], "cluster": int(net["labels"][n]),
                           "x": float(net["xy"][n, 0] * 1000), "y": float(net["xy"][n, 1] * 1000),
                           **{k: (v if isinstance(v, str) else float(v)) for k, v in r.items() if pd.notna(v)}})
    nx.write_graphml(H, path)
    return path
