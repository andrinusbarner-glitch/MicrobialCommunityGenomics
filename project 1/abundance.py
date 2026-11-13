"""Abundance layer: do co-occurring mOTUs also rise and fall together?

    python abundance.py [--data data] [--out results] [--min-n 30]

Run after p9_pipeline.py (and phylogeny.py for the relatedness check).

Method
  - Counts -> centred log-ratio (CLR) per sample, over the mOTUs detected in
    that sample. Library size cancels, so no depth strata are needed.
  - For each pair, Pearson r of CLR values over samples where BOTH are present,
    computed inside each environment x study stratum (protocols and kits bias
    abundances), only where >= --min-n joint samples. Fisher-z pooled over
    strata (weight n-3) -> z and pooled r. Pairs never reaching --min-n in any
    stratum are "untested".
  - BH-FDR over tested pairs.

It is used to grade the presence/absence (PA) results, not to replace them:
  abund_edge_validation.tsv   PA network edges: abundance-confirmed / -contradicted / untested
  abund_cluster_coherence.tsv per PA cluster: share of tested internal pairs that
                              co-vary positively vs negatively, against same-environment pairs
  abund_pairs_phylo.tsv       relatedness of abundance-positive / -negative pairs
                              (competition shows up as close relatives trading off)
  abund_network_edges.tsv.gz, abund_clusters.tsv, abund_labels.tsv
                              an abundance-only network (top-k partners, Louvain)
  abund_summary.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.metrics import adjusted_mutual_info_score

import p9_pipeline as p

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["positive (r >= min_r)", "positive (weak)", "not significant", "negative", "untested"]


def log_counts(data_dir, index, columns, chunksize=1000):
    """log(count) for the kept mOTUs (float32, 0 where absent) and, per sample,
    the mean log count over ALL detected mOTUs (the CLR reference)."""
    L = np.zeros((len(index), len(columns)), np.float32)
    pos = pd.Series(np.arange(len(index)), index=index)
    s_log = np.zeros(len(columns)); s_n = np.zeros(len(columns))
    for ch in pd.read_csv(f"{data_dir}/motus_subset_matrix.tsv.gz", sep="\t", index_col=0, chunksize=chunksize):
        ch = ch.reindex(columns=columns)
        C = ch.values.astype(np.float64)
        lg = np.where(C > 0, np.log(np.where(C > 0, C, 1)), 0.0)
        s_log += lg.sum(0); s_n += (C > 0).sum(0)
        keep = ch.index.isin(index)
        L[pos[ch.index[keep]].values] = lg[keep]
    return L, s_log / np.maximum(s_n, 1)


def fisher_pool(L, P, mean_log, strata, min_n, block):
    """Sum over strata of atanh(r)*(n-3) and of (n-3), pairwise-complete r."""
    M = L.shape[0]
    Zs = np.zeros((M, M), np.float32); Ws = np.zeros((M, M), np.float32)
    labels = pd.Series(strata)
    for k, (st, idx) in enumerate(labels.groupby(labels).groups.items()):
        cols = np.asarray(list(idx))
        if len(cols) < min_n:
            continue
        Ps = P[:, cols].astype(np.float32)
        Ls = ((L[:, cols] - mean_log[cols][None, :].astype(np.float32)) * Ps).astype(np.float32)  # CLR, 0 where absent
        cnt = Ps.sum(1)
        mu = np.where(cnt > 0, Ls.sum(1) / np.maximum(cnt, 1), 0).astype(np.float32)
        Ls = (Ls - mu[:, None]) * Ps                               # shift per mOTU: r unchanged, better precision
        live = np.where(cnt >= min_n)[0]                           # mOTUs that can reach min_n at all
        if len(live) < 2:
            continue
        Ls, Ps = Ls[live], Ps[live]
        Ls2 = Ls ** 2
        for i in range(0, len(live), block):
            b = slice(i, min(i + block, len(live)))
            n = Ps[b] @ Ps.T
            sx = Ls[b] @ Ps.T; sy = Ps[b] @ Ls.T
            num = n * (Ls[b] @ Ls.T) - sx * sy
            den = (n * (Ls2[b] @ Ps.T) - sx ** 2) * (n * (Ps[b] @ Ls2.T) - sy ** 2)
            r = np.clip(num / np.sqrt(np.maximum(den, 1e-12)), -0.999999, 0.999999)
            w = np.where(n >= min_n, n - 3.0, 0.0).astype(np.float32)
            rows = live[b]
            Zs[np.ix_(rows, live)] += np.arctanh(r).astype(np.float32) * w
            Ws[np.ix_(rows, live)] += w
        p.log(f"  stratum {k + 1}: {st} ({len(cols)} samples, {len(live)} mOTUs testable)")
    np.fill_diagonal(Zs, 0); np.fill_diagonal(Ws, 0)
    return Zs, Ws


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--min-n", type=int, default=30, help="min joint-presence samples within a stratum")
    ap.add_argument("--block", type=int, default=1000)
    ap.add_argument("--pairs", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    with open(f"{a.out}/diagnostics.json") as fh:
        prm = json.load(fh)["params"]
    pa, samples, _ = p.load(a.data, prm["min_prev"], prm["rich_bins"], prm["min_study"])
    mk = pd.read_csv(f"{a.out}/motus.tsv", sep="\t", index_col=0).loc[pa.index]
    M = len(pa)
    env = mk["dom_env"].values

    # ---- 1. pooled correlations
    L, mean_log = log_counts(a.data, pa.index, pa.columns)
    P = pa.values                                                 # int8 presence
    strata = samples[p.ABUND_STRATA_COL].values
    p.log(f"abundance correlations within {pd.unique(strata).size} strata ({p.ABUND_STRATA_COL}), min {a.min_n} joint samples")
    Zs, Ws = fisher_pool(L, P, mean_log, strata, a.min_n, a.block)
    del L
    tested = Ws > 0
    z = np.where(tested, Zs / np.sqrt(np.maximum(Ws, 1e-12)), 0).astype(np.float32)
    rbar = np.where(tested, np.tanh(Zs / np.maximum(Ws, 1e-12)), 0).astype(np.float32)
    del Zs, Ws
    def upper(rows):
        return np.arange(M)[None, :] > np.arange(rows.start, rows.stop)[:, None]

    vals = []
    for s in range(0, M, a.block):
        rows = slice(s, min(s + a.block, M))
        vals.append(np.abs(z[rows][tested[rows] & upper(rows)]))
    zz = np.concatenate(vals); del vals
    m = len(zz)
    zz.sort(); zz = zz[::-1]                                   # descending |z| = ascending p
    thr = np.inf
    for s in range(0, m, 5_000_000):
        q = zz[s:s + 5_000_000].astype(np.float64)
        ok = np.nonzero(2 * stats.norm.sf(q) <= prm["fdr"] * np.arange(s + 1, s + len(q) + 1) / m)[0]
        if len(ok):
            thr = float(q[ok[-1]])
    del zz
    sig = tested & (np.abs(z) >= thr)
    np.fill_diagonal(sig, False)
    n_pos = n_neg = 0
    for s in range(0, M, a.block):
        rows = slice(s, min(s + a.block, M))
        su = sig[rows] & upper(rows)
        n_pos += int((su & (z[rows] > 0)).sum()); n_neg += int((su & (z[rows] < 0)).sum())
    n_all = M * (M - 1) // 2
    p.log(f"tested pairs {m} of {n_all} ({m / n_all:.1%}); |z| >= {thr:.2f}; +{n_pos} -{n_neg}")

    def cls(i, j):
        zz_, rr_, tt_ = z[i, j], rbar[i, j], tested[i, j]
        s_ = np.abs(zz_) >= thr
        return np.select([~tt_, s_ & (zz_ > 0) & (rr_ >= prm["min_r"]), s_ & (zz_ > 0), s_ & (zz_ < 0)],
                         [CLASSES[4], CLASSES[0], CLASSES[1], CLASSES[3]], CLASSES[2])

    # ---- 2. validate PA network edges against random same-environment pairs
    pos = pd.Series(np.arange(M), index=mk.index)
    pools = {e: np.where(env == e)[0] for e in p.ENVS}
    ri, rj = [], []
    for e in p.ENVS:
        x = rng.choice(pools[e], a.pairs // 4); y = rng.choice(pools[e], a.pairs // 4)
        k = x != y; ri.append(x[k]); rj.append(y[k])
    ri, rj = np.concatenate(ri), np.concatenate(rj)
    rc = cls(ri, rj)
    vrows = []
    for model in p.MODELS:
        E = pd.read_csv(f"{a.out}/network_edges_{model}.tsv.gz", sep="\t")
        i, j = pos[E.a].values, pos[E.b].values
        c = cls(i, j)
        for scope in ["all"] + p.ENVS:
            sel = np.ones(len(i), bool) if scope == "all" else (env[i] == scope) & (env[j] == scope)
            bsel = np.ones(len(ri), bool) if scope == "all" else env[ri] == scope
            row = {"model": model, "scope": scope, "n_edges": int(sel.sum())}
            for k_ in CLASSES:
                row[k_] = float((c[sel] == k_).mean())
                row["baseline " + k_] = float((rc[bsel] == k_).mean())
            t_e, t_b = c[sel] != CLASSES[4], rc[bsel] != CLASSES[4]
            row["confirmed_of_tested"] = float(np.isin(c[sel][t_e], CLASSES[:2]).mean()) if t_e.any() else np.nan
            row["baseline_confirmed_of_tested"] = float(np.isin(rc[bsel][t_b], CLASSES[:2]).mean())
            row["contradicted_of_tested"] = float((c[sel][t_e] == CLASSES[3]).mean()) if t_e.any() else np.nan
            row["baseline_contradicted_of_tested"] = float((rc[bsel][t_b] == CLASSES[3]).mean())
            row["median_r_tested"] = float(np.median(rbar[i[sel], j[sel]][t_e])) if t_e.any() else np.nan
            vrows.append(row)
    V = pd.DataFrame(vrows)
    V.to_csv(f"{a.out}/abund_edge_validation.tsv", sep="\t", index=False)
    vm = V[(V.model == p.MAIN) & (V.scope == "all")].iloc[0]
    p.log(f"PA edges ({p.MAIN}): {vm['untested']:.0%} untested; of tested {vm['confirmed_of_tested']:.0%} abundance-positive "
          f"(baseline {vm['baseline_confirmed_of_tested']:.0%}), {vm['contradicted_of_tested']:.0%} negative "
          f"(baseline {vm['baseline_contradicted_of_tested']:.0%})")

    # ---- 3. cluster coherence
    lab = mk[f"cluster_{p.MAIN}"].fillna(-1).astype(int).values
    cl = pd.read_csv(f"{a.out}/clusters_{p.MAIN}.tsv", sep="\t").set_index("cluster")
    crow = []
    for c_ in sorted(set(lab[lab >= 0])):
        mem = np.where(lab == c_)[0]
        sub_t = tested[np.ix_(mem, mem)]; sub_s = sig[np.ix_(mem, mem)]
        sub_z = z[np.ix_(mem, mem)]; sub_r = rbar[np.ix_(mem, mem)]
        tri = np.triu(np.ones_like(sub_t), 1).astype(bool)
        t_ = sub_t & tri
        de = cl.loc[c_, "dominant_env"]
        bsel = (env[ri] == de) & (rc != CLASSES[4])
        crow.append({"cluster": c_, "n_motus": len(mem), "top_family": cl.loc[c_, "top_family"], "dominant_env": de,
                     "frac_pairs_tested": float(t_.sum() / tri.sum()),
                     "frac_positive": float((sub_s & (sub_z > 0) & t_).sum() / max(t_.sum(), 1)),
                     "frac_negative": float((sub_s & (sub_z < 0) & t_).sum() / max(t_.sum(), 1)),
                     "mean_r": float(sub_r[t_].mean()) if t_.any() else np.nan,
                     "env_frac_positive": float(np.isin(rc[bsel], CLASSES[:2]).mean()),
                     "env_frac_negative": float((rc[bsel] == CLASSES[3]).mean()),
                     "env_mean_r": float(rbar[ri[bsel], rj[bsel]].mean())})
    Co = pd.DataFrame(crow)
    Co.to_csv(f"{a.out}/abund_cluster_coherence.tsv", sep="\t", index=False)

    # ---- 4. abundance-only network (same rules as the PA network)
    I, J, W = [], [], []
    for s in range(0, M, a.block):
        rows = slice(s, min(s + a.block, M))
        cand = np.where(sig[rows] & (z[rows] > 0) & (rbar[rows] >= prm["min_r"]), rbar[rows], 0.0)
        x, y, w = p.knn_edges(cand, rows, prm["knn"])
        I.append(x); J.append(y); W.append(w)
    I, J, W = np.concatenate(I), np.concatenate(J), np.concatenate(W)
    lo, hi = np.minimum(I, J), np.maximum(I, J)
    _, u = np.unique(lo.astype(np.int64) * M + hi, return_index=True)
    lo, hi, W = lo[u], hi[u], W[u]
    G = nx.Graph(); G.add_weighted_edges_from(zip(lo.tolist(), hi.tolist(), W.astype(float).tolist()))
    comms = p.louvain(G, prm["resolution"], 0)
    alab = p.labels_from(comms, M)
    deg = np.bincount(np.concatenate([lo, hi]), minlength=M)
    pd.DataFrame({"a": pa.index[lo], "b": pa.index[hi], "r": W, "z": z[lo, hi]}).to_csv(
        f"{a.out}/abund_network_edges.tsv.gz", sep="\t", index=False, float_format="%.4g")
    p.cluster_table(alab, mk, deg).to_csv(f"{a.out}/abund_clusters.tsv", sep="\t", index=False)
    pd.DataFrame({"cluster_abund": alab, "abund_degree": deg}, index=pa.index).to_csv(f"{a.out}/abund_labels.tsv", sep="\t")
    both = (alab >= 0) & (lab >= 0)
    env_parts = [q for q in (set(np.where(env == e)[0]) & set(G.nodes) for e in p.ENVS) if q]
    net = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "n_clusters": len(comms),
           "n_clusters_ge10": sum(len(c) >= 10 for c in comms),
           "Q_louvain": nx.community.modularity(G, comms, weight="weight") if comms else np.nan,
           "Q_env": nx.community.modularity(G, env_parts, weight="weight") if env_parts else np.nan,
           "AMI_vs_PA_main": float(adjusted_mutual_info_score(lab[both], alab[both])) if both.any() else np.nan,
           "AMI_cluster_env": float(adjusted_mutual_info_score(env[alab >= 0], alab[alab >= 0])) if comms else np.nan}
    p.log(f"abundance network: {net['nodes']} nodes, {net['edges']} edges, {net['n_clusters']} clusters, "
          f"AMI vs PA {net['AMI_vs_PA_main']:.2f}, AMI vs env {net['AMI_cluster_env']:.2f}")

    # ---- 5. relatedness of abundance-positive vs -negative pairs (competition check)
    try:
        import phylogeny
        tax = pd.read_csv(f"{a.data}/motus_subset_taxonomy.tsv", sep="\t").set_index("MOTU")
        tree = os.path.join(a.data, "mOTUs_tree_bacteria.nwk")
        dist, okd, what = phylogeny.make_distance(tree, mk, tax)
        keep = okd[ri] & okd[rj]
        d = dist(ri[keep], rj[keep]); c2 = rc[keep]; e2 = env[ri][keep]
        prow = []
        for scope in ["all"] + p.ENVS:
            sel = np.ones(len(d), bool) if scope == "all" else e2 == scope
            for k_ in CLASSES:
                mm = sel & (c2 == k_)
                if mm.sum() < 30:
                    continue
                boot = [d[rng.choice(np.where(mm)[0], mm.sum())].mean() for _ in range(200)]
                prow.append({"scope": scope, "class": k_, "n_pairs": int(mm.sum()), "mean_dist": float(d[mm].mean()),
                             "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))})
        Ph = pd.DataFrame(prow)
        Ph.to_csv(f"{a.out}/abund_pairs_phylo.tsv", sep="\t", index=False)
    except SystemExit as ex:
        what, Ph = str(ex), None

    summ = {"min_n": a.min_n, "n_strata": int(pd.unique(strata).size), "strata": p.ABUND_STRATA_COL, "z_threshold": thr,
            "n_pairs": int(M * (M - 1) // 2), "n_tested": int(m), "sig_pos": n_pos, "sig_neg": n_neg,
            "network": net, "relatedness": what, "runtime_min": (time.time() - t0) / 60}
    with open(f"{a.out}/abund_summary.json", "w") as fh:
        json.dump(summ, fh, indent=1, default=float)
    pd.set_option("display.width", 220)
    cols = ["model", "scope", "n_edges", "untested", "confirmed_of_tested", "baseline_confirmed_of_tested",
            "contradicted_of_tested", "baseline_contradicted_of_tested", "median_r_tested"]
    print(V[cols].round(3).to_string(index=False))
    print(Co.round(3).to_string(index=False))
    if Ph is not None:
        print(Ph[Ph.scope == "all"].round(4).to_string(index=False))
    print(json.dumps(summ, indent=1, default=float))
    p.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
