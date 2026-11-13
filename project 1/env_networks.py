"""Separate, comparable co-occurrence networks for the four environments (P9).

    python env_networks.py [--reps 10] [--frac 0.9] [--min-det 20]

Run after p9_pipeline.py and habitats.py. For each environment:

  Equal sample size. Environments differ in sample number (Non-saline 1,817 vs
  Saline 4,334) and power grows with sample number, so raw edge counts are not
  comparable. Every environment is subsampled without replacement to the same
  n = --frac x (smallest environment), --reps times; all statistics are means
  (and SDs) over these repeats.

  Tested mOTUs: those with >= --min-det detections in the subsample.

  Two null models (same test as p9_pipeline.py: hypergeometric, BH-FDR, effect
  size r = within-stratum phi):
    uncontrolled  one stratum = the environment. Pairs that share a sub-habitat
                  (gut vs oral, wastewater vs lake) still count as associated.
    controlled    sub-habitat x study x richness quartile strata (as the main model).
                  The conservative network: residual associations.

  Per repeat (env_network_reps.tsv):
    nodes (tested mOTUs), significant positive / negative pairs, their ratio and
    their share of all pairs; the same shares among pairs of mOTUs with 5-50 %
    prevalence (prevalence-matched density, since rare and ubiquitous mOTUs have
    little power); network as in the main pipeline (r >= --min-r, kNN, Louvain):
    nodes, edges, density, clusters, clusters >= 10, modularity; Spearman
    correlation of positive degree with prevalence.

  Across repeats:
    env_network_summary.tsv    mean and SD of every statistic
    env_network_stability.tsv  reproducibility of significant pairs between repeats
                               (mean Jaccard; share of pairs found in >= 80 % of repeats)
    env_network_motus.tsv      per mOTU: mean positive / negative degree, prevalence
    env_network_edges_<env>.tsv.gz  consensus network (kNN edges in >= half the repeats)
    env_network_phylo.tsv      relatedness of consensus positive edges and of
                               consensus negative pairs, vs (a) a configuration null
                               that keeps every mOTU's degree (so prevalence and hub
                               structure are preserved) and (b) random tested pairs

  Sub-habitats. The four categories are mixtures (Human = gut + oral, Animal =
  pig + cattle + chicken + goat gut, Non-saline = 59 % wastewater + lake + ...), so
  a category network partly rediscovers its sub-habitats. The same comparison is
  therefore repeated for every sub-habitat with >= --min-habitat samples (human gut,
  human oral, pig gut, bovine gut, wastewater, marine), again at one common sample
  size; "controlled" then means study x richness within the sub-habitat.
  -> env_network_*_habitat.tsv. Smaller sub-habitats (lake water, goat and chicken
  gut) cannot be compared at equal power; true freshwater is therefore not covered.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

import p9_pipeline as p

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {"uncontrolled": None, "controlled": "stratum_hab_study_rich"}


def run_one(X, strata, a, seed):
    """One network on subsample X (mOTU x sample, float32, tested mOTUs only)."""
    M, N = X.shape
    O = p.cooccurrence(X, a.block)
    model = p.NullModel(X, strata if strata is not None else np.zeros(N), "sub")
    thr = p.bh_threshold(model, O, a.block, a.fdr)
    prev = X.sum(1) / N
    win = (prev >= 0.05) & (prev <= 0.5)
    pos_deg, neg_deg = np.zeros(M, np.int64), np.zeros(M, np.int64)
    P_, N_, knn = [], [], ([], [], [])
    c = {"win_pairs": int(win.sum() * (win.sum() - 1) // 2), "win_pos": 0, "win_neg": 0, "pos_r": 0}
    for i in range(0, M, a.block):
        rows = slice(i, min(i + a.block, M))
        nb = rows.stop - rows.start
        z, r = model.block(O[rows], rows)
        sig = np.abs(z) >= thr
        sig[np.arange(nb), np.arange(rows.start, rows.stop)] = False
        pos, neg = sig & (z > 0), sig & (z < 0)
        upper = np.arange(M)[None, :] > np.arange(rows.start, rows.stop)[:, None]
        pos_deg[rows] += pos.sum(1); neg_deg[rows] += neg.sum(1)
        pu, nu = pos & upper, neg & upper
        ww = win[rows][:, None] & win[None, :]
        c["win_pos"] += int((pu & ww).sum()); c["win_neg"] += int((nu & ww).sum())
        c["pos_r"] += int((pu & (r >= a.min_r)).sum())
        ii, jj = np.nonzero(pu); P_.append(np.c_[ii + rows.start, jj])
        ii, jj = np.nonzero(nu & (r <= -a.min_r)); N_.append(np.c_[ii + rows.start, jj])
        I, J, W = p.knn_edges(np.where(pos & (r >= a.min_r), r, 0.0), rows, a.knn)
        knn[0].append(I); knn[1].append(J); knn[2].append(W)
    del O
    I, J, W = (np.concatenate(v) for v in knn)
    lo, hi = np.minimum(I, J), np.maximum(I, J)
    _, u = np.unique(lo.astype(np.int64) * M + hi, return_index=True)
    lo, hi, W = lo[u], hi[u], W[u]
    G = nx.Graph(); G.add_weighted_edges_from(zip(lo.tolist(), hi.tolist(), W.astype(float).tolist()))
    comms = p.louvain(G, a.resolution, seed) if G.number_of_edges() else []
    n_g = G.number_of_nodes()
    npairs = M * (M - 1) / 2
    sp, sn = int(pos_deg.sum() // 2), int(neg_deg.sum() // 2)
    st = {"z_threshold": thr, "nodes_tested": M, "sig_pos": sp, "sig_neg": sn,
          "pos_neg_ratio": sp / max(sn, 1), "frac_pos": sp / npairs, "frac_neg": sn / npairs,
          "frac_pos_r": c["pos_r"] / npairs,
          "frac_pos_prev_matched": c["win_pos"] / max(c["win_pairs"], 1),
          "frac_neg_prev_matched": c["win_neg"] / max(c["win_pairs"], 1),
          "n_prev_matched_motus": int(win.sum()),
          "net_nodes": n_g, "net_edges": G.number_of_edges(),
          "net_density": G.number_of_edges() / max(n_g * (n_g - 1) / 2, 1),
          "net_mean_degree": 2 * G.number_of_edges() / max(n_g, 1),
          "n_clusters": len(comms), "n_clusters_ge10": sum(len(x) >= 10 for x in comms),
          "modularity": nx.community.modularity(G, comms, weight="weight") if comms else np.nan,
          "largest_cluster_share": max((len(x) for x in comms), default=0) / max(n_g, 1),
          "spearman_posdeg_prev": stats.spearmanr(pos_deg, prev).correlation,
          "spearman_negdeg_prev": stats.spearmanr(neg_deg, prev).correlation}
    net_deg = np.bincount(np.concatenate([lo, hi]), minlength=M)
    return st, prev, pos_deg, neg_deg, net_deg, np.concatenate(P_), np.concatenate(N_), (lo, hi, W)


def jaccard_stats(key_sets, n_reps):
    allk = np.concatenate(key_sets)
    uk, cnt = np.unique(allk, return_counts=True)
    jac = []
    for i in range(len(key_sets)):
        for j in range(i + 1, len(key_sets)):
            inter = np.intersect1d(key_sets[i], key_sets[j], assume_unique=True).size
            jac.append(inter / max(len(key_sets[i]) + len(key_sets[j]) - inter, 1))
    return {"mean_jaccard": float(np.mean(jac)) if jac else np.nan,
            "frac_in_80pct_reps": float((cnt >= 0.8 * n_reps).mean()) if len(cnt) else np.nan,
            "n_union": int(len(uk)), "n_consensus": int((cnt >= n_reps / 2).sum())}, uk, cnt


def phylo_test(a_idx, b_idx, tested, dist, ok, rng, reps, max_pairs=200_000):
    """Mean relatedness of pairs (a, b) vs a degree-preserving configuration null and random pairs."""
    m = ok[a_idx] & ok[b_idx]
    a_idx, b_idx = a_idx[m], b_idx[m]
    if len(a_idx) < 50:
        return None
    if len(a_idx) > max_pairs:
        s = rng.choice(len(a_idx), max_pairs, replace=False); a_idx, b_idx = a_idx[s], b_idx[s]
    obs = float(dist(a_idx, b_idx).mean())
    stubs = np.concatenate([a_idx, b_idx])
    null = np.empty(reps)
    for k in range(reps):
        s = rng.permutation(stubs)
        u, v = s[:len(a_idx)], s[len(a_idx):]
        keep = u != v
        null[k] = dist(u[keep], v[keep]).mean()
    pool = tested[ok[tested]]
    ra, rb = pool[rng.integers(len(pool), size=len(a_idx))], pool[rng.integers(len(pool), size=len(a_idx))]
    keep = ra != rb
    rnd = float(dist(ra[keep], rb[keep]).mean())
    return {"n_pairs": int(len(a_idx)), "mean_dist": obs, "null_config_mean": float(null.mean()),
            "null_config_sd": float(null.std()), "ratio_config": obs / null.mean(),
            "ses_config": (obs - null.mean()) / max(null.std(), 1e-12),
            "p_closer": float((1 + (null <= obs).sum()) / (reps + 1)),
            "p_farther": float((1 + (null >= obs).sum()) / (reps + 1)),
            "random_pairs_mean": rnd, "ratio_random": obs / rnd}


def compare(groups, n_sub, variants, X, samples, motus_k, dist, ok, a, tag, rng, write=True):
    """Run all groups x variants x repeats; write tables with suffix `tag`
    (write=False: only return the per-repeat statistics)."""
    Mall = X.shape[0]
    rep_rows, stab_rows, phy_rows, motu_rows = [], [], [], []
    for gname, gmask in groups.items():
        cols_all = np.where(gmask)[0]
        for var in variants:
            scol = VARIANTS[var]
            pos_sets, neg_sets, net_sets = [], [], []
            acc = {k: np.zeros(Mall) for k in ["n", "prev", "pos", "neg", "net", "pos_frac", "neg_frac"]}
            for rep in range(a.reps):
                t0 = time.time()
                cols = np.sort(rng.choice(cols_all, n_sub, replace=False))
                Xs = X[:, cols]
                tested = np.where(Xs.sum(1) >= a.min_det)[0]
                strata = samples[scol].values[cols] if scol else None
                st, prev, pd_, nd_, netd, P, Nn, (lo, hi, W) = run_one(Xs[tested], strata, a, rep)
                st.update({"group": gname, "variant": var, "rep": rep, "n_samples": n_sub,
                           "n_strata": len(np.unique(strata)) if strata is not None else 1})
                rep_rows.append(st)
                g = tested
                acc["n"][g] += 1; acc["prev"][g] += prev; acc["pos"][g] += pd_; acc["neg"][g] += nd_
                acc["net"][g] += netd; acc["pos_frac"][g] += pd_ / (len(g) - 1); acc["neg_frac"][g] += nd_ / (len(g) - 1)
                key = lambda A, B: np.unique(np.minimum(g[A], g[B]).astype(np.int64) * Mall + np.maximum(g[A], g[B]))
                pos_sets.append(key(P[:, 0], P[:, 1])); neg_sets.append(key(Nn[:, 0], Nn[:, 1]))
                net_sets.append((key(lo, hi), np.minimum(g[lo], g[hi]), np.maximum(g[lo], g[hi]), W))
                p.log(f"[{tag}] {gname} / {var} rep {rep}: {len(g)} mOTUs, +{st['sig_pos']} -{st['sig_neg']}, "
                      f"net {st['net_edges']} edges, {st['n_clusters_ge10']} clusters>=10, Q {st['modularity']:.3f} "
                      f"({time.time() - t0:.0f}s)")
            if not write:
                continue
            # stability
            for kind, sets in [("positive", pos_sets), ("negative |r|>=min_r", neg_sets),
                               ("network edge", [s[0] for s in net_sets])]:
                js, uk, cnt = jaccard_stats(sets, a.reps)
                stab_rows.append({"group": gname, "variant": var, "pairs": kind, **js})
                if kind == "network edge":
                    cons = uk[cnt >= a.reps / 2]
                elif kind.startswith("negative"):
                    neg_cons = uk[cnt >= a.reps / 2]
            # consensus network edges with mean r
            allk = np.concatenate([s[0] for s in net_sets]); allw = np.concatenate([s[3] for s in net_sets])
            order = np.argsort(allk); allk, allw = allk[order], allw[order]
            uk, start, cnt = np.unique(allk, return_index=True, return_counts=True)
            meanw = np.add.reduceat(allw, start) / cnt
            keep = cnt >= a.reps / 2
            ea, eb = (uk[keep] // Mall).astype(int), (uk[keep] % Mall).astype(int)
            if var == "controlled":
                pd.DataFrame({"a": motus_k.index[ea], "b": motus_k.index[eb], "r_mean": meanw[keep],
                              "frac_reps": cnt[keep] / a.reps}).to_csv(
                    f"{a.out}/env_network_edges_{gname.replace(' ', '_')}.tsv.gz", sep="\t", index=False,
                    float_format="%.4g")
            tested_any = np.where(acc["n"] > 0)[0]
            if dist is not None:
                na, nb_ = (neg_cons // Mall).astype(int), (neg_cons % Mall).astype(int)
                for kind, (A, B) in [("positive network edges", (ea, eb)), ("negative pairs (r <= -min_r)", (na, nb_))]:
                    res = phylo_test(A, B, tested_any, dist, ok, rng, a.phylo_reps)
                    if res:
                        phy_rows.append({"group": gname, "variant": var, "pairs": kind, **res})
            n = np.maximum(acc["n"], 1)
            for i in tested_any:
                motu_rows.append({"group": gname, "variant": var, "motu_id": motus_k.index[i],
                                  "n_reps_tested": int(acc["n"][i]), "prevalence": acc["prev"][i] / n[i],
                                  "pos_degree": acc["pos"][i] / n[i], "neg_degree": acc["neg"][i] / n[i],
                                  "pos_frac": acc["pos_frac"][i] / n[i], "neg_frac": acc["neg_frac"][i] / n[i],
                                  "net_degree": acc["net"][i] / n[i]})
    R = pd.DataFrame(rep_rows)
    if not write:
        return R
    R.to_csv(f"{a.out}/env_network_reps{tag}.tsv", sep="\t", index=False, float_format="%.5g")
    num = [c for c in R.columns if c not in ("group", "variant", "rep") and pd.api.types.is_numeric_dtype(R[c])]
    S = R.groupby(["group", "variant"], sort=False)[num].agg(["mean", "std"])
    S.columns = [f"{c}_{s}" if s == "std" else c for c, s in S.columns]
    S.reset_index().to_csv(f"{a.out}/env_network_summary{tag}.tsv", sep="\t", index=False, float_format="%.5g")
    pd.DataFrame(stab_rows).to_csv(f"{a.out}/env_network_stability{tag}.tsv", sep="\t", index=False, float_format="%.4g")
    if phy_rows:
        pd.DataFrame(phy_rows).to_csv(f"{a.out}/env_network_phylo{tag}.tsv", sep="\t", index=False, float_format="%.5g")
    pd.DataFrame(motu_rows).to_csv(f"{a.out}/env_network_motus{tag}.tsv", sep="\t", index=False, float_format="%.5g")
    return S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--tree", default=os.path.join(HERE, "data", "mOTUs_tree_bacteria.nwk"))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--frac", type=float, default=0.9, help="subsample size as a fraction of the smallest group")
    ap.add_argument("--min-det", type=int, default=20, help="min detections in the subsample to test a mOTU")
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--min-r", type=float, default=0.1)
    ap.add_argument("--knn", type=int, default=50)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--phylo-reps", type=int, default=199)
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-habitats", action="store_true", help="skip the sub-habitat comparison")
    ap.add_argument("--min-habitat", type=int, default=1000, help="min samples for a sub-habitat to get its own network")
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    with open(f"{a.out}/diagnostics.json") as fh:
        prm = json.load(fh)["params"]
    pa, samples, _ = p.load(a.data, prm["min_prev"], prm["rich_bins"], prm["min_study"])
    motus_k = pd.read_csv(f"{a.out}/motus.tsv", sep="\t", index_col=0).loc[pa.index]
    X = pa.values.astype(np.float32)
    del pa
    dist, ok = None, None
    if os.path.exists(a.tree):
        import phylogeny as ph
        tax = pd.read_csv(f"{a.data}/motus_subset_taxonomy.tsv", sep="\t").set_index("MOTU")
        dist, ok, _ = ph.make_distance(a.tree, motus_k, tax)
    env = samples.environment.values
    groups = {e: env == e for e in p.ENVS}
    sizes = {g: int(m.sum()) for g, m in groups.items()}
    n_sub = int(a.frac * min(sizes.values()))
    p.log(f"group sizes {sizes}; subsample n = {n_sub}, {a.reps} repeats")
    S = compare(groups, n_sub, list(VARIANTS), X, samples, motus_k, dist, ok, a, "", rng)
    info = {"n_sub": n_sub, "group_sizes": sizes, "reps": a.reps, "min_det": a.min_det, "params": vars(a)}

    if not a.no_habitats and "habitat" in samples:
        hc = samples.habitat.value_counts()
        g2 = {h: (samples.habitat == h).values for h in hc.index[hc >= a.min_habitat]}
        sizes2 = {g: int(m.sum()) for g, m in g2.items()}
        n2 = int(a.frac * min(sizes2.values()))
        p.log(f"sub-habitats: sizes {sizes2}; subsample n = {n2}")
        compare(g2, n2, list(VARIANTS), X, samples, motus_k, dist, ok, a, "_habitat", rng)
        info.update({"n_sub_habitat": n2, "group_sizes_habitat": sizes2,
                     "habitats_too_small": {h: int(c) for h, c in hc[hc < a.min_habitat].head(8).items()},
                     "habitat_environment": {h: samples.environment[samples.habitat == h].mode().iloc[0] for h in g2}})
    info["runtime_min"] = (time.time() - t0) / 60
    with open(f"{a.out}/env_network_info.json", "w") as fh:
        json.dump(info, fh, indent=1, default=str)
    pd.set_option("display.width", 250)
    show = ["nodes_tested", "sig_pos", "sig_neg", "pos_neg_ratio", "frac_pos", "frac_neg", "frac_pos_prev_matched",
            "frac_neg_prev_matched", "net_edges", "net_density", "n_clusters_ge10", "modularity", "spearman_posdeg_prev"]
    print(S[show].round(4).to_string())
    p.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
