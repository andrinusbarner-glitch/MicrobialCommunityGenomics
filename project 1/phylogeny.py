"""Are co-occurring mOTUs more closely related than chance?

    python phylogeny.py [--tree data/mOTUs_tree_bacteria.nwk] [--data data] [--out results]

Run after p9_pipeline.py. Relatedness between two mOTUs is
  - with the tree: patristic distance (sum of branch lengths between the tips);
  - without it:    taxonomic distance = number of ranks (domain ... species)
                   below the deepest rank they share (0 = same species, 7 = different domain).
The tree covers bacteria only, so archaea drop out when it is used.

Three tests, all against nulls that keep habitat fixed (otherwise "co-occurring
mOTUs are related" would only say "mOTUs from one habitat are related"):
  1. Network edges: mean relatedness of edge partners vs random pairs with the
     same combination of dominant environments.        -> phylo_edges.tsv
  2. Clusters: mean pairwise distance (MPD) of members vs random sets with the
     same environment composition; SES = (obs - null mean) / null sd.
     SES < 0: members closer relatives than expected (shared traits / habitat
     filtering); SES > 0: more distant (complementary or competing relatives
     kept apart).                                        -> phylo_clusters.tsv
  3. Pairs: random same-environment pairs tested under the main model; mean
     relatedness of significantly positive, significantly negative and
     non-significant pairs.                              -> phylo_pairs.tsv
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

import p9_pipeline as p

HERE = os.path.dirname(os.path.abspath(__file__))
RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]


# ------------------------------------------------------------------ tree
def read_newick(path):
    """Minimal iterative Newick parser -> parent, branch length, name per node
    (node 0 = root; parents always have smaller ids than their children)."""
    s = open(path, encoding="utf-8").read().strip().rstrip(";")
    parent, length, name = [-1], [0.0], [""]
    cur, i, n = 0, 0, len(s)
    stop = set(":,();")

    def new_child(par):
        parent.append(par); length.append(0.0); name.append("")
        return len(parent) - 1

    while i < n:
        c = s[i]
        if c == "(":
            cur = new_child(cur); i += 1
        elif c == ",":
            cur = new_child(parent[cur]); i += 1
        elif c == ")":
            cur = parent[cur]; i += 1
        elif c == ":":
            j = i + 1
            while j < n and s[j] not in stop:
                j += 1
            length[cur] = float(s[i + 1:j] or 0); i = j
        elif c in " \n\r\t":
            i += 1
        else:
            if c == "'":
                j = s.index("'", i + 1); name[cur] = s[i + 1:j]; i = j + 1
            else:
                j = i
                while j < n and s[j] not in stop:
                    j += 1
                name[cur] = s[i:j].strip(); i = j
    return np.array(parent), np.array(length), name


class Tree:
    """Patristic distances between tips via LCA (Euler tour + sparse table)."""

    def __init__(self, path):
        par, ln, self.name = read_newick(path)
        N = len(par)
        self.depth = np.zeros(N)
        for v in range(1, N):                      # parents come first
            self.depth[v] = self.depth[par[v]] + ln[v]
        kids = [[] for _ in range(N)]
        for v in range(1, N):
            kids[par[v]].append(v)
        self.is_tip = np.array([not k for k in kids])
        level = np.zeros(N, np.int32)
        euler, first = [], np.zeros(N, np.int64)
        stack = [(0, 0)]
        while stack:                                # iterative DFS, Euler tour
            v, k = stack.pop()
            if k == 0:
                first[v] = len(euler)
            euler.append(v)
            if k < len(kids[v]):
                stack.append((v, k + 1))
                c = kids[v][k]; level[c] = level[v] + 1
                stack.append((c, 0))
        self.euler = np.array(euler, np.int64)
        self.first = first
        lv = level[self.euler]
        self.sp = [np.arange(len(lv))]
        k = 1
        while (1 << k) <= len(lv):
            prev = self.sp[-1]; h = 1 << (k - 1)
            a, b = prev[:-h], prev[h:]
            self.sp.append(np.where(lv[a] <= lv[b], a, b)); k += 1
        self.lv = lv

    def dist(self, u, v):
        l = np.minimum(self.first[u], self.first[v]); r = np.maximum(self.first[u], self.first[v]) + 1
        k = np.floor(np.log2(np.maximum(r - l, 1))).astype(int)
        out = np.empty(len(u), np.int64)
        for kk in np.unique(k):
            m = k == kk
            a = self.sp[kk][l[m]]; b = self.sp[kk][r[m] - (1 << kk)]
            out[m] = np.where(self.lv[a] <= self.lv[b], a, b)
        lca = self.euler[out]
        return self.depth[u] + self.depth[v] - 2 * self.depth[lca]


# ------------------------------------------------------------------ relatedness
def make_distance(tree_path, mk, tax):
    """Returns (dist(i, j) on mOTU row indices, usable mask, description)."""
    if tree_path and os.path.exists(tree_path):
        T = Tree(tree_path)
        tips = {T.name[v]: v for v in np.where(T.is_tip)[0]}
        node = np.full(len(mk), -1)
        for k, (mid, gen) in enumerate(zip(mk.index, tax["GENOME"].reindex(mk.index))):
            for key in (mid, mid.replace("mOTUv4.0_", ""), gen):
                if isinstance(key, str) and key in tips:
                    node[k] = tips[key]; break
        ok = node >= 0
        p.log(f"tree: {T.is_tip.sum()} tips; matched {ok.sum()} of {len(mk)} mOTUs")
        if ok.sum() == 0:
            raise SystemExit("no tree tip matches an mOTU ID or genome name - check the tip labels")
        return (lambda i, j: T.dist(node[i], node[j])), ok, "patristic distance (tree)"
    codes = np.stack([pd.factorize(mk[r].fillna(f"NA_{r}_" + mk.index.to_series()))[0] for r in RANKS], 1)

    def d(i, j):
        same = np.cumprod(codes[i] == codes[j], axis=1)        # shared from the top down
        return (len(RANKS) - same.sum(1)).astype(float)
    p.log("no tree file found: using taxonomic distance (ranks below the deepest shared rank)")
    return d, np.ones(len(mk), bool), "taxonomic distance (0 = same species, 7 = different domain)"


def rand_pairs(pool_a, pool_b, n, rng):
    a = rng.choice(pool_a, n); b = rng.choice(pool_b, n)
    bad = a == b
    while bad.any():
        b[bad] = rng.choice(pool_b, bad.sum()); bad = a == b
    return a, b


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--tree", default=os.path.join(HERE, "data", "mOTUs_tree_bacteria.nwk"))
    ap.add_argument("--reps", type=int, default=199)
    ap.add_argument("--pairs", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    with open(f"{a.out}/diagnostics.json") as fh:
        prm = json.load(fh)["params"]
    summ = pd.read_csv(f"{a.out}/run_summary.tsv", sep="\t", index_col="model")
    pa, samples, _ = p.load(a.data, prm["min_prev"], prm["rich_bins"], prm["min_study"])
    mk = pd.read_csv(f"{a.out}/motus.tsv", sep="\t", index_col=0).loc[pa.index]
    tax = pd.read_csv(f"{a.data}/motus_subset_taxonomy.tsv", sep="\t").set_index("MOTU")
    dist, ok, what = make_distance(a.tree, mk, tax)
    M = len(mk)
    pos = pd.Series(np.arange(M), index=mk.index)
    env = mk["dom_env"].values
    pools = {e: np.where((env == e) & ok)[0] for e in p.ENVS}

    # ---- 1. network edges vs habitat-matched random pairs
    rows = []
    for m in p.MODELS:
        E = pd.read_csv(f"{a.out}/network_edges_{m}.tsv.gz", sep="\t")
        i, j = pos[E.a].values, pos[E.b].values
        keep = ok[i] & ok[j]
        i, j = i[keep], j[keep]
        d_obs = dist(i, j)
        for scope in ["all"] + p.ENVS:
            sel = np.ones(len(i), bool) if scope == "all" else (env[i] == scope) & (env[j] == scope)
            if sel.sum() < 100:
                continue
            ii, jj = i[sel], j[sel]
            nulls = []
            for _ in range(20):
                ni, nj = np.empty_like(ii), np.empty_like(jj)
                for e1 in p.ENVS:
                    for e2 in p.ENVS:
                        mm = (env[ii] == e1) & (env[jj] == e2)
                        if mm.any():
                            ni[mm], nj[mm] = rand_pairs(pools[e1], pools[e2], mm.sum(), rng)
                nulls.append(dist(ni, nj).mean())
            nulls = np.array(nulls)
            obs = d_obs[sel].mean()
            rows.append({"model": m, "scope": scope, "n_edges": int(sel.sum()), "mean_dist_edges": obs,
                         "mean_dist_null": nulls.mean(), "ratio": obs / nulls.mean(),
                         "z": (obs - nulls.mean()) / max(nulls.std(ddof=1), 1e-12)})
        r_all = next(r for r in rows if r["model"] == m and r["scope"] == "all")
        p.log(f"[edges] {m}: edge partners / matched random pairs = {r_all['ratio']:.3f} (z {r_all['z']:.1f})")
    edges = pd.DataFrame(rows)
    edges.to_csv(f"{a.out}/phylo_edges.tsv", sep="\t", index=False)

    # ---- 2. clusters (main model): MPD vs composition-matched random sets
    lab = mk[f"cluster_{p.MAIN}"].fillna(-1).astype(int).values
    cl = pd.read_csv(f"{a.out}/clusters_{p.MAIN}.tsv", sep="\t").set_index("cluster")

    def mpd(members, n=5000):
        x, y = rand_pairs(members, members, n, rng)
        return dist(x, y).mean()

    crow = []
    for c in sorted(set(lab[lab >= 0])):
        mem = np.where((lab == c) & ok)[0]
        if len(mem) < 10:
            continue
        comp = pd.Series(env[mem]).value_counts()
        obs = mpd(mem)
        null = []
        for _ in range(a.reps):
            s = np.concatenate([rng.choice(pools[e], k, replace=False) for e, k in comp.items()])
            null.append(mpd(s))
        null = np.array(null)
        ses = (obs - null.mean()) / max(null.std(ddof=1), 1e-12)
        crow.append({"cluster": c, "n_motus": len(mem), "top_family": cl.loc[c, "top_family"],
                     "dominant_env": cl.loc[c, "dominant_env"], "mpd": obs, "mpd_null": null.mean(),
                     "ses": ses, "p_clustered": (np.sum(null <= obs) + 1) / (a.reps + 1),
                     "p_overdispersed": (np.sum(null >= obs) + 1) / (a.reps + 1)})
    clus = pd.DataFrame(crow)
    clus.to_csv(f"{a.out}/phylo_clusters.tsv", sep="\t", index=False)
    p.log(f"[clusters] {len(clus)} clusters; SES < -2 in {(clus.ses < -2).sum()}, > 2 in {(clus.ses > 2).sum()}")

    # ---- 3. random same-environment pairs: relatedness by association class
    X = pa.values.astype(np.int8)
    model = p.NullModel(X.astype(np.float32), samples[p.STRATA_COL[p.MAIN]].values, p.MAIN)
    thr = summ.loc[p.MAIN, "z_threshold"]
    per = a.pairs // 4
    I, J = [], []
    for e in p.ENVS:
        x, y = rand_pairs(pools[e], pools[e], per, rng); I.append(x); J.append(y)
    I, J = np.concatenate(I), np.concatenate(J)
    O, Ex, V = np.empty(len(I)), np.empty(len(I)), np.empty(len(I))
    for s in range(0, len(I), 2000):
        a_, b_ = I[s:s + 2000], J[s:s + 2000]
        O[s:s + 2000] = (X[a_] & X[b_]).sum(1)
        Ex[s:s + 2000] = (model.B[a_] * model.B[b_]).sum(1)
        V[s:s + 2000] = (model.A[a_] * model.A[b_]).sum(1)
    z = (O - Ex) / np.sqrt(np.maximum(V, 1e-12)); z[V <= 1e-12] = 0
    r = (O - Ex) / np.sqrt(np.maximum(model.s[I] * model.s[J], 1e-12))
    d = dist(I, J)
    cls = np.where((z >= thr) & (r >= prm["min_r"]), "positive (edge-grade)",
                   np.where(z >= thr, "positive (weak)", np.where(z <= -thr, "negative", "not significant")))
    prow = []
    for scope in ["all"] + p.ENVS:
        sel = np.ones(len(I), bool) if scope == "all" else env[I] == scope
        base = d[sel].mean()
        rho = stats.spearmanr(r[sel], d[sel]).correlation
        for k in ["positive (edge-grade)", "positive (weak)", "not significant", "negative"]:
            mm = sel & (cls == k)
            if mm.sum() < 30:
                continue
            boot = [d[rng.choice(np.where(mm)[0], mm.sum())].mean() for _ in range(200)]
            prow.append({"scope": scope, "class": k, "n_pairs": int(mm.sum()), "mean_dist": d[mm].mean(),
                         "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                         "ratio_to_all_pairs": d[mm].mean() / base, "spearman_r_vs_dist": rho})
    pairs = pd.DataFrame(prow)
    pairs.to_csv(f"{a.out}/phylo_pairs.tsv", sep="\t", index=False)
    with open(f"{a.out}/phylo_info.json", "w") as fh:
        json.dump({"distance": what, "n_matched": int(ok.sum()), "n_motus": int(M),
                   "reps": a.reps, "pairs": int(len(I))}, fh, indent=1)

    pd.set_option("display.width", 200)
    print(f"\nrelatedness = {what}")
    print(edges.round(3).to_string(index=False))
    print(clus.round(3).to_string(index=False))
    print(pairs.round(3).to_string(index=False))
    p.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
