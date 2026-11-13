"""
P9 co-occurrence pipeline (streamlined), BC2026 mOTUs subset.

What it does
  1. Load presence/absence (PA), taxonomy and environment once. Drop mOTUs that
     are detected in fewer than --min-prev samples *of this subset*. (The README
     filter of >=500 samples refers to the full atlas; inside the subset some
     mOTUs occur in only 1-5 samples, and those produce huge, meaningless z.)
  2. Pairwise co-occurrence test under a hypergeometric null, pooled over
     strata (Cochran-Mantel-Haenszel style), for five stratifications, each
     adding one correction:
        global          - one stratum (baseline: no control at all)
        env             - environment (removes "both like the same habitat")
        env_rich        - + within-environment sample-richness quartile (removes
                          "both detected in the same deep / rich samples")
        env_study_rich  - + study (sample-ID prefix; studies with < --min-study
                          samples pooled per environment). Removes "both found
                          in the same study / batch / location".
        hab_study_rich  - sub-habitat instead of environment (mOTUs-db NCBI term:
                          human gut / oral / skin, pig / cattle / chicken gut,
                          wastewater, lake, estuary, ...; habitats with < --min-study
                          samples pooled per environment) x study x richness
                          quartile within the sub-habitat. MAIN RESULT when
                          data/motusdb/ is present; otherwise env_study_rich is.
     Per cluster, the study concentration of its detections is reported next
     to the same number for its whole environment.
  3. Significance: z-score -> two-sided p -> Benjamini-Hochberg FDR.
     Edge weight: an effect size, not z. We use the within-stratum correlation
     r of the two 0/1 vectors (phi coefficient after centring each mOTU within
     each stratum). z grows with sample size; r does not.
  4. Positive network only: significant, r >= --min-r, each mOTU keeps its
     --knn strongest partners (edge kept if either end chooses it), Louvain.
  5. Write per-mOTU and per-cluster tables (with correct taxonomy), the network
     edges, and the global-vs-stratified comparison statistics.

Why it fits in memory
  O = X X^T (observed co-occurrence) is the same for every stratification;
  only E (expected) and V (variance) change, and both are low-rank sums over
  strata: E = B B^T and V = A A^T with one column per stratum. So O is computed
  once, and every model is evaluated block by block.

Usage
  python p9_pipeline.py                     # data in ./data, results in ./results
  python p9_pipeline.py --data DIR --out OUT --min-prev 50 --fdr 0.01 --min-r 0.1 --knn 50
  In a notebook:  import p9_pipeline as p; p.main(["--data", D, "--out", OUT])
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

ENVS = ["Human-associated", "Animal-associated", "Non-saline", "Saline"]
RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
HAVE_HABITAT = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "motusdb",
                                            "Supplementary_Table_4.tsv.gz"))
MODELS = ["global", "env", "env_rich", "env_study_rich"] + (["hab_study_rich"] if HAVE_HABITAT else [])
STRATA_COL = {"env": "environment", "env_rich": "stratum", "env_study_rich": "stratum_study_rich",
              "hab_study_rich": "stratum_hab_study_rich"}
MAIN = MODELS[-1]
# strata for the abundance correlations (no richness: CLR cancels library size)
ABUND_STRATA_COL = "stratum_hab_study" if HAVE_HABITAT else "stratum_study"
HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================ loading
def load(data_dir, min_prev=50, n_rich_bins=4, min_study=50):
    """Shared loader (also used by the other scripts).

    Returns
      pa       DataFrame int8, mOTU x sample, mOTUs with >= min_prev detections
      samples  DataFrame per sample: environment, richness (all mOTUs),
               richness bin within environment, stratum label env_rich
      motus    DataFrame per mOTU (all mOTUs, before the filter): prevalence,
               kept flag, habitat profile and taxonomy
    """
    d = data_dir.rstrip("/\\") + "/"
    pa = pd.read_csv(d + "motus_subset_pa.tsv.gz", sep="\t", index_col=0).astype(np.int8)
    tax = pd.read_csv(d + "motus_subset_taxonomy.tsv", sep="\t").set_index("MOTU")
    env = (pd.read_csv(d + "motus_subset_environment.tsv", sep="\t")
             .set_index("sample_id")["environment"].reindex(pa.columns))
    assert env.notna().all(), "samples without environment label"
    assert set(env.unique()) == set(ENVS), f"unexpected environments: {set(env.unique())}"

    # ---- samples: richness is counted over ALL mOTUs (it is a property of the sample)
    samples = pd.DataFrame({"environment": env.values,
                            "richness": pa.values.sum(0).astype(int)}, index=pa.columns)
    samples["rich_bin"] = (samples.groupby("environment")["richness"]
                           .transform(lambda r: pd.qcut(r.rank(method="first"), n_rich_bins,
                                                        labels=False)).astype(int))
    samples["stratum"] = samples["environment"] + "|r" + samples["rich_bin"].astype(str)
    # study = prefix of the sample ID (e.g. ACIN21-1_SAMN05421561_METAG -> ACIN21-1).
    # Studies with < min_study samples in an environment are pooled as "<env>|other".
    samples["study"] = samples.index.str.split("_").str[0]
    n_in_env = samples.groupby(["environment", "study"])["study"].transform("size")
    samples["study_group"] = np.where(n_in_env >= min_study, samples["study"], "other")
    samples["stratum_study"] = samples["environment"] + "|" + samples["study_group"]
    samples["stratum_study_rich"] = samples["stratum_study"] + "|r" + samples["rich_bin"].astype(str)
    samples = samples.join(sample_habitats(d, samples.index))
    if "habitat" in samples:
        # sub-habitat groups: habitats with < min_study samples pooled per environment
        n_hab = samples.groupby(["environment", "habitat"])["habitat"].transform("size")
        samples["habitat_group"] = np.where(n_hab >= min_study, samples["habitat"].fillna("other"), "other")
        hab = samples["environment"] + "|" + samples["habitat_group"]
        samples["hab_rich_bin"] = (samples.groupby(hab)["richness"]
                                   .transform(lambda r: pd.qcut(r.rank(method="first"), n_rich_bins,
                                                                labels=False)).astype(int))
        samples["stratum_hab_study"] = hab + "|" + samples["study_group"]
        samples["stratum_hab_study_rich"] = samples["stratum_hab_study"] + "|r" + samples["hab_rich_bin"].astype(str)

    # ---- mOTUs: habitat profile on all mOTUs, then filter
    X = pa.values
    motus = pd.DataFrame(index=pa.index)
    motus["prevalence"] = X.sum(1).astype(int)
    motus["kept"] = motus["prevalence"] >= min_prev
    prev = pd.DataFrame({e: X[:, env.values == e].mean(1) for e in ENVS}, index=pa.index)
    tot = prev.sum(1).replace(0, np.nan)
    aff = prev.div(tot, axis=0)
    for e in ENVS:
        motus["prev_" + e] = prev[e]
    for e in ENVS:
        motus["aff_" + e] = aff[e]
    motus["dom_env"] = aff.idxmax(axis=1)
    motus["purity"] = aff.max(axis=1)
    p = aff.values.clip(1e-12)
    motus["breadth"] = -(p * np.log(p)).sum(1) / np.log(len(ENVS))
    mu = X.mean(1)
    sst = X.sum(1) - len(env) * mu ** 2                 # sum (x - mu)^2 for 0/1 data
    ssb = sum((env.values == e).sum() * (prev[e].values - mu) ** 2 for e in ENVS)
    motus["eta2_env"] = ssb / np.maximum(sst, 1e-9)
    for r in RANKS:
        motus[r] = tax[r].reindex(pa.index)       # plain column per rank, no 'p__' prefix
    pa = pa.loc[motus["kept"]]
    return pa, samples, motus


def sample_habitats(d, index):
    """Finer per-sample environment from mOTUs-db (Zenodo 10.5281/zenodo.13325008,
    Supplementary Table 4, CC-BY-4.0), if data/motusdb/ holds it.
    ncbi_environment: the NCBI metagenome term(s), e.g. 'human oral metagenome';
    habitat: the first term without ' metagenome' (e.g. 'human oral', 'wastewater')."""
    path = d + "motusdb/Supplementary_Table_4.tsv.gz"
    if not os.path.exists(path):
        return pd.DataFrame(index=index)
    t = pd.read_csv(path, sep="\t").set_index("#SAMPLE").reindex(index)
    return pd.DataFrame({"biosample": t["BIOSAMPLE"],
                         "ncbi_environment": t["ENVIRONMENT"],
                         "habitat": t["ENVIRONMENT"].str.split(",").str[0].str.replace(" metagenome", "").str.strip()},
                        index=index)


def read_depth(data_dir, columns, chunksize=1000):
    """Total read count per sample from the count matrix, read in chunks
    (diagnostic only: is richness just sequencing depth?)."""
    tot = None
    for ch in pd.read_csv(data_dir.rstrip("/\\") + "/motus_subset_matrix.tsv.gz", sep="\t",
                          index_col=0, chunksize=chunksize):
        s = ch.sum(axis=0)
        tot = s if tot is None else tot + s
    return tot.reindex(columns)


# ============================================================ statistics
class NullModel:
    """Hypergeometric null pooled over strata, in low-rank form.

    For stratum s with N_s samples and mOTU counts n_is:
      E_ij = sum_s n_is n_js / N_s                         = (B B^T)_ij
      V_ij = sum_s n_is(N_s-n_is) n_js(N_s-n_js) / (N_s^2 (N_s-1)) = (A A^T)_ij
      z    = (O - E) / sqrt(V)                             (CMH-type z)
      r    = (O - E) / sqrt(s_i s_j),  s_i = sum_s n_is(N_s-n_is)/N_s
             = correlation of the two 0/1 vectors after centring each within
               every stratum (a pooled phi coefficient; bounded, N-free)
    """

    def __init__(self, X, labels, name):
        self.name = name
        labels = np.asarray(labels)
        self.strata = pd.unique(labels)
        Bc, Ac, sc = [], [], 0.0
        for st in self.strata:
            cols = labels == st
            Ns = float(cols.sum())
            n = X[:, cols].sum(1, dtype=np.float64)
            a = n * (Ns - n)
            Bc.append(n / np.sqrt(Ns))
            Ac.append(a / np.sqrt(Ns ** 2 * max(Ns - 1, 1)))   # Ns = 1: a = 0, no variance
            sc = sc + a / Ns
        self.B = np.stack(Bc, 1)
        self.A = np.stack(Ac, 1)
        self.s = np.asarray(sc)

    def block(self, O_b, rows):
        """z and r for rows `rows` (slice) against all mOTUs, given O[rows]."""
        E = self.B[rows] @ self.B.T
        V = self.A[rows] @ self.A.T
        D = O_b - E
        z = D / np.sqrt(np.maximum(V, 1e-12))
        r = D / np.sqrt(np.maximum(np.outer(self.s[rows], self.s), 1e-12))
        z[V <= 1e-12] = 0.0          # mOTU constant in every stratum: untestable
        return z.astype(np.float32), r.astype(np.float32)


def cooccurrence(X, block):
    """O = X X^T for 0/1 float32 X, filled block by block (exact up to 2^24)."""
    M = X.shape[0]
    O = np.empty((M, M), np.float32)
    for i in range(0, M, block):
        O[i:i + block] = X[i:i + block] @ X.T
    return O


def bh_threshold(model, O, block, fdr):
    """Smallest |z| that is significant under Benjamini-Hochberg (two-sided),
    over all unordered pairs."""
    M = O.shape[0]
    m = M * (M - 1) // 2
    buf = np.empty(m, np.float32)
    pos = 0
    for i in range(0, M, block):
        rows = slice(i, min(i + block, M))
        z, _ = model.block(O[rows], rows)
        for k, row in enumerate(range(rows.start, rows.stop)):
            seg = np.abs(z[k, row + 1:])
            buf[pos:pos + len(seg)] = seg
            pos += len(seg)
    buf.sort()
    buf = buf[::-1]                                    # descending |z| = ascending p
    thr, step = np.inf, 5_000_000
    for s in range(0, m, step):                        # last k with p_(k) <= fdr k / m
        zz = buf[s:s + step].astype(np.float64)
        p = 2 * stats.norm.sf(zz)
        ok = np.nonzero(p <= fdr * np.arange(s + 1, s + len(zz) + 1) / m)[0]
        if len(ok):
            thr = float(zz[ok[-1]])
    del buf
    return thr


# ============================================================ network
def knn_edges(cand_w, rows, k):
    """Per row, the k largest positive weights. Returns (i, j, w) arrays."""
    kk = min(k, cand_w.shape[1] - 1)
    idx = np.argpartition(-cand_w, kk, axis=1)[:, :kk]
    w = np.take_along_axis(cand_w, idx, 1)
    ok = w > 0
    I = np.repeat(np.arange(rows.start, rows.stop), kk).reshape(-1, kk)[ok]
    return I, idx[ok], w[ok]


def louvain(G, resolution, seed):
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    return sorted(comms, key=len, reverse=True)


def labels_from(comms, M):
    lab = np.full(M, -1)
    for c, s in enumerate(comms):
        lab[list(s)] = c
    return lab


def study_profile(rows, motu_study, studies):
    """Pooled detections of a set of mOTUs per study -> top study, its share,
    and the effective number of studies (inverse Simpson)."""
    v = motu_study[rows].sum(0)
    q = v / max(v.sum(), 1)
    k = int(q.argmax())
    return studies[k], float(q[k]), float(1 / max((q ** 2).sum(), 1e-12))


def cluster_table(lab, motus_k, degree, motu_study=None, studies=None):
    """One row per cluster. With motu_study: study concentration of the cluster's
    detections, next to the same numbers for all mOTUs of its dominant environment
    (the baseline: environments themselves come from a limited set of studies)."""
    df = motus_k.assign(cluster=lab, degree=degree, _row=np.arange(len(lab)))[lab >= 0]
    env_base = {}
    if motu_study is not None:
        for e in ENVS:
            env_base[e] = study_profile(np.where(motus_k["dom_env"].values == e)[0], motu_study, studies)
    rows = []
    for c, g in df.groupby("cluster"):
        a = g[[f"aff_{e}" for e in ENVS]].mean()
        row = {"cluster": c, "n_motus": len(g)}
        row.update({e: a[f"aff_{e}"] for e in ENVS})
        row["dominant_env"] = a.idxmax().replace("aff_", "")
        row["env_purity"] = a.max()
        row["frac_dom_env"] = (g["dom_env"] == row["dominant_env"]).mean()
        for rk in ["phylum", "family", "genus"]:
            t = g[rk].value_counts(normalize=True)
            row[f"top_{rk}"] = t.index[0] if len(t) else None
            row[f"top_{rk}_frac"] = t.iloc[0] if len(t) else np.nan
        row["median_prevalence"] = g["prevalence"].median()
        row["mean_breadth"] = g["breadth"].mean()
        row["mean_degree"] = g["degree"].mean()
        if motu_study is not None:
            s, sh, ne = study_profile(g["_row"].values, motu_study, studies)
            bs, bsh, bne = env_base[row["dominant_env"]]
            row.update({"top_study": s, "top_study_share": sh, "eff_n_studies": ne,
                        "env_top_study": bs, "env_top_study_share": bsh, "env_eff_n_studies": bne,
                        "study_concentration": sh / bsh})
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--min-prev", type=int, default=50, help="min detections in this subset")
    ap.add_argument("--rich-bins", type=int, default=4, help="richness bins within each environment")
    ap.add_argument("--min-study", type=int, default=50,
                    help="studies with fewer samples in an environment are pooled per environment")
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--min-r", type=float, default=0.1, help="min effect size (pooled phi) for a network edge")
    ap.add_argument("--knn", type=int, default=50)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=5, help="Louvain seeds for the stability check (main model)")
    ap.add_argument("--block", type=int, default=500)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    # ---- 1. data ---------------------------------------------------------
    pa, samples, motus = load(a.data, a.min_prev, a.rich_bins, a.min_study)
    log(f"{len(motus)} mOTUs x {len(samples)} samples; kept {len(pa)} mOTUs with >= {a.min_prev} "
        f"detections (dropped {int((~motus.kept).sum())})")
    samples["reads"] = read_depth(a.data, samples.index).values
    motus_k = motus.loc[pa.index]
    M = len(pa)
    de = motus_k["dom_env"].map({e: k for k, e in enumerate(ENVS)}).values

    X = pa.values.astype(np.float32)
    log("computing observed co-occurrence O = X X^T")
    O = cooccurrence(X, a.block)
    models = {"global": NullModel(X, np.zeros(X.shape[1]), "global")}
    for n in MODELS[1:]:
        models[n] = NullModel(X, samples[STRATA_COL[n]].values, n)
    # detections of each mOTU per study (for the study-concentration diagnostic)
    studies = pd.Index(sorted(samples["study"].unique()))
    onehot = np.zeros((X.shape[1], len(studies)), np.float32)
    onehot[np.arange(X.shape[1]), studies.get_indexer(samples["study"])] = 1
    motu_study = X @ onehot
    del X

    # ---- 2. significance thresholds -------------------------------------
    thr = {}
    for name, mdl in models.items():
        thr[name] = bh_threshold(mdl, O, a.block, a.fdr)
        log(f"[{name}] {len(mdl.strata)} strata; BH {a.fdr} -> significant if |z| >= {thr[name]:.2f}")

    # ---- 3. one pass over all pairs: counts, kNN candidates, comparisons --
    names = list(models)
    cnt = {n: {"sig_pos": 0, "sig_neg": 0, "sig_pos_min_r": 0, "pos_same_env": 0, "neg_same_env": 0}
           for n in names}
    deg_sig = {n: np.zeros(M, np.int64) for n in names}
    rhist_edges = np.linspace(-1, 1, 201)
    rhist = {n: np.zeros(200, np.int64) for n in names}
    dens = {n: {s: np.zeros((4, 4), np.int64) for s in ["pos", "neg"]} for n in names}
    pairs_fate = [(x, y) for x, y in zip(MODELS[:-1], MODELS[1:])] + [("global", MAIN)]
    fate = {f"{x}->{y}": {f"{s}_{k}": 0 for s in ["pos", "neg"]
                          for k in ["kept", "lost", "flipped", "new", "lost_same_env"]}
            for x, y in pairs_fate}
    knn = {n: ([], [], [], []) for n in names}

    for i in range(0, M, a.block):
        rows = slice(i, min(i + a.block, M))
        nb = rows.stop - rows.start
        O_b = O[rows]
        upper = np.arange(M)[None, :] > np.arange(rows.start, rows.stop)[:, None]
        same = de[rows][:, None] == de[None, :]
        pair_code = (np.minimum(de[rows][:, None], de[None, :]) * 4
                     + np.maximum(de[rows][:, None], de[None, :]))
        sgn = {}
        for n in names:
            z, r = models[n].block(O_b, rows)
            sig = np.abs(z) >= thr[n]
            sig[np.arange(nb), np.arange(rows.start, rows.stop)] = False
            pos, neg = sig & (z > 0), sig & (z < 0)
            pu, nu = pos & upper, neg & upper
            c = cnt[n]
            c["sig_pos"] += int(pu.sum()); c["sig_neg"] += int(nu.sum())
            c["sig_pos_min_r"] += int((pu & (r >= a.min_r)).sum())
            c["pos_same_env"] += int((pu & same).sum()); c["neg_same_env"] += int((nu & same).sum())
            deg_sig[n][rows] += pos.sum(1)
            rhist[n] += np.histogram(r[pu | nu], bins=rhist_edges)[0]
            dens[n]["pos"] += np.bincount(pair_code[pu], minlength=16).reshape(4, 4)
            dens[n]["neg"] += np.bincount(pair_code[nu], minlength=16).reshape(4, 4)
            cand = np.where(pos & (r >= a.min_r), r, 0.0)
            I, J, W = knn_edges(cand, rows, a.knn)
            knn[n][0].append(I); knn[n][1].append(J); knn[n][2].append(W)
            knn[n][3].append(z[I - rows.start, J])
            sgn[n] = np.where(sig, np.sign(z), 0).astype(np.int8)
        for x, y in pairs_fate:
            f = fate[f"{x}->{y}"]
            gx, gy = sgn[x][upper], sgn[y][upper]
            sm = same[upper]
            for s, v in [("pos", 1), ("neg", -1)]:
                f[f"{s}_kept"] += int(((gx == v) & (gy == v)).sum())
                f[f"{s}_flipped"] += int(((gx == v) & (gy == -v)).sum())
                lost = (gx == v) & (gy == 0)
                f[f"{s}_lost"] += int(lost.sum())
                f[f"{s}_lost_same_env"] += int((lost & sm).sum())
                f[f"{s}_new"] += int(((gx == 0) & (gy == v)).sum())
        if (i // a.block) % 4 == 0:
            log(f"pairs pass: rows {rows.stop}/{M}")
    del O

    # ---- 4. networks + clusters -----------------------------------------
    summary, lab_all = [], {}
    n_env = motus_k["dom_env"].value_counts().reindex(ENVS).fillna(0).values.astype(float)
    possible = np.outer(n_env, n_env); np.fill_diagonal(possible, n_env * (n_env - 1) / 2)
    for n in names:
        I, J, W, Z = (np.concatenate(v) for v in knn[n])
        lo, hi = np.minimum(I, J), np.maximum(I, J)
        key = lo.astype(np.int64) * M + hi
        _, u = np.unique(key, return_index=True)            # union of both ends' choices
        lo, hi, W, Z = lo[u], hi[u], W[u], Z[u]
        G = nx.Graph()
        G.add_weighted_edges_from(zip(lo.tolist(), hi.tolist(), W.astype(float).tolist()))
        comms = louvain(G, a.resolution, 0)
        lab = labels_from(comms, M)
        lab_all[n] = lab
        deg_net = np.bincount(np.concatenate([lo, hi]), minlength=M)
        env_parts = [p for p in (set(np.where(de == k)[0]) & set(G.nodes) for k in range(4)) if p]
        row = {"model": n, "n_strata": len(models[n].strata), "z_threshold": thr[n],
               "sig_pos_pairs": cnt[n]["sig_pos"], "sig_neg_pairs": cnt[n]["sig_neg"],
               "pos_neg_ratio": cnt[n]["sig_pos"] / max(cnt[n]["sig_neg"], 1),
               "frac_pairs_sig": (cnt[n]["sig_pos"] + cnt[n]["sig_neg"]) / (M * (M - 1) / 2),
               "sig_pos_pairs_r_ge_min": cnt[n]["sig_pos_min_r"],
               "frac_pos_same_env": cnt[n]["pos_same_env"] / max(cnt[n]["sig_pos"], 1),
               "frac_neg_same_env": cnt[n]["neg_same_env"] / max(cnt[n]["sig_neg"], 1),
               "net_nodes": G.number_of_nodes(), "net_edges": G.number_of_edges(),
               "n_clusters": len(comms), "n_clusters_ge10": sum(len(c) >= 10 for c in comms),
               "Q_louvain": nx.community.modularity(G, comms, weight="weight"),
               "Q_env": nx.community.modularity(G, env_parts, weight="weight"),
               "AMI_cluster_env": adjusted_mutual_info_score(de[lab >= 0], lab[lab >= 0]),
               "spearman_sigdegree_prevalence": stats.spearmanr(deg_sig[n], motus_k.prevalence).correlation,
               "spearman_netdegree_prevalence": stats.spearmanr(deg_net, motus_k.prevalence).correlation}
        if n == MAIN and a.seeds > 1:
            aris = []
            for sd in range(1, a.seeds):
                l2 = labels_from(louvain(G, a.resolution, sd), M)
                m_ = (lab >= 0) & (l2 >= 0)
                aris.append(adjusted_rand_score(lab[m_], l2[m_]))
            row["seed_ARI_mean"], row["seed_ARI_min"] = float(np.mean(aris)), float(np.min(aris))
        summary.append(row)
        log(f"[{n}] network {row['net_nodes']} nodes, {row['net_edges']} edges, {row['n_clusters']} clusters; "
            f"Q {row['Q_louvain']:.3f} (env-only {row['Q_env']:.3f}); AMI(env) {row['AMI_cluster_env']:.2f}")

        pd.DataFrame({"a": pa.index[lo], "b": pa.index[hi], "r": W, "z": Z}).to_csv(
            f"{a.out}/network_edges_{n}.tsv.gz", sep="\t", index=False, float_format="%.4g")
        ct = cluster_table(lab, motus_k, deg_net, motu_study, studies)
        ct.to_csv(f"{a.out}/clusters_{n}.tsv", sep="\t", index=False)
        if len(ct):
            w = ct["n_motus"] / ct["n_motus"].sum()
            row["study_top_share_wmean"] = float((ct["top_study_share"] * w).sum())
            row["n_clusters_study_dominated"] = int((ct["top_study_share"] >= 0.5).sum())
        motus.loc[pa.index, f"cluster_{n}"] = lab
        motus.loc[pa.index, f"sigdegree_{n}"] = deg_sig[n]
        motus.loc[pa.index, f"netdegree_{n}"] = deg_net
        for s in ["pos", "neg"]:
            C = dens[n][s]
            C = C + C.T - np.diag(np.diag(C))
            pd.DataFrame(C / np.maximum(possible, 1), index=ENVS, columns=ENVS).to_csv(
                f"{a.out}/envpair_density_{n}_{s}.tsv", sep="\t")

    # ---- 5. tables -------------------------------------------------------
    pd.DataFrame(summary).to_csv(f"{a.out}/run_summary.tsv", sep="\t", index=False)
    fr = pd.DataFrame(fate).T
    fr.index.name = "comparison"
    fr.to_csv(f"{a.out}/edge_fate.tsv", sep="\t")
    pd.DataFrame({"bin_lo": rhist_edges[:-1], "bin_hi": rhist_edges[1:], **rhist}).to_csv(
        f"{a.out}/effect_size_hist.tsv", sep="\t", index=False)
    motus.to_csv(f"{a.out}/motus.tsv", sep="\t")
    samples.to_csv(f"{a.out}/samples.tsv", sep="\t")
    diag = {"richness_vs_reads_spearman": {
        e: float(stats.spearmanr(g.richness, g.reads).correlation)
        for e, g in samples.groupby("environment")},
        "richness_by_env_median": samples.groupby("environment").richness.median().to_dict(),
        "params": vars(a), "runtime_min": (time.time() - t0) / 60}
    with open(f"{a.out}/diagnostics.json", "w") as fh:
        json.dump(diag, fh, indent=1, default=str)
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}")
    print(pd.DataFrame(summary).T.to_string())


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
