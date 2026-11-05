"""
Co-occurrence clustering of mOTUs (BC2026 mOTUs subset).

Pipeline
  1. Load PA matrix, taxonomy, environment; align IDs.
  2. Pairwise association z-scores under a hypergeometric null, computed
       (a) globally, and
       (b) stratified by environment (Cochran-Mantel-Haenszel style), which
           removes co-occurrence that is merely "both like the same habitat".
  3. Keep significant edges (BH-FDR), build signed network.
  4. Cluster the positive network (Louvain), annotate clusters with
       environment affinity and taxonomic composition.

Usage:
  python cooccurrence_clusters.py --data DIR --out OUT [--fdr 0.01] [--knn 50]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx


def load(data_dir):
    pa = pd.read_csv(f"{data_dir}/motus_subset_pa.tsv.gz", sep="\t", index_col=0)
    tax = pd.read_csv(f"{data_dir}/motus_subset_taxonomy.tsv", sep="\t")
    env = pd.read_csv(f"{data_dir}/motus_subset_environment.tsv", sep="\t")
    # PA is mOTU x sample; make sure environment lines up with columns
    env = env.set_index("sample_id").reindex(pa.columns)
    assert env["environment"].notna().all(), "samples without environment label"
    return pa, tax, env["environment"]
"""
Loads presence/absence data (whether each mOTU is detected).
Loads taxonomy (which organisms they belong to).
Loads environmental information (e.g. habitat type).
Makes sure sample IDs match across datasets.
"""

def hypergeom_stats(X):
    """X: motus x samples (float32 0/1). Returns observed, expected, variance
    of pairwise co-occurrence counts under the hypergeometric null
    (fixed mOTU prevalences, N samples)."""
    N = X.shape[1]
    n = X.sum(1)
    O = X @ X.T
    E = np.outer(n, n) / N
    # Var = n_i n_j (N-n_i)(N-n_j) / (N^2 (N-1))
    a = n * (N - n)
    V = np.outer(a, a) / (N**2 * (N - 1))
    return O, E, V


def association_z(pa, env, stratify=True):
    X = pa.values.astype(np.float32)
    if not stratify:
        O, E, V = hypergeom_stats(X)
        return (O - E) / np.sqrt(np.maximum(V, 1e-12))
    Osum = Esum = Vsum = 0
    for e in env.unique():
        cols = np.where(env.values == e)[0]
        O, E, V = hypergeom_stats(X[:, cols])
        Osum, Esum, Vsum = Osum + O, Esum + E, Vsum + V
    return (Osum - Esum) / np.sqrt(np.maximum(Vsum, 1e-12))

"""
Compares how often two mOTUs occur together against how often you'd expect by chance.

Calculates a z-score:
- Positive = occur together more than expected.
- Negative = occur together less than expected.

Can calculate this globally or separately within each environment (stratification), avoiding associations caused simply by organisms preferring the same habitat.
"""

def load_counts(data_dir, pa):
    """Read the read-count matrix and align it exactly to the PA matrix."""
    cnt = pd.read_csv(f"{data_dir}/motus_subset_matrix.tsv.gz", sep="\t", index_col=0)
    cnt = cnt.reindex(index=pa.index, columns=pa.columns)
    assert cnt.notna().all().all(), "count matrix does not match PA matrix IDs"
    # sanity check: PA should equal (counts > 0)
    mismatch = ((cnt.values > 0) != (pa.values > 0)).mean()
    print(f"counts vs PA: {mismatch:.4%} of cells disagree")
    return cnt

"""
Loads the original read-count matrix, which contains the abundance of each mOTU in each sample.
Aligns the count matrix with the presence/absence matrix.
Checks that the two matrices agree (i.e. counts greater than zero correspond to presence).
This allows the analysis to investigate not only whether microorganisms occur together, but also how their abundances vary together.
"""

def clr_present(counts):
    """Centred log-ratio computed ONLY over the mOTUs detected in each sample.
    log(count) minus the mean log(count) of the detected mOTUs in that sample.
    Library size cancels, so this is a compositional (ratio) representation.
    Returns (L, P): L = CLR values (0 where absent), P = presence mask."""
    C = counts.values.astype(np.float64)
    P = (C > 0).astype(np.float64)
    logC = np.where(C > 0, np.log(np.where(C > 0, C, 1.0)), 0.0)
    mean_log = logC.sum(axis=0) / P.sum(axis=0)          # per sample
    L = (logC - mean_log[None, :]) * P
    return L, P
"""
Applies a centred log-ratio (CLR) transformation to the read counts.
Accounts for the compositional nature of microbiome sequencing data.
Calculates the log abundance of each detected mOTU relative to the mean log abundance of all detected mOTUs in that sample.
Keeps absent mOTUs excluded from subsequent pairwise correlations.
"""

def _fisher_accumulate(L, P, cols, min_n, block, Zsum, Wsum):
    """Pairwise-complete Pearson r of CLR values for one stratum (set of
    sample columns), converted to Fisher z and accumulated with weight n-3.
    Only samples where BOTH mOTUs are present contribute to a pair."""
    Ls, Ps = L[:, cols], P[:, cols]
    Ls2 = Ls ** 2
    for i in range(0, Ls.shape[0], block):
        b = slice(i, min(i + block, Ls.shape[0]))
        n = Ps[b] @ Ps.T                                  # joint presence counts
        sxy = Ls[b] @ Ls.T
        sx = Ls[b] @ Ps.T                                 # sum of x_i over joint samples
        sy = Ps[b] @ Ls.T                                 # sum of x_j over joint samples
        sxx = Ls2[b] @ Ps.T
        syy = Ps[b] @ Ls2.T
        num = n * sxy - sx * sy
        den = np.sqrt(np.maximum((n * sxx - sx ** 2) * (n * syy - sy ** 2), 1e-12))
        r = np.clip(num / den, -0.999999, 0.999999)
        w = np.where(n >= min_n, n - 3.0, 0.0)
        Zsum[b] += (np.arctanh(r) * w).astype(np.float32)
        Wsum[b] += w.astype(np.float32)

"""
Calculates Pearson correlations between pairs of mOTUs using their CLR-transformed abundances.
Only uses samples in which both mOTUs are detected (pairwise-complete observations).
Converts correlations into Fisher z-scores.
Weights correlations according to the number of samples in which both mOTUs are present.
Requires at least 30 jointly detected samples by default.
"""

def abundance_z(counts, env, stratify=True, min_n=30, block=500):
    """Association z-score from how mOTU abundances co-vary across samples
    where both are detected (CLR + pairwise-complete correlation).
    stratify=True: correlation is computed inside each environment and the
    Fisher-z values are pooled (weight n-3), so shared habitat preference
    cannot produce an edge on its own. Returns a (mOTU x mOTU) z matrix."""
    L, P = clr_present(counts)
    m = L.shape[0]
    Zsum = np.zeros((m, m), np.float32)
    Wsum = np.zeros((m, m), np.float32)
    groups = [np.where(env.values == e)[0] for e in env.unique()] if stratify \
        else [np.arange(L.shape[1])]
    for cols in groups:
        _fisher_accumulate(L, P, cols, min_n, block, Zsum, Wsum)
    z = np.where(Wsum > 0, Zsum / np.sqrt(np.maximum(Wsum, 1e-12)), 0.0)
    np.fill_diagonal(z, 0)
    return z.astype(np.float32)

"""
Combines the abundance correlations calculated in the previous step into an association z-score matrix.
Can calculate associations globally or separately within each environment.
Stratification helps prevent correlations caused simply by shared environmental preferences.
Produces a matrix describing positive and negative abundance associations between mOTUs.
"""

def bh_mask(z, fdr):
    """Benjamini-Hochberg on the upper triangle (two-sided). Returns boolean
    matrix of significant pairs (symmetric)."""
    n = z.shape[0]
    iu = np.triu_indices(n, k=1)
    p = 2 * stats.norm.sf(np.abs(z[iu]))
    order = np.argsort(p)
    m = len(p)
    thresh = fdr * (np.arange(1, m + 1) / m)
    ok = p[order] <= thresh
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    sig = np.zeros(m, bool)
    sig[order[:k]] = True
    S = np.zeros((n, n), bool)
    S[iu] = sig
    return S | S.T

"""
Applies the Benjamini-Hochberg false discovery rate (FDR) correction.
Accounts for the large number of pairwise comparisons being performed.
Identifies statistically significant associations using the selected FDR threshold (default = 0.01).
Returns a matrix indicating which associations are significant.
"""

def build_graph(z, motus, sig, knn=None, sign=+1):
    """Network of significant edges of one sign. Optionally keep, per node,
    only its top-`knn` strongest neighbours (edge kept if either end wants it)."""
    Z = np.where(sig & (np.sign(z) == sign), np.abs(z), 0.0)
    np.fill_diagonal(Z, 0)
    if knn:
        keep = np.zeros_like(Z, bool)
        idx = np.argpartition(-Z, knn, axis=1)[:, :knn]
        rows = np.arange(Z.shape[0])[:, None]
        keep[rows, idx] = True
        Z = np.where((keep | keep.T) & (Z > 0), Z, 0.0)
    G = nx.from_numpy_array(Z)
    G = nx.relabel_nodes(G, dict(enumerate(motus)))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G
"""
Converts significant associations into networks.
Each node represents one mOTU.
Each edge represents a significant association between two mOTUs.
Edge weights represent association strength (absolute z-score).
Creates separate positive and negative networks:
- Positive network = mOTUs that occur together more than expected or have positively correlated abundances.
- Negative network = mOTUs that occur together less than expected or have negatively correlated abundances.
Optionally limits connections to the 50 strongest neighbours per mOTU (knn=50) to reduce network complexity.
"""

def cluster(G, resolution=1.0, seed=0):
    comms = nx.community.louvain_communities(G, weight="weight",
                                             resolution=resolution, seed=seed)
    comms = sorted(comms, key=len, reverse=True)
    return {m: i for i, c in enumerate(comms) for m in c}

"""
Applies the Louvain community detection algorithm to identify groups of strongly connected mOTUs.
Groups microorganisms that have relatively strong connections within the same cluster.
Clusters positive and negative networks separately.
Allows the clustering resolution to be adjusted to obtain larger or smaller clusters.
"""

def env_affinity(pa, env):
    """Fraction of each mOTU's detections falling in each environment, and
    prevalence within each environment (more robust to uneven sample counts)."""
    prev = pa.T.groupby(env.values).mean().T          # motu x env, P(present | env)
    aff = prev.div(prev.sum(axis=1), axis=0)               # normalised profile
    return prev, aff

"""
- Calculates the prevalence of each mOTU within each environment (proportion of samples in which it is detected).
- Calculates environmental affinity by normalising prevalence across environments.
- Identifies which environments are most strongly associated with each mOTU
"""

def annotate(labels, pa, tax, env, rank="phylum"):
    prev, aff = env_affinity(pa, env)
    df = pd.DataFrame({"motu": list(labels), "cluster": list(labels.values())})
    df = df.set_index("motu")
    df = df.join(aff.add_prefix("aff_"))
    tcol = tax.columns[0]
    t = tax.set_index(tcol)
    # GTDB strings like d__Bacteria;p__Firmicutes;...  -> pull requested rank
    tag = {"phylum": "p__", "class": "c__", "order": "o__", "family": "f__"}[rank]
    def pick(row):
        for col in t.columns:
            for part in str(row[col]).split(";"):
                if part.strip().startswith(tag):
                    return part.strip()
        return np.nan
    df[rank] = t.reindex(df.index).apply(pick, axis=1)
    summ = []
    for c, g in df.groupby("cluster"):
        a = g.filter(like="aff_").mean()
        top_tax = g[rank].value_counts(normalize=True)
        summ.append({"cluster": c, "n_motus": len(g),
                     "dominant_env": a.idxmax().replace("aff_", ""),
                     "env_purity": a.max(),
                     f"top_{rank}": top_tax.index[0] if len(top_tax) else None,
                     f"top_{rank}_frac": top_tax.iloc[0] if len(top_tax) else np.nan})
    return df, pd.DataFrame(summ)

"""
Combines cluster assignments with environmental and taxonomic information.
Extracts the taxonomic classification of each mOTU (default = phylum).
Calculates the number of mOTUs in each cluster.
Identifies the dominant environmental affinity of each cluster.
Calculates environmental purity (average affinity towards the dominant environment).
Identifies the most common phylum and its proportion within each cluster.
Produces individual mOTU annotations and a cluster summary table.
"""


def run_network_analysis(tag, z, pa, tax, env, a):
    os.makedirs(a.out, exist_ok=True)   # safety net: never write into a missing folder
    sig = bh_mask(z, a.fdr)
    n_pos = int(((z > 0) & sig).sum() // 2)
    n_neg = int(((z < 0) & sig).sum() // 2)
    print(f"[{tag}] significant pairs: +{n_pos}  -{n_neg}")
    result = {"labels": {}, "summary": {"analysis": tag, "sig_pos_pairs": n_pos,
                                        "sig_neg_pairs": n_neg}}
    for sign, association in [(+1, "positive"), (-1, "negative")]:
        G = build_graph(z, pa.index, sig, knn=a.knn, sign=sign)
        if G.number_of_nodes() == 0:
            print(f"[{tag}] no significant {association} edges; skipping")
            continue
        labels = cluster(G, a.resolution)
        result["labels"][f"{tag}_{association}"] = labels
        result["summary"][f"{association}_nodes"] = G.number_of_nodes()
        result["summary"][f"{association}_edges"] = G.number_of_edges()
        result["summary"][f"{association}_clusters"] = len(set(labels.values()))
        df, summ = annotate(labels, pa, tax, env)
        df.to_csv(f"{a.out}/clusters_{tag}_{association}.tsv", sep="\t")
        summ.to_csv(f"{a.out}/cluster_summary_{tag}_{association}.tsv",
                    sep="\t", index=False)
        print(f"[{tag}] {association} network: {G.number_of_nodes()} mOTUs, "
              f"{G.number_of_edges()} edges, {len(summ)} clusters")
        print(summ.head(10).to_string(index=False))
    iu = np.where(np.triu(sig, 1))
    pd.DataFrame({"a": pa.index[iu[0]], "b": pa.index[iu[1]], "z": z[iu]}
                 ).to_csv(f"{a.out}/edges_{tag}.tsv.gz", sep="\t", index=False)
    return result
"""
Brings together the statistical testing, network construction, clustering and annotation steps.
Runs the analysis separately for positive and negative associations.
Saves the resulting networks' significant edges, cluster assignments and cluster summaries.
Reports the number of significant associations, nodes, edges and clusters.
"""

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="p3_out")
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--knn", type=int, default=50)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--no-stratify", action="store_true",
                    help="skip the environment-stratified analyses")
    ap.add_argument("--no-abundance", action="store_true",
                    help="skip the abundance (read-count) analyses")
    ap.add_argument("--run-name", default=None,
                    help="subfolder of --out for this run's files (default: "
                         "'with_abundance' or 'pa_only'), so runs never overwrite each other")
    ap.add_argument("--min-n", type=int, default=30,
                    help="min joint-presence samples per environment for an abundance correlation")
    a = ap.parse_args(argv)
    run_name = a.run_name or ("pa_only" if a.no_abundance else "with_abundance")
    a.out = os.path.join(a.out, run_name)
    os.makedirs(a.out, exist_ok=True)
    print(f"writing results to {a.out}")

    pa, tax, env = load(a.data)
    print(f"{pa.shape[0]} mOTUs x {pa.shape[1]} samples")
    counts = None if a.no_abundance else load_counts(a.data, pa)
    results = []

    for strat_tag, strat in [("stratified", True), ("global", False)]:
        if a.no_stratify and strat:
            continue
        # 1) presence/absence co-occurrence
        z_pa = association_z(pa, env, stratify=strat)
        results.append(run_network_analysis(f"pa_{strat_tag}", z_pa, pa, tax, env, a))
        if counts is None:
            continue
        # 2) abundance covariation (where both mOTUs are present)
        z_ab = abundance_z(counts, env, stratify=strat, min_n=a.min_n)
        results.append(run_network_analysis(f"abund_{strat_tag}", z_ab, pa, tax, env, a))
        # 3) combined evidence (Stouffer): presence overlap + abundance covariation
        z_comb = (z_pa + z_ab) / np.sqrt(2.0)
        results.append(run_network_analysis(f"combined_{strat_tag}", z_comb, pa, tax, env, a))
        del z_pa, z_ab, z_comb

    # ---- side-by-side comparison files -------------------------------
    pd.DataFrame([r["summary"] for r in results]).to_csv(
        f"{a.out}/run_summary.tsv", sep="\t", index=False)
    cmp = pd.DataFrame(index=pa.index)
    for r in results:
        for name, labels in r["labels"].items():
            cmp[name] = pd.Series(labels)          # NaN where mOTU not in that network
    cmp = cmp.join(env_affinity(pa, env)[1].add_prefix("aff_"))
    cmp.to_csv(f"{a.out}/cluster_comparison.tsv", sep="\t")
    print(f"saved run_summary.tsv and cluster_comparison.tsv in {a.out}")

"""
Defines the analysis parameters (FDR, k-nearest neighbours, clustering resolution, etc.).
Loads and aligns all datasets.
Runs the analyses globally and stratified by environment.
Performs three types of analysis:
- Presence/absence: based on co-occurrence patterns.
- Abundance: based on correlations in CLR-transformed read counts.
- Combined: combines presence/absence and abundance association z-scores using Stouffer's method.
Saves a summary comparing the different analyses.
Saves a cluster comparison table showing cluster assignments across analyses, together with environmental affinity.
"""

# Do not auto-run inside Jupyter/IPython (sys.argv there is the kernel's, not yours)
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()

"""
run_summary.tsv	==== Summary of significant associations, network sizes and number of clusters for each analysis.
cluster_comparison.tsv	==== Compares mOTU cluster assignments across all analyses and includes environmental affinities.
edges_*.tsv.gz	==== Significant pairwise associations and their z-scores.
clusters_*.tsv	==== Individual mOTU cluster assignments with environmental and taxonomic annotations.
cluster_summary_*.tsv	==== Summary of each cluster, including size, dominant environment and dominant phylum.
"""