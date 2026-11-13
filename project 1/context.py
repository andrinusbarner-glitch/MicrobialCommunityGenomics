"""Are associations properties of the organisms or of the context? (hypotheses 2 and 4)

    python context.py [--splits 2]

Run after p9_pipeline.py. Asks whether the same mOTU pair is associated in the same
way in different environments, or whether associations depend on where they are
measured, and whether widespread ("generalist") mOTUs keep their partners.

Design. A difference between two environments only means something if it is
larger than the difference between two independent sample sets from ONE
environment. So every group (each of the four environment labels; each of the six
sub-habitats with >= 1,000 samples) is split into two disjoint halves of equal size
n (the same n for all groups of a level), the controlled model (sub-habitat x
study x richness strata) is fitted on every half on its own, and
  within-group  half 1 vs half 2 of the same group    = the reproducibility ceiling
  between-group half of group A vs half of group B    = the cross-context agreement
are compared on the pairs testable in both (both mOTUs >= --min-det detections in
both halves of each group). Repeated for --splits random splits.

Pair level (context_compare.tsv), for each comparison:
  of the pairs positive in one set: share positive / not significant / negative in
  the other (averaged over both directions); the same for negative pairs; rank
  correlation of the effect sizes r over all common pairs.
Pair classes (context_pair_classes.tsv): a pair is ROBUST positive (negative) in a
group if positive (negative) in both halves. Among pairs testable in >= 2 groups
and robust in at least one:
  general            robust positive in >= 2 groups
  context-specific   robust positive in exactly one group, robust negative in none
  sign-reversing     robust positive in one group and robust negative in another
mOTU level (context_motus.tsv): for each mOTU with >= 5 positive partners in both
sets, Jaccard overlap of its positive partner sets (partners testable in both);
context-dependence = between-group Jaccard / within-group Jaccard (1 = same
partners everywhere, 0 = different partners in every context).
"""
import argparse
import json
import os
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import p9_pipeline as p

HERE = os.path.dirname(os.path.abspath(__file__))


def fit(Xh, lab, a):
    M = Xh.shape[0]
    O = p.cooccurrence(Xh, a.block)
    m = p.NullModel(Xh, lab, "c")
    thr = p.bh_threshold(m, O, a.block, a.fdr)
    S = np.zeros((M, M), np.int8); R = np.zeros((M, M), np.float16)
    for i in range(0, M, a.block):
        rows = slice(i, min(i + a.block, M))
        z, r = m.block(O[rows], rows)
        S[rows] = np.where(np.abs(z) >= thr, np.sign(z), 0); R[rows] = r
    np.fill_diagonal(S, 0)
    return S, R


def compare(A, B, rng):
    """A, B: dicts with T (global mOTU idx, sorted), S, R. Returns pair metrics and per-mOTU Jaccard."""
    C, ia, ib = np.intersect1d(A["T"], B["T"], assume_unique=True, return_indices=True)
    SA, SB = A["S"][np.ix_(ia, ia)], B["S"][np.ix_(ib, ib)]
    iu = np.triu_indices(len(C), 1)
    a, b = SA[iu], SB[iu]
    out = {"n_motus": int(len(C)), "n_pairs": int(len(a))}
    for v, nm in [(1, "pos"), (-1, "neg")]:
        for k, (x, y) in enumerate([(a, b), (b, a)]):
            base = x == v; n = max(int(base.sum()), 1)
            for lab, cond in [("same", y == v), ("ns", y == 0), ("opposite", y == -v)]:
                out[f"{nm}_{lab}"] = out.get(f"{nm}_{lab}", 0) + float((base & cond).sum() / n) / 2
        out[f"n_{nm}"] = int(((a == v).sum() + (b == v).sum()) // 2)
    s = rng.choice(len(a), min(1_000_000, len(a)), replace=False)
    out["rho_r"] = float(stats.spearmanr(A["R"][np.ix_(ia, ia)][iu][s].astype(np.float32),
                                         B["R"][np.ix_(ib, ib)][iu][s].astype(np.float32)).correlation)
    PA, PB = SA == 1, SB == 1
    dA, dB = PA.sum(1), PB.sum(1)
    inter, union = (PA & PB).sum(1), (PA | PB).sum(1)
    ok = (dA >= 5) & (dB >= 5)
    jac = pd.Series(np.where(ok, inter / np.maximum(union, 1), np.nan), index=C)
    return out, jac, pd.Series(dA, index=C), pd.Series(dB, index=C)


def robust_keys(H1, H2, Mall):
    """Pairs with the same sign in both halves -> (pos keys, neg keys), global pair keys."""
    T = H1["T"]
    iu = np.triu_indices(len(T), 1)
    s1, s2 = H1["S"][iu], H2["S"][iu]
    key = T[iu[0]].astype(np.int64) * Mall + T[iu[1]]
    return key[(s1 == 1) & (s2 == 1)], key[(s1 == -1) & (s2 == -1)]


def run_level(level, groups, n, X, samples, a, rng, Mall):
    halves = {}
    for g, mask in groups.items():
        cols = rng.permutation(np.where(mask)[0])[:2 * n]
        h1, h2 = np.sort(cols[:n]), np.sort(cols[n:])
        T = np.where((X[:, h1].sum(1) >= a.min_det) & (X[:, h2].sum(1) >= a.min_det))[0]
        halves[g] = []
        for h in (h1, h2):
            S, R = fit(X[np.ix_(T, h)], samples[p.STRATA_COL[p.MAIN]].values[h], a)
            halves[g].append({"T": T, "S": S, "R": R})
        p.log(f"[{level}] {g}: {len(T)} mOTUs tested in both halves of {n} samples")
    rows, jw, jb = [], {}, {}
    for g in groups:                                           # within-group ceiling
        o, jac, _, _ = compare(halves[g][0], halves[g][1], rng)
        rows.append({"level": level, "type": "within", "a": g, "b": g, **o}); jw[g] = jac
    for g, h in combinations(groups, 2):                       # between groups: all four half pairings
        acc, jacs = [], []
        for i in (0, 1):
            for j in (0, 1):
                o, jac, _, _ = compare(halves[g][i], halves[h][j], rng); acc.append(o); jacs.append(jac)
        o = {k: float(np.mean([x[k] for x in acc])) for k in acc[0]}
        rows.append({"level": level, "type": "between", "a": g, "b": h, **o})
        jb[(g, h)] = pd.concat(jacs, axis=1).mean(1)
    # pair classes from robust (both-halves) signs
    pos, neg, tested = {}, {}, {}
    for g in groups:
        pos[g], neg[g] = robust_keys(halves[g][0], halves[g][1], Mall)
        tested[g] = halves[g][0]["T"]
    allpos = np.unique(np.concatenate(list(pos.values())))
    lo, hi = allpos // Mall, allpos % Mall
    n_test = sum(np.isin(lo, tested[g]) & np.isin(hi, tested[g]) for g in groups)
    n_pos = sum(np.isin(allpos, pos[g]) for g in groups)
    n_neg = sum(np.isin(allpos, neg[g]) for g in groups)
    m = n_test >= 2
    cls = {"level": level, "pairs_robust_pos_testable_in_ge2": int(m.sum()),
           "general": int((m & (n_pos >= 2)).sum()), "context_specific": int((m & (n_pos == 1) & (n_neg == 0)).sum()),
           "sign_reversing": int((m & (n_pos >= 1) & (n_neg >= 1)).sum())}
    # mOTU table
    mt = []
    for g in groups:
        for k, v in jw[g].dropna().items():
            mt.append({"level": level, "motu": k, "type": "within", "context": g, "jaccard": v})
    for (g, h), s in jb.items():
        for k, v in s.dropna().items():
            mt.append({"level": level, "motu": k, "type": "between", "context": f"{g} | {h}", "jaccard": v})
    del halves
    return rows, cls, mt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--splits", type=int, default=2)
    ap.add_argument("--min-det", type=int, default=10)
    ap.add_argument("--min-habitat", type=int, default=1000)
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    with open(f"{a.out}/diagnostics.json") as fh:
        prm = json.load(fh)["params"]
    pa, samples, _ = p.load(a.data, prm["min_prev"], prm["rich_bins"], prm["min_study"])
    X = pa.values.astype(np.float32); ids = pa.index; Mall = X.shape[0]
    del pa
    env, hab = samples.environment.values, samples.habitat.fillna("none").values
    hc = pd.Series(hab).value_counts()
    levels = [("category", {e: env == e for e in p.ENVS}),
              ("sub-habitat", {h: hab == h for h in hc.index[(hc >= a.min_habitat) & (hc.index != "none")]})]
    rows, cls, mt = [], [], []
    for sp in range(a.splits):
        for level, groups in levels:
            n = min(int(m.sum()) for m in groups.values()) // 2
            r_, c_, m_ = run_level(level, groups, n, X, samples, a, rng, Mall)
            for x in r_: x["split"] = sp
            c_["split"] = sp
            for x in m_: x["split"] = sp
            rows += r_; cls.append(c_); mt += m_
            p.log(f"split {sp} {level} done ({(time.time() - t0) / 60:.1f} min)")
    R = pd.DataFrame(rows)
    num = [c for c in R.columns if c not in ("level", "type", "a", "b", "split")]
    Rm = R.groupby(["level", "type", "a", "b"], sort=False)[num].mean().reset_index()
    Rm.to_csv(f"{a.out}/context_compare.tsv", sep="\t", index=False, float_format="%.5g")
    Cl = pd.DataFrame(cls).groupby("level", sort=False).mean(numeric_only=True).drop(columns="split").reset_index()
    for c in ["general", "context_specific", "sign_reversing"]:
        Cl[f"frac_{c}"] = Cl[c] / Cl.pairs_robust_pos_testable_in_ge2
    Cl.to_csv(f"{a.out}/context_pair_classes.tsv", sep="\t", index=False, float_format="%.5g")
    MT = pd.DataFrame(mt).groupby(["level", "motu", "type"])["jaccard"].mean().unstack("type").reset_index()
    MT = MT.dropna(subset=["within", "between"])
    MT["context_dependence"] = MT.between / MT.within
    MT["motu_id"] = ids[MT.motu.astype(int)]
    mk = pd.read_csv(f"{a.out}/motus.tsv", sep="\t", index_col=0)
    for c in ["species", "genus", "family", "phylum", "prevalence", "breadth"]:
        MT[c] = mk[c].reindex(MT.motu_id).values
    MT.drop(columns="motu").to_csv(f"{a.out}/context_motus.tsv", sep="\t", index=False, float_format="%.4g")
    summ = {lv: {"median_jaccard_within": float(g.within.median()), "median_jaccard_between": float(g.between.median()),
                 "median_context_dependence": float(g.context_dependence.median()), "n_motus": int(len(g)),
                 "rho_breadth_vs_context_dependence": float(stats.spearmanr(g.breadth, g.context_dependence, nan_policy="omit").correlation)}
            for lv, g in MT.groupby("level")}
    with open(f"{a.out}/context_summary.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print(Rm[["level", "type", "a", "b", "n_motus", "n_pos", "pos_same", "pos_ns", "pos_opposite", "n_neg", "neg_same",
              "neg_opposite", "rho_r"]].round(3).to_string(index=False))
    print(Cl.round(3).to_string(index=False))
    print(json.dumps(summ, indent=1))
    p.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
