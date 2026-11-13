"""Shortlist of clearly statistically linked mOTU pairs, validated in held-out studies.

    python shortlist.py

Run after p9_pipeline.py. Written 2026-11-11, rules fixed before the result.

Goal: not to identify symbionts, but to list pairs whose statistical link is strong, specific and
reproducible enough to warrant experiments. "Reproducible" is tested on data that played no part
in selecting the pair:

  split      the studies with >= 30 samples are divided at random into two halves, separately
             within each environment (seed fixed). Fold A: discovery = all samples NOT in half B
             (half A plus all small studies), validation = samples of half B only. Fold B swaps.
  discovery  (main-model strata, sub-habitat x study x richness, restricted to discovery samples)
             D1 association   CMH z >= 3.15 (main model's 1% FDR threshold) and effect r >= 0.2
             D2 dependency    P(H | S) >= 0.9 in discovery samples, S less prevalent than H,
                              S detected in >= 30 discovery samples (direction S -> H)
             D3 not relatives H from a different order than S
             D4 abundance     pooled CLR correlation within sub-habitat x study strata (>= 30
                              joint samples per stratum): z >= 3 and r >= 0.3
             D5 specific      per S, only its best H (highest discovery r) among pairs passing D1-D4
  validation (validation samples only; strata sub-habitat x richness within the held-out studies)
             V1 presence/absence CMH z, one-sided p, Benjamini-Hochberg 5% over the fold's
                shortlist pairs testable in validation (both detected >= 3 times there)
             V2 abundance coupling in validation, reported (z >= 1.645), not required
  shortlist  pairs passing V1; "both folds" = selected and validated in both directions.
Baselines (same validation test, same fold): random different-order pairs of the same environment,
and a random sample of pairs passing D1 only. TM7 check: TM7 pairs in the shortlist and their
host phylum.

-> results/shortlist_pairs.tsv     every discovery pick, per fold, with validation statistics
   results/shortlist_final.tsv     validated pairs (one row per pair, folds in which validated)
   results/shortlist_summary.tsv   counts and validation rates vs baselines
   results/shortlist.json          rules + hash, headline numbers
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

import p9_pipeline as p
from abundance import log_counts, fisher_pool

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = {"version": "v1 (2026-11-11, fixed before the result)", "min_study_samples": 30, "seed": 21,
         "D1_z": 3.15, "D1_r": 0.2, "D2_cover": 0.9, "D2_min_det": 30, "D3": "different order",
         "D4_abund_z": 3.0, "D4_abund_r": 0.3, "D4_min_joint": 30, "D5": "best H per S",
         "V_min_det": 3, "V_fdr": 0.05, "n_baseline": 20000}


def strata_stats(Xs, lab):
    st = pd.unique(lab)
    N = np.stack([Xs[:, lab == s].sum(1) for s in st], 1).astype(np.float64)
    Ns = np.array([(lab == s).sum() for s in st], float)
    A = N * (Ns - N) / (Ns * np.sqrt(np.maximum(Ns - 1, 1)))
    sv = (N * (Ns - N) / Ns).sum(1)
    return N, Ns, A, sv


def pair_z(Xs, N, Ns, A, sv, a, b):
    z = np.empty(len(a)); r = np.empty(len(a))
    for c in range(0, len(a), 20000):
        ia, ib = a[c:c + 20000], b[c:c + 20000]
        O = np.einsum("ij,ij->i", Xs[ia], Xs[ib]).astype(np.float64)
        E = np.einsum("ij,ij->i", N[ia] / Ns, N[ib]); V = np.einsum("ij,ij->i", A[ia], A[ib])
        z[c:c + 20000] = (O - E) / np.sqrt(np.maximum(V, 1e-12))
        r[c:c + 20000] = (O - E) / np.sqrt(np.maximum(sv[ia] * sv[ib], 1e-12))
    return z, r


def pair_abund(L, P, mean_log, lab, a, b, min_n, chunk=3000):
    """Fisher-pooled CLR correlation of pairs (a, b) within strata, joint-presence samples only
    (pairs processed in chunks to bound memory)."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) > chunk:
        parts = [pair_abund(L, P, mean_log, lab, a[c:c + chunk], b[c:c + chunk], min_n, chunk)
                 for c in range(0, len(a), chunk)]
        return np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts])
    Zs = np.zeros(len(a)); Ws = np.zeros(len(a))
    for st in pd.unique(lab):
        cols = np.where(lab == st)[0]
        if len(cols) < min_n:
            continue
        Pa, Pb = P[np.ix_(a, cols)], P[np.ix_(b, cols)]
        J = Pa & Pb
        n = J.sum(1)
        ok = n >= min_n
        if not ok.any():
            continue
        La = (L[np.ix_(a, cols)] - mean_log[cols]) * J; Lb = (L[np.ix_(b, cols)] - mean_log[cols]) * J
        ma, mb = La.sum(1) / np.maximum(n, 1), Lb.sum(1) / np.maximum(n, 1)
        da, db = (La - ma[:, None]) * J, (Lb - mb[:, None]) * J
        rr = (da * db).sum(1) / np.sqrt(np.maximum((da ** 2).sum(1) * (db ** 2).sum(1), 1e-12))
        rr = np.clip(rr, -0.999999, 0.999999)
        w = np.where(ok, n - 3.0, 0.0)
        Zs += np.arctanh(rr) * w; Ws += w
    z = np.where(Ws > 0, Zs / np.sqrt(np.maximum(Ws, 1e-12)), np.nan)
    r = np.where(Ws > 0, np.tanh(Zs / np.maximum(Ws, 1e-12)), np.nan)
    return z, r


def bh(pv):
    pv = np.asarray(pv); n = len(pv); o = np.argsort(pv)
    q = np.empty(n); q[o] = np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a_ = ap.parse_args(argv)
    t0 = time.time(); R_ = RULES
    h = hashlib.sha256(json.dumps(R_, sort_keys=True).encode()).hexdigest()[:12]
    p.log(f"shortlist {R_['version']} (hash {h})")
    pa, S, M = p.load(a_.data)
    M = M.loc[pa.index]; S = S.loc[pa.columns]
    X = pa.values.astype(bool); Xf = X.astype(np.float32)
    L, mean_log = log_counts(a_.data, pa.index, pa.columns)
    order = M["order"].to_numpy(dtype=object); env = M["dom_env"].to_numpy(dtype=object); ids = M.index.to_numpy(dtype=object)
    labs_main = S[p.STRATA_COL[p.MAIN]].values
    labs_ab = S[p.ABUND_STRATA_COL].values

    # ---- split studies within environment
    rng = np.random.default_rng(R_["seed"])
    vc = S.study.value_counts(); big = vc[vc >= R_["min_study_samples"]].index
    st_env = S[S.study.isin(big)].groupby("study").environment.agg(lambda x: x.mode().iat[0])
    half = {}
    for e, g in st_env.groupby(st_env):
        s = list(g.index); rng.shuffle(s)
        for k, x in enumerate(s):
            half[x] = "AB"[k % 2]
    S["half"] = S.study.map(half).fillna("-")
    p.log("studies per half: " + ", ".join(f"{k} {sum(v == k for v in half.values())}" for k in "AB"))

    picks, summ = [], []
    for fold, val_half in [("A", "B"), ("B", "A")]:
        D = np.where(S.half.values != val_half)[0]; V = np.where(S.half.values == val_half)[0]
        XD = Xf[:, D]; nD = X[:, D].sum(1)
        N, Ns, A, sv = strata_stats(XD, labs_main[D])
        cand, d1_sample = [], []
        for c in range(0, len(pa), 800):
            blk = np.arange(c, min(c + 800, len(pa)))
            O = (XD[blk] @ XD.T).astype(np.float64)
            E = (N[blk] / Ns) @ N.T; Vv = A[blk] @ A.T
            z = (O - E) / np.sqrt(np.maximum(Vv, 1e-12)); r = (O - E) / np.sqrt(np.maximum(np.outer(sv[blk], sv), 1e-12))
            cover = O / np.maximum(nD[blk][:, None], 1)
            d1 = (z >= R_["D1_z"]) & (r >= R_["D1_r"])
            d1[np.arange(len(blk)), blk] = False
            ii, jj = np.where(d1)
            if len(ii):
                take = rng.random(len(ii)) < 0.002
                d1_sample.append(np.c_[blk[ii[take]], jj[take]])
            m = d1 & (cover >= R_["D2_cover"]) & (nD[blk][:, None] < nD[None, :]) & (nD[blk][:, None] >= R_["D2_min_det"]) \
                & (order[blk][:, None] != order[None, :])
            ii, jj = np.where(m)
            if len(ii):
                cand.append(pd.DataFrame({"S": blk[ii], "H": jj, "z_D": z[ii, jj], "r_D": r[ii, jj], "cover_D": cover[ii, jj]}))
        C = pd.concat(cand, ignore_index=True)
        n_d123 = len(C)
        zA, rA = pair_abund(L[:, D], X[:, D], mean_log[D], labs_ab[D], C.S.values, C.H.values, R_["D4_min_joint"])
        C["abund_z_D"], C["abund_r_D"] = zA, rA
        C = C[(C.abund_z_D >= R_["D4_abund_z"]) & (C.abund_r_D >= R_["D4_abund_r"])]
        n_d4 = len(C)
        C = C.sort_values("r_D", ascending=False).drop_duplicates("S")
        p.log(f"fold {fold}: D1-D3 {n_d123:,} pairs, D4 {n_d4:,}, D5 {len(C):,} (one per symbiont)")

        # ---- validation
        XV = Xf[:, V]; nV = X[:, V].sum(1)
        Nv, Nsv, Av, svv = strata_stats(XV, labs_main[V])

        def validate(a, b):
            t = (nV[a] >= R_["V_min_det"]) & (nV[b] >= R_["V_min_det"])
            z, r = pair_z(XV, Nv, Nsv, Av, svv, a, b)
            return t, z, r
        t, zv, rv = validate(C.S.values, C.H.values)
        C = C.assign(testable_V=t, z_V=zv, r_V=rv)
        C["q_V"] = np.nan
        C.loc[C.testable_V, "q_V"] = bh(stats.norm.sf(C.loc[C.testable_V, "z_V"]))
        za, ra = pair_abund(L[:, V], X[:, V], mean_log[V], labs_ab[V], C.S.values, C.H.values, R_["D4_min_joint"])
        C["abund_z_V"], C["abund_r_V"] = za, ra
        C["validated"] = C.testable_V & (C.q_V < R_["V_fdr"]) & (C.z_V > 0)
        C["fold"] = fold
        picks.append(C)
        # baselines
        rb = rng.integers(0, len(pa), (R_["n_baseline"] * 4, 2))
        rb = rb[(rb[:, 0] != rb[:, 1]) & (env[rb[:, 0]] == env[rb[:, 1]]) & (order[rb[:, 0]] != order[rb[:, 1]])][: R_["n_baseline"]]
        d1s = np.concatenate(d1_sample)[: R_["n_baseline"]]
        for name, a, b in [("shortlist (D1-D5)", C.S.values, C.H.values), ("D1 only (significant, r >= 0.2)", d1s[:, 0], d1s[:, 1]),
                           ("random same-env, different order", rb[:, 0], rb[:, 1])]:
            t_, z_, _ = validate(a, b)
            q_ = bh(stats.norm.sf(z_[t_])) if t_.any() else np.array([])
            summ.append({"fold": fold, "set": name, "n": len(a), "testable_in_validation": int(t_.sum()),
                         "share_positive_z_1645": float((z_[t_] >= 1.645).mean()) if t_.any() else np.nan,
                         "share_validated_bh5": float(((q_ < R_["V_fdr"]) & (z_[t_] > 0)).mean()) if t_.any() else np.nan,
                         "median_z_V": float(np.median(z_[t_])) if t_.any() else np.nan})
        p.log(f"fold {fold}: {int(C.testable_V.sum())} testable in validation, {int(C.validated.sum())} validated")

    Pk = pd.concat(picks, ignore_index=True)
    for side in ["S", "H"]:
        for col in ["species", "family", "order", "phylum", "dom_env", "class"]:
            Pk[f"{side}_{col}"] = M[col].values[Pk[side].values]
        Pk[side] = ids[Pk[side].values]
    Pk.to_csv(f"{a_.out}/shortlist_pairs.tsv", sep="\t", index=False, float_format="%.4g")
    Vd = Pk[Pk.validated]
    F = Vd.groupby(["S", "H"]).agg(folds=("fold", lambda x: "".join(sorted(x))), n_folds=("fold", "size"),
                                   z_V_min=("z_V", "min"), r_V_mean=("r_V", "mean"), abund_r_V=("abund_r_V", "mean"),
                                   r_D=("r_D", "mean"), cover_D=("cover_D", "mean")).reset_index()
    meta = Pk.drop_duplicates(["S", "H"]).set_index(["S", "H"])[[c for c in Pk.columns if c[:2] in ("S_", "H_")]]
    F = F.join(meta, on=["S", "H"]).sort_values(["n_folds", "z_V_min"], ascending=False)
    F.to_csv(f"{a_.out}/shortlist_final.tsv", sep="\t", index=False, float_format="%.4g")
    T = pd.DataFrame(summ); T.to_csv(f"{a_.out}/shortlist_summary.tsv", sep="\t", index=False, float_format="%.4g")
    with pd.option_context("display.width", 230, "display.max_rows", 60, "display.max_colwidth", 40):
        print(T.round(3).to_string(index=False)); print()
        print(f"validated pairs: {len(F)} ({(F.n_folds == 2).sum()} in both folds); by environment of S:")
        print(F.S_dom_env.value_counts().to_string()); print()
        print(F.head(30)[["S_species", "S_phylum", "H_species", "H_phylum", "S_dom_env", "folds", "z_V_min", "r_V_mean", "abund_r_V"]].round(2).to_string(index=False))
        tm = Pk[Pk.S_class == "Saccharimonadia"]
        print(f"\nTM7 picked in discovery: {len(tm)} (folds combined), validated {int(tm.validated.sum())}; host phyla of validated: "
              f"{tm[tm.validated].H_phylum.value_counts().to_dict()}")
    with open(f"{a_.out}/shortlist.json", "w") as fh:
        json.dump({"rules": R_, "rules_hash": h, "n_validated": int(len(F)), "n_both_folds": int((F.n_folds == 2).sum()),
                   "runtime_min": (time.time() - t0) / 60}, fh, indent=1)
    p.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
