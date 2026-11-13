"""Re-test the marine shortlist pairs with depth, latitude and size-fraction controls.

    python marine_retest.py

Run after shortlist.py and marine_metadata.py. Written 2026-11-11, rules fixed before the result.

The 493 marine pairs on the shortlist (confirmed in both folds, S predominantly marine) may only
share a water layer, because marine samples had no depth information. Using the fetched metadata,
each pair is re-tested in exactly the held-out marine samples of shortlist.py's two folds (same
study split, seed 21), restricted to samples WITH a known depth so that every variant uses the same
samples:
  V0  original strata (sub-habitat x study x richness)
  V1  V0 x depth band (<= 10 m, 10-50 m, 50-200 m, > 200 m; Tara layers from the sample description:
      surface -> <= 10 m, deep chlorophyll maximum as its own band, mesopelagic -> > 200 m)
  V2  V1 x latitude band (|lat| < 23.5, 23.5-45, >= 45; unknown kept as its own band)
      x size fraction (lower bound < 1 um vs >= 1 um; unknown kept as its own band)
A pair is confirmed under a variant if its CMH z is significant (BH 5% over the marine pairs,
one-sided) in both folds. Baseline: 5,000 random pairs of marine mOTUs (different order), same test.
Comparing V0 with V2 on the same samples shows how much of the marine shortlist is explained by
depth, latitude and size fraction.

-> results/marine_retest_pairs.tsv     per pair: z and confirmation under V0, V1, V2 per fold
   results/marine_retest_summary.tsv   counts and baseline shares
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

import p9_pipeline as p
from shortlist import RULES, bh, pair_z, strata_stats

HERE = os.path.dirname(os.path.abspath(__file__))
N_BASE, SEED = 5000, 9


def bands(md):
    d = md.depth_m
    depth = pd.cut(d, [-np.inf, 10, 50, 200, np.inf], labels=["d0-10", "d10-50", "d50-200", "d200+"]).astype(str)
    if "layer" in md:                                   # Tara: layer from the sample description
        lay = md.layer.map({"surface": "d0-10", "DCM": "dDCM", "mesopelagic": "d200+"})
        depth = depth.where(d.notna(), lay.fillna("nan"))
    la = md.lat.abs()
    lat = pd.cut(la, [-np.inf, 23.5, 45, np.inf], labels=["trop", "temp", "polar"]).astype(str).where(la.notna(), "lat?")
    lo = md.size_fraction.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    size = np.where(lo.isna(), "size?", np.where(lo < 1, "small", "large"))
    return depth, lat, pd.Series(size, index=md.index)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)
    t0 = time.time()
    pa, S, M = p.load(a.data)
    M = M.loc[pa.index]; S = S.loc[pa.columns]
    Xf = pa.values.astype(np.float32); X = Xf > 0
    pos = pd.Series(np.arange(len(pa)), index=pa.index)
    md = pd.read_csv(f"{a.data}/motusdb/marine_metadata.tsv", sep="\t", index_col=0).reindex(S.index)
    depth, lat, size = bands(md)
    known = (depth.values != "nan") & (S.environment.values == "Saline")
    p.log(f"marine samples with depth: {known.sum()} of {(S.environment == 'Saline').sum()}")

    # the shortlist's study split
    rng = np.random.default_rng(RULES["seed"])
    vc = S.study.value_counts(); big = vc[vc >= RULES["min_study_samples"]].index
    st_env = S[S.study.isin(big)].groupby("study").environment.agg(lambda x: x.mode().iat[0])
    half = {}
    for e, g in st_env.groupby(st_env):
        s_ = list(g.index); rng.shuffle(s_)
        for k, x in enumerate(s_):
            half[x] = "AB"[k % 2]
    hv = S.study.map(half).fillna("-").values

    F = pd.read_csv(f"{a.out}/shortlist_final.tsv", sep="\t")
    F = F[(F.n_folds == 2) & (F.S_dom_env == "Saline")].reset_index(drop=True)
    a_, b_ = pos.reindex(F.S).values.astype(int), pos.reindex(F.H).values.astype(int)
    marine_m = np.where(M.dom_env.values == "Saline")[0]
    order = M["order"].to_numpy(dtype=object)
    r2 = np.random.default_rng(SEED)
    rb = r2.choice(marine_m, (N_BASE * 3, 2))
    rb = rb[(rb[:, 0] != rb[:, 1]) & (order[rb[:, 0]] != order[rb[:, 1]])][:N_BASE]
    base = S[p.STRATA_COL[p.MAIN]].values
    labs = {"V0": base, "V1": base + "|" + depth.values,
            "V2": base + "|" + depth.values + "|" + lat.values + "|" + size.values}
    res = {}
    summ = []
    for fold in ["A", "B"]:
        cols = np.where(known & (hv == fold))[0]          # held-out half of fold "A" is B and vice versa: test both halves
        Xs = Xf[:, cols]; nV = X[:, cols].sum(1)
        for v, lab in labs.items():
            N, Ns, A, sv = strata_stats(Xs, lab[cols])
            for name, aa, bb in [("marine shortlist", a_, b_), ("random marine pairs", rb[:, 0], rb[:, 1])]:
                t = (nV[aa] >= RULES["V_min_det"]) & (nV[bb] >= RULES["V_min_det"])
                z, _ = pair_z(Xs, N, Ns, A, sv, aa, bb)
                q = np.full(len(aa), np.nan)
                if t.any():
                    q[t] = bh(stats.norm.sf(z[t]))
                ok = t & (q < RULES["V_fdr"]) & (z > 0)
                if name == "marine shortlist":
                    res[(fold, v)] = (z, ok, t)
                summ.append({"half": fold, "variant": v, "set": name, "samples": len(cols), "strata": len(pd.unique(lab[cols])),
                             "testable": int(t.sum()), "confirmed": int(ok.sum()), "share_confirmed": float(ok[t].mean()) if t.any() else np.nan})
        p.log(f"half {fold}: {len(cols)} marine samples with depth")
    T = F[["S", "H", "S_species", "S_family", "H_species", "H_family"]].copy()
    for (fold, v), (z, ok, t) in res.items():
        T[f"z_{v}_{fold}"], T[f"ok_{v}_{fold}"], T[f"testable_{fold}"] = z, ok, t
    for v in labs:
        T[f"confirmed_{v}"] = T[f"ok_{v}_A"] & T[f"ok_{v}_B"]
    T.to_csv(f"{a.out}/marine_retest_pairs.tsv", sep="\t", index=False, float_format="%.4g")
    Sm = pd.DataFrame(summ)
    both = {v: int(T[f"confirmed_{v}"].sum()) for v in labs}
    testable_both = int((T.testable_A & T.testable_B).sum())
    for v in labs:
        Sm = pd.concat([Sm, pd.DataFrame([{"half": "both", "variant": v, "set": "marine shortlist", "testable": testable_both,
                                           "confirmed": both[v], "share_confirmed": both[v] / max(testable_both, 1)}])])
    Sm.to_csv(f"{a.out}/marine_retest_summary.tsv", sep="\t", index=False, float_format="%.4g")
    with pd.option_context("display.width", 200):
        print(Sm.round(3).to_string(index=False))
        surv = T[T.confirmed_V2].copy()
        surv["zmin"] = surv[["z_V2_A", "z_V2_B"]].min(1)
        print(f"\nsurvive depth + latitude + size fraction (both halves): {len(surv)} of {testable_both} testable")
        print(surv.sort_values("zmin", ascending=False).head(20)[["S_species", "S_family", "H_species", "H_family", "zmin"]].round(1).to_string(index=False))
    p.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
