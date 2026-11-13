"""What kind of link? Ratio/asymmetry patterns and targeted KEGG hypotheses for the shortlist.

    python interaction_types.py

Run after shortlist.py (and kegg_pilot.py for its cached downloads; missing KEGG files are fetched
from the mOTUs API). Written 2026-11-11, rules fixed before the result. Descriptive: the flags
suggest a type of link to test, they do not demonstrate it.

Pairs: the shortlist pairs confirmed in both folds whose S is not marine (shortlist_final.tsv).

1 Occurrence and abundance pattern (all samples)
    p_H_given_S, p_S_given_H      presence asymmetry
    r_raw                         CLR correlation over joint-presence samples
    r_within                      same after centring both within sub-habitat x study strata
    pS_highH / pS_lowH            P(S present) where H is in its top vs bottom abundance tercile
                                  (among samples with H present, terciles within stratum)
    quad_t                        t of the quadratic term in S_res ~ H_res + H_res^2 (joint samples)
  Flags:
    one_way      p_H_given_S >= 0.9, p_S_given_H <= 0.5, pS_highH / pS_lowH >= 2
    mutual       p_H_given_S >= 0.9 and p_S_given_H >= 0.8
    habitat      r_within < 0.15 or r_within < 0.5 * r_raw (coupling mostly between habitats)
    hump         quad_t <= -3 and the fitted maximum inside the central 90% of H_res
2 Targeted KEGG hypotheses (representative genomes), each pair compared with 10 random partners
  of S from the same environment (seed fixed):
    H2_syntrophy   one partner has mcrA (K00399), the other a hydrogenase
                   ([FeFe] K00532/K00533/K00534 or [NiFe] K06281/K06282)
    succinate      one partner has fumarate reductase frdA (K00244), the other methylmalonyl-CoA
                   mutase (K01847 or K01848+K01849), the succinate -> propionate route
    B12_to_S       S is B12-dependent (metH K00548, no metE K00549, no cobS K02233 / cobT K00768),
                   H can make B12 (cobS or cobT)
    B12_to_H       the reverse
3 Non-MAG genomes (isolates or single-cell genomes) per shortlisted mOTU, from genome_quality.tsv.
  mOTUs-db labels isolates and SAGs alike (_GENO); telling them apart needs NCBI assembly metadata.

-> results/interaction_types.tsv           one row per pair: pattern, flags, KEGG hypotheses
   results/interaction_types_summary.tsv   flag counts; KEGG hypotheses: pairs vs random partners
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import p9_pipeline as p
from abundance import log_counts
from kegg_pilot import API, get, kos_of

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, N_RAND = 5, 10
HYD = {"K00532", "K00533", "K00534", "K06281", "K06282"}
MCR, FRD, MUT, MUT2 = "K00399", "K00244", "K01847", {"K01848", "K01849"}
METH, METE, COB = "K00548", "K00549", {"K02233", "K00768"}


def hypotheses(ka, kb):
    has = lambda k, x: x in k
    mut = lambda k: MUT in k or MUT2 <= k
    b12dep = lambda k: METH in k and METE not in k and not (COB & k)
    return {"H2_syntrophy": (has(ka, MCR) and bool(HYD & kb)) or (has(kb, MCR) and bool(HYD & ka)),
            "succinate": (has(ka, FRD) and mut(kb)) or (has(kb, FRD) and mut(ka)),
            "B12_to_S": b12dep(ka) and bool(COB & kb),
            "B12_to_H": b12dep(kb) and bool(COB & ka)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)
    t0 = time.time()
    F = pd.read_csv(f"{a.out}/shortlist_final.tsv", sep="\t")
    F = F[(F.n_folds == 2) & (F.S_dom_env != "Saline")].reset_index(drop=True)
    pa, S, M = p.load(a.data)
    M = M.loc[pa.index]; S = S.loc[pa.columns]
    pos = pd.Series(np.arange(len(pa)), index=pa.index)
    X = pa.values.astype(bool)
    L, mean_log = log_counts(a.data, pa.index, pa.columns)
    lab = S[p.ABUND_STRATA_COL].values
    p.log(f"{len(F)} non-marine shortlist pairs")

    # ---- 1 occurrence and abundance patterns
    rows = []
    for _, r in F.iterrows():
        i, j = pos[r.S], pos[r.H]
        xs, xh = X[i], X[j]
        out = {"S": r.S, "H": r.H, "p_H_given_S": xh[xs].mean(), "p_S_given_H": xs[xh].mean()}
        J = xs & xh
        cs, ch = L[i, J] - mean_log[J], L[j, J] - mean_log[J]
        out["n_joint"] = int(J.sum())
        out["r_raw"] = float(np.corrcoef(cs, ch)[0, 1]) if J.sum() > 10 else np.nan
        g = pd.Series(lab[J])
        rs = cs - pd.Series(cs).groupby(g.values).transform("mean").values
        rh = ch - pd.Series(ch).groupby(g.values).transform("mean").values
        out["r_within"] = float(np.corrcoef(rs, rh)[0, 1]) if J.sum() > 10 and rs.std() > 0 and rh.std() > 0 else np.nan
        Hs = np.where(xh)[0]
        chH = L[j, Hs] - mean_log[Hs]
        terc = pd.Series(chH).groupby(lab[Hs]).transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False)
                                                         if len(v) >= 6 else pd.Series(np.nan, index=v.index)).values
        lo, hi = xs[Hs][terc == 0], xs[Hs][terc == 2]
        out["pS_lowH"] = float(lo.mean()) if len(lo) else np.nan
        out["pS_highH"] = float(hi.mean()) if len(hi) else np.nan
        if J.sum() > 30 and rh.std() > 0:
            A = np.c_[np.ones_like(rh), rh, rh ** 2]
            beta, res, *_ = np.linalg.lstsq(A, rs, rcond=None)
            sig2 = ((rs - A @ beta) ** 2).sum() / max(len(rs) - 3, 1)
            cov = sig2 * np.linalg.pinv(A.T @ A)
            out["quad_t"] = float(beta[2] / np.sqrt(max(cov[2, 2], 1e-12)))
            vx = -beta[1] / (2 * beta[2]) if beta[2] != 0 else np.nan
            q5, q95 = np.quantile(rh, [0.05, 0.95])
            out["hump_inside"] = bool(beta[2] < 0 and q5 < vx < q95)
        else:
            out["quad_t"], out["hump_inside"] = np.nan, False
        rows.append(out)
    T = F.merge(pd.DataFrame(rows), on=["S", "H"])
    ratio = T.pS_highH / T.pS_lowH.replace(0, np.nan)
    T["flag_one_way"] = (T.p_H_given_S >= 0.9) & (T.p_S_given_H <= 0.5) & ((ratio >= 2) | ((T.pS_lowH == 0) & (T.pS_highH > 0)))
    T["flag_mutual"] = (T.p_H_given_S >= 0.9) & (T.p_S_given_H >= 0.8)
    T["flag_habitat"] = (T.r_within < 0.15) | (T.r_within < 0.5 * T.r_raw)
    T["flag_hump"] = (T.quad_t <= -3) & T.hump_inside

    # ---- 2 KEGG hypotheses
    GQ = pd.read_csv(f"{a.data}/motusdb/genome_quality.tsv", sep="\t", index_col="motu_id")
    kdir = os.path.join(a.data, "motusdb", "kegg"); os.makedirs(kdir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    env = M.dom_env.to_numpy(dtype=object)
    rand = {}
    for s in T.S.unique():
        pool = np.where((env == env[pos[s]]) & (M.index.values != s))[0]
        rand[s] = list(M.index.values[rng.choice(pool, N_RAND, replace=False)])
    need = set(T.S) | set(T.H) | {m for v in rand.values() for m in v}
    rep = GQ.rep_genome.reindex(list(need)).dropna()
    todo = [(m, g) for m, g in rep.items() if not os.path.exists(os.path.join(kdir, f"{g}.kegg.gz"))]
    p.log(f"KEGG: {len(rep)} genomes needed, downloading {len(todo)}")
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda mg: get(API.format(g=mg[1]), os.path.join(kdir, f"{mg[1]}.kegg.gz")), todo))
    K = {m: kos_of(os.path.join(kdir, f"{g}.kegg.gz")) for m, g in rep.items()
         if os.path.exists(os.path.join(kdir, f"{g}.kegg.gz"))}
    hyp_rows, null_rows = [], []
    for _, r in T.iterrows():
        h_ = hypotheses(K.get(r.S, set()), K.get(r.H, set())) if r.S in K and r.H in K else {}
        hyp_rows.append(h_)
        for m in rand[r.S]:
            if r.S in K and m in K:
                null_rows.append(hypotheses(K[r.S], K[m]))
    Hy = pd.DataFrame(hyp_rows).reindex(columns=["H2_syntrophy", "succinate", "B12_to_S", "B12_to_H"])
    T = pd.concat([T, Hy.add_prefix("kegg_")], axis=1)
    Nu = pd.DataFrame(null_rows)
    T["n_ko_S"] = [len(K.get(s, ())) for s in T.S]; T["n_ko_H"] = [len(K.get(h, ())) for h in T.H]

    # ---- 3 non-MAG genomes per mOTU
    T["nonMAG_genomes_S"] = GQ.n_isolates.reindex(T.S).values
    T["nonMAG_genomes_H"] = GQ.n_isolates.reindex(T.H).values

    T.to_csv(f"{a.out}/interaction_types.tsv", sep="\t", index=False, float_format="%.4g")
    summ = [{"item": f, "pairs": int(T[f].sum()), "share": float(T[f].mean())}
            for f in ["flag_one_way", "flag_mutual", "flag_habitat", "flag_hump"]]
    for h in ["H2_syntrophy", "succinate", "B12_to_S", "B12_to_H"]:
        obs = T[f"kegg_{h}"].astype(float)
        summ.append({"item": f"kegg_{h}", "pairs": int(obs.sum()), "share": float(obs.mean()),
                     "random_partner_share": float(Nu[h].mean()) if len(Nu) else np.nan})
    summ.append({"item": "pairs with a non-MAG genome for S or H", "pairs": int(((T.nonMAG_genomes_S > 0) | (T.nonMAG_genomes_H > 0)).sum()),
                 "share": float(((T.nonMAG_genomes_S > 0) | (T.nonMAG_genomes_H > 0)).mean())})
    Sm = pd.DataFrame(summ); Sm.to_csv(f"{a.out}/interaction_types_summary.tsv", sep="\t", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_rows", 120, "display.max_colwidth", 34):
        print(Sm.round(3).to_string(index=False)); print()
        cols = ["S_species", "H_species", "S_dom_env", "p_H_given_S", "p_S_given_H", "r_raw", "r_within", "pS_lowH", "pS_highH",
                "quad_t", "flag_one_way", "flag_mutual", "flag_habitat", "flag_hump", "kegg_H2_syntrophy", "kegg_succinate",
                "kegg_B12_to_S", "kegg_B12_to_H", "nonMAG_genomes_S", "nonMAG_genomes_H"]
        print(T[cols].round(2).to_string(index=False))
    p.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
