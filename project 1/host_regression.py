"""Host-abundance regression: does a symbiont's abundance depend on its host's, after habitat and study?

    python host_regression.py

Run after tm7_benchmark.py (it reuses the benchmark's frozen tiers, control rule and metrics).

This is the regression version of the abundance coupling of presence/absence pairs, built
like MaAsLin2's default model (Mallick et al. 2021) so it can be read the same way:
  - normalisation TSS (count / reads of the sample), transform log2, zeros replaced by half the
    smallest non-zero value of that mOTU (MaAsLin2 'LOG');
  - for a query mOTU S (a TM7, or a matched control) and every candidate partner H, a linear model
    over the samples where S is detected:
        log2 TSS(S) ~ log2 TSS(H) + log2 reads + fixed effect of sub-habitat x study
    (MaAsLin2 would usually put study in as a random effect; here every sub-habitat x study cell gets
    its own intercept, which is stricter and lets all ~9,600 partners be fitted at once);
  - the score of H is the t statistic of its coefficient. Partners are ranked by it.
  - variant 'unadjusted': the same model with log2 reads only, to show what the adjustment changes.
Only samples where S is present are used: an obligate symbiont's abundance should track its host's
where the symbiont exists. Host absence in those samples (host at its half-minimum) counts against H.

The ranking is then scored with exactly the frozen TM7 benchmark (tm7_benchmark.DEFS: tiers,
matched controls, metrics, draws), so the numbers sit next to the presence/absence results.

lme4 check: for each TM7 and its top partner the data are written to
host_regression_lme4_input.tsv; host_regression_lme4.R refits them with study as a random effect
(log2 S ~ log2 H + log2 reads + sub-habitat + (1 | study)), the MaAsLin2 mixed-model form.

-> results/host_regression_queries.tsv    one row per query x variant (benchmark metrics)
   results/host_regression_summary.tsv    TM7 vs matched controls per variant x group x metric
   results/host_regression_tm7_hosts.tsv  top 5 partners of every TM7 (adjusted model): the candidate host list
   results/host_regression_lme4_input.tsv data for the lme4 check
   results/host_regression.json           settings and headline numbers
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import p9_pipeline as p
import tm7_benchmark as tb
from abundance import log_counts

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = {"version": "v1 (2026-11-10)", "min_samples_with_query": 30,
            "strata": "stratum_hab_study (sub-habitat x study)", "transform": "log2 TSS, zeros -> min/2 per mOTU",
            "score": "t statistic of log2 TSS(H)", "variants": ["unadjusted", "adjusted"]}


def residualise(Y, groups, z):
    """Remove group means (fixed effects) and then the covariate z from every row of Y."""
    Y = Y.astype(np.float64, copy=True)
    for g in np.unique(groups):
        c = groups == g
        Y[:, c] -= Y[:, c].mean(1, keepdims=True)
    zz = z.astype(np.float64).copy()
    for g in np.unique(groups):
        c = groups == g
        zz[c] -= zz[c].mean()
    if (zz ** 2).sum() > 0:
        Y -= np.outer(Y @ zz / (zz ** 2).sum(), zz)
    return Y


def fit_query(LT, i, groups, depth):
    """t statistic of every row of LT as predictor of row i, over the given samples."""
    Yr = residualise(LT[[i]], groups, depth)[0]
    Xr = residualise(LT, groups, depth)
    sxx = (Xr ** 2).sum(1); sxy = Xr @ Yr; syy = (Yr ** 2).sum()
    df = len(Yr) - len(np.unique(groups)) - 2
    b = np.where(sxx > 1e-9, sxy / np.maximum(sxx, 1e-12), 0.0)
    rss = np.maximum(syy - b * sxy, 1e-12)
    se = np.sqrt(rss / max(df, 1) / np.maximum(sxx, 1e-12))
    t = np.where(sxx > 1e-9, b / se, 0.0)
    return t, b, df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)
    t0 = time.time()
    D = tb.DEFS
    p.log(f"host regression {SETTINGS['version']}, benchmark definitions {D['version']}")

    pa, S, M = p.load(a.data)
    M = M.loc[pa.index]; S = S.loc[pa.columns]
    P = pa.values.astype(bool)
    L, _ = log_counts(a.data, pa.index, pa.columns)                 # natural log count, 0 where absent
    reads = p.read_depth(a.data, pa.columns).values.astype(np.float64)
    LT = ((L - np.log(reads)[None, :]) / np.log(2)).astype(np.float32)   # log2 TSS where present
    del L
    for k in range(LT.shape[0]):                                     # MaAsLin2 LOG: zeros -> min/2
        pres = P[k]
        mn = LT[k, pres].min() if pres.any() else 0.0
        LT[k, ~pres] = mn - 1.0
    depth = np.log2(reads)
    strata = (S["stratum_hab_study"] if "stratum_hab_study" in S.columns else S["stratum_study"]).values
    p.log(f"{LT.shape[0]} mOTUs x {LT.shape[1]} samples; strata column: {SETTINGS['strata']}")

    phylum = M["phylum"].values
    tier1 = np.isin(phylum, D["tier1_phylum"]); tier2 = M["family"].isin(D["tier2_families"]).values
    tier3 = M["genus"].isin(D["tier3_genera"]).values
    pates = phylum == "Patescibacteria"
    tm7 = np.where(M["class"].eq("Saccharimonadia").values)[0]
    lp = np.log2(M["prevalence"].values.astype(float)); env = M["dom_env"].values
    pool = ~pates & ~tier1
    ctrl = {}
    for i in tm7:                                                    # same rule as tm7_benchmark.py
        d = np.abs(lp - lp[i])
        c = np.where(pool & (env == env[i]) & (d <= D["max_log2_prev_ratio"]))[0]
        ctrl[i] = c[np.argsort(d[c])[: D["n_controls_per_query"]]]
    rows = np.unique(np.concatenate([tm7] + list(ctrl.values())))
    tmset = set(tm7)

    Q, hosts, lme = [], [], []
    for n_done, r in enumerate(rows):
        cols = P[r]
        if cols.sum() < SETTINGS["min_samples_with_query"]:
            continue
        ok = ~pates & (phylum != phylum[r]); ok[r] = False
        for var in SETTINGS["variants"]:
            g = strata[cols] if var == "adjusted" else np.zeros(cols.sum(), int)
            t, b, df = fit_query(LT[:, cols], r, g, depth[cols])
            Q.append({"variant": var, "motu_id": M.index[r], "role": "TM7" if r in tmset else "control",
                      "dom_env": env[r], "n_samples": int(cols.sum()), "df": int(df),
                      **tb.query_metrics(t, ok, tier1, tier2, tier3)})
            if r in tmset and var == "adjusted":
                order = np.argsort(-np.where(ok, t, -np.inf))[:5]
                for k, h in enumerate(order):
                    hosts.append({"tm7": M.index[r], "tm7_species": M["species"].iat[r], "tm7_family": M["family"].iat[r],
                                  "dom_env": env[r], "rank": k + 1, "partner": M.index[h], "partner_species": M["species"].iat[h],
                                  "partner_family": M["family"].iat[h], "partner_phylum": phylum[h],
                                  "tier1": bool(tier1[h]), "tier2": bool(tier2[h]), "tier3": bool(tier3[h]),
                                  "coef_log2": round(float(b[h]), 4), "t": round(float(t[h]), 2), "n_samples": int(cols.sum()),
                                  "partner_present_share": round(float(P[h, cols].mean()), 3)})
                h = order[0]
                lme.append(pd.DataFrame({"pair": f"{M.index[r]}|{M.index[h]}", "y": LT[r, cols], "x": LT[h, cols],
                                         "log2reads": depth[cols], "habitat": S["habitat_group"].values[cols]
                                         if "habitat_group" in S else S["environment"].values[cols],
                                         "study": S["study_group"].values[cols]}))
        if n_done % 100 == 0:
            p.log(f"  {n_done + 1}/{len(rows)} queries")
    Q = pd.DataFrame(Q)
    Q.to_csv(f"{a.out}/host_regression_queries.tsv", sep="\t", index=False, float_format="%.5g")
    pd.DataFrame(hosts).to_csv(f"{a.out}/host_regression_tm7_hosts.tsv", sep="\t", index=False)
    if lme:
        pd.concat(lme).to_csv(f"{a.out}/host_regression_lme4_input.tsv", sep="\t", index=False, float_format="%.5g")

    # TM7 vs matched controls, same draws as tm7_benchmark.py
    rng = np.random.default_rng(D["seed"])
    have = set(Q.motu_id)
    groups = {"all TM7": tm7, "Human-associated": tm7[env[tm7] == "Human-associated"],
              "Non-saline": tm7[env[tm7] == "Non-saline"]}
    draws = {}
    for i in tm7:
        c = [j for j in ctrl[i] if M.index[j] in have]
        if M.index[i] in have and c:
            draws[i] = rng.choice(c, D["n_draws"])
    out = []
    for var in SETTINGS["variants"]:
        Qv = Q[Q.variant == var].set_index("motu_id")
        for gname, members in groups.items():
            mem = [i for i in members if i in draws]
            if not mem:
                continue
            for met, hb in tb.HIGHER_BETTER.items():
                obs = tb.stat(Qv.loc[M.index[mem], met].values, met)
                null = tb.stat(np.stack([Qv.loc[M.index[draws[i]], met].values for i in mem], 0).T, met)
                better = (null >= obs) if hb else (null <= obs)
                out.append({"variant": var, "group": gname, "n_tm7": len(mem), "metric": met, "tm7": obs,
                            "control_mean": np.nanmean(null), "control_lo": np.nanpercentile(null, 2.5),
                            "control_hi": np.nanpercentile(null, 97.5), "p": (1 + better.sum()) / (1 + len(null))})
    T = pd.DataFrame(out)
    T.to_csv(f"{a.out}/host_regression_summary.tsv", sep="\t", index=False, float_format="%.4g")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(T[T.group == "all TM7"].round(4).to_string(index=False))
    H = pd.DataFrame(hosts)
    head = {"n_tm7_tested": int(len(set(H.tm7))) if len(H) else 0,
            "top1_tier1": float(H[H["rank"] == 1].tier1.mean()) if len(H) else None,
            "top1_tier3": float(H[H["rank"] == 1].tier3.mean()) if len(H) else None}
    with open(f"{a.out}/host_regression.json", "w") as fh:
        json.dump({"settings": SETTINGS, "benchmark_definitions": D["version"], "headline": head,
                   "runtime_min": (time.time() - t0) / 60}, fh, indent=1)
    p.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
