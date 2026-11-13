"""TM7 benchmark: do the null models recover the known hosts of Saccharibacteria?

    python tm7_benchmark.py

Run after p9_pipeline.py (needs nothing else from results/; recomputes the
pooled phi r of the five null models for the query mOTUs only).

Saccharibacteria (TM7) are obligate epibionts. Every cultured TM7-host pair has an
Actinomycetota host. That makes them the one place in this data set where a
biological interaction is known in advance, so they are used here as a benchmark
for the null models, not as a discovery screen.

The definitions below are FROZEN (version DEFS["version"]). Do not tune them on
the result. Any change gets a new version string, and both versions are reported.

Three levels of evidence, reported separately (never pooled into one number):
  tier 1  broad enrichment     partner phylum is Actinomycetota
  tier 2  known host families  Actinomycetaceae, Propionibacteriaceae
  tier 3  documented host genera (GTDB names): Pauljensenia (= Schaalia/Actinomyces
          odontolytica and meyeri), Actinomyces, Arachnia (= Pseudopropionibacterium
          propionicum), Gordonia (G. amarae). Literature list to be checked against
          the primary co-culture papers before publication.

Validation (matched null): every TM7 is paired with control mOTUs that are not
TM7 and not Actinomycetota, from the same dominant environment and with similar
prevalence. Controls are scored exactly like TM7. If a null model raises the
Actinomycetota share among top partners for ANY mOTU of that habitat and
prevalence, the controls show it too, and the TM7 result is not specific.
p = share of 'one random control per TM7' draws that do at least as well as TM7.

-> results/tm7_benchmark_queries.tsv   one row per query mOTU (TM7 and controls) x model
   results/tm7_benchmark_summary.tsv   per model x group x metric: TM7, control mean,
                                       control 95% range, p
   results/tm7_benchmark.json          frozen definitions (+ hash) and headline numbers
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

import p9_pipeline as p

HERE = os.path.dirname(os.path.abspath(__file__))

DEFS = {
    "version": "v1 (frozen 2026-11-10)",
    "query": "kept mOTUs of class Saccharimonadia",
    "candidates": "kept mOTUs, not Patescibacteria, not the query's own phylum",
    "score": "pooled phi r of the null model (p9_pipeline.NullModel definition)",
    "models": ["global", "env", "env_rich", "env_study_rich", "hab_study_rich"],
    "tier1_phylum": ["Actinomycetota"],
    "tier2_families": ["Actinomycetaceae", "Propionibacteriaceae"],
    "tier3_genera": ["Pauljensenia", "Actinomyces", "Arachnia", "Gordonia"],
    "controls": "same dom_env, not Patescibacteria, not Actinomycetota, |log2 prevalence ratio| <= 0.5, "
                "25 nearest in prevalence",
    "n_controls_per_query": 25,
    "max_log2_prev_ratio": 0.5,
    "n_draws": 5000,
    "seed": 1,
    "metrics": {
        "top1_tier1": "share of queries whose top partner is tier 1 (higher = better)",
        "top5_tier1": "mean share of tier 1 among the top 5 partners (higher = better)",
        "best_rank_tier2": "median percentile rank of the best tier 2 partner (lower = better)",
        "best_rank_tier3": "median percentile rank of the best tier 3 partner (lower = better)",
        "top10_tier3": "share of queries with a tier 3 genus among the top 10 partners (higher = better)",
    },
}
HIGHER_BETTER = {"top1_tier1": True, "top5_tier1": True, "best_rank_tier2": False,
                 "best_rank_tier3": False, "top10_tier3": True}


def r_rows(X, rows, labels):
    """Pooled phi r of X[rows] against all rows of X, with strata = labels."""
    labels = np.asarray(labels)
    strata = pd.unique(labels)
    N = np.stack([X[:, labels == s].sum(1) for s in strata], 1).astype(np.float64)   # mOTU x stratum
    Ns = np.array([(labels == s).sum() for s in strata], float)
    O = (X[rows] @ X.T).astype(np.float64)
    E = (N[rows] / Ns) @ N.T
    s = (N * (Ns - N) / Ns).sum(1)
    return (O - E) / np.sqrt(np.maximum(np.outer(s[rows], s), 1e-12))


def query_metrics(score, ok, tier1, tier2, tier3):
    sc = np.where(ok, score, -np.inf)
    order = np.argsort(-sc)[: ok.sum()]
    n = len(order)
    t1, t2, t3 = tier1[order], tier2[order], tier3[order]
    best = lambda t: (np.argmax(t) + 1) / n if t.any() else np.nan
    return {"top1_tier1": float(t1[0]), "top5_tier1": float(t1[:5].mean()),
            "best_rank_tier2": best(t2), "best_rank_tier3": best(t3),
            "top10_tier3": float(t3[:10].any()), "n_candidates": n}


def stat(values, metric):
    v = np.asarray(values, float)
    return np.nanmedian(v, axis=-1) if metric.startswith("best_rank") else np.nanmean(v, axis=-1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)
    t0 = time.time()
    defs_hash = hashlib.sha256(json.dumps(DEFS, sort_keys=True).encode()).hexdigest()[:12]
    p.log(f"TM7 benchmark {DEFS['version']} (definitions hash {defs_hash})")

    pa, S, M = p.load(a.data)
    M = M.loc[pa.index]; S = S.loc[pa.columns]
    X = pa.values.astype(np.float32)
    phylum = M["phylum"].values
    tier1 = np.isin(phylum, DEFS["tier1_phylum"])
    tier2 = M["family"].isin(DEFS["tier2_families"]).values
    tier3 = M["genus"].isin(DEFS["tier3_genera"]).values
    pates = phylum == "Patescibacteria"
    tm7 = np.where(M["class"].eq("Saccharimonadia").values)[0]
    p.log(f"{len(tm7)} TM7 queries; tier 1/2/3 partners: {tier1.sum()}/{tier2.sum()}/{tier3.sum()}")

    # ---- matched controls
    lp = np.log2(M["prevalence"].values.astype(float))
    env = M["dom_env"].values
    pool = ~pates & ~tier1
    ctrl = {}
    for i in tm7:
        d = np.abs(lp - lp[i])
        c = np.where(pool & (env == env[i]) & (d <= DEFS["max_log2_prev_ratio"]))[0]
        ctrl[i] = c[np.argsort(d[c])[: DEFS["n_controls_per_query"]]]
    n_ctrl = np.array([len(c) for c in ctrl.values()])
    p.log(f"controls per TM7: min {n_ctrl.min()}, median {int(np.median(n_ctrl))}")
    rows = np.unique(np.concatenate([tm7] + list(ctrl.values())))
    pos = {r: k for k, r in enumerate(rows)}

    # ---- score every query under every model
    labels = {"global": np.zeros(X.shape[1], int)}
    labels |= {m: S[c].values for m, c in p.STRATA_COL.items() if c in S}
    Q = []
    for m in DEFS["models"]:
        if m not in labels:
            p.log(f"  {m}: strata not available, skipped"); continue
        R = r_rows(X, rows, labels[m])
        for r in rows:
            ok = ~pates & (phylum != phylum[r]); ok[r] = False
            Q.append({"model": m, "motu_id": M.index[r], "role": "TM7" if r in set(tm7) else "control",
                      "phylum": phylum[r], "family": M["family"].iat[r], "dom_env": env[r],
                      "prevalence": int(M["prevalence"].iat[r]),
                      **query_metrics(R[pos[r]], ok, tier1, tier2, tier3)})
        p.log(f"  {m}: scored {len(rows)} queries")
    Q = pd.DataFrame(Q)
    Q.to_csv(f"{a.out}/tm7_benchmark_queries.tsv", sep="\t", index=False, float_format="%.5g")

    # ---- TM7 vs matched-control draws
    rng = np.random.default_rng(DEFS["seed"])
    groups = {"all TM7": tm7}
    for e in ["Human-associated", "Non-saline"]:
        groups[e] = np.array([i for i in tm7 if env[i] == e])
    draws = {i: rng.choice(ctrl[i], DEFS["n_draws"]) for i in tm7 if len(ctrl[i])}
    out = []
    for m in Q["model"].unique():
        Qm = Q[Q.model == m].set_index("motu_id")
        for g, members in groups.items():
            members = [i for i in members if i in draws]
            if not members:
                continue
            for met, hb in HIGHER_BETTER.items():
                obs = stat(Qm.loc[M.index[members], met].values, met)
                null = stat(np.stack([Qm.loc[M.index[draws[i]], met].values for i in members], 0).T, met)
                better = (null >= obs) if hb else (null <= obs)
                out.append({"model": m, "group": g, "n_tm7": len(members), "metric": met, "tm7": obs,
                            "control_mean": np.nanmean(null), "control_lo": np.nanpercentile(null, 2.5),
                            "control_hi": np.nanpercentile(null, 97.5),
                            "p": (1 + np.sum(better)) / (1 + len(null))})
    # change from the uncorrected to the main model, TM7 vs controls (difference in differences)
    Qi = Q.pivot_table(index="motu_id", columns="model", values="top1_tier1")
    if {"global", p.MAIN} <= set(Qi.columns):
        dlt = (Qi[p.MAIN] - Qi["global"])
        obs = dlt.loc[M.index[tm7]].mean()
        null = np.stack([dlt.loc[M.index[draws[i]]].values for i in tm7 if i in draws], 0).mean(0)
        out.append({"model": f"{p.MAIN} minus global", "group": "all TM7", "n_tm7": len(draws),
                    "metric": "top1_tier1", "tm7": obs, "control_mean": null.mean(),
                    "control_lo": np.percentile(null, 2.5), "control_hi": np.percentile(null, 97.5),
                    "p": (1 + np.sum(null >= obs)) / (1 + len(null))})
    T = pd.DataFrame(out)
    T.to_csv(f"{a.out}/tm7_benchmark_summary.tsv", sep="\t", index=False, float_format="%.4g")

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(T[T.group == "all TM7"].round(4).to_string(index=False))
    head = T[(T.group == "all TM7")].set_index(["model", "metric"])[["tm7", "control_mean", "p"]]
    with open(f"{a.out}/tm7_benchmark.json", "w") as fh:
        json.dump({"definitions": DEFS, "definitions_hash": defs_hash, "n_tm7": int(len(tm7)),
                   "controls_per_tm7_median": int(np.median(n_ctrl)),
                   "headline": {f"{k[0]} | {k[1]}": v for k, v in head.round(5).to_dict("index").items()},
                   "runtime_min": (time.time() - t0) / 60}, fh, indent=1)
    p.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
