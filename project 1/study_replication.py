"""Study-level replication: do associations hold up in independent studies?

    python study_replication.py

Run after p9_pipeline.py and tm7_benchmark.py.

edge_stability.py splits the samples at random. Both halves then share the same studies,
protocols and sites, so a study-specific artefact replicates by construction. Here every study
with >= MIN_SAMPLES samples is analysed on its own (strata = sub-habitat x richness quartile
inside that study, the main model's strata restricted to one study), and a pair counts as
replicated in a study when its within-study CMH z >= Z_STUDY (one-sided 5%) with r > 0.
"Testable" in a study = both mOTUs detected >= MIN_DET times there.

Reported:
  1 main-model network edges: share positive in >= 2 independent studies, among edges testable
    in >= 2, against random same-environment pairs (baseline) and against the random-half
    replication of edge_stability.py;
  2 the same per environment;
  3 TM7 benchmark: partners ranked by the number of studies in which they replicate (ties
    broken by the main-model r), scored with the frozen benchmark (tm7_benchmark.DEFS) against
    the same matched controls.

-> results/study_replication_edges.tsv.gz   per network edge: n_testable, n_pos, n_neg
   results/study_replication_summary.tsv    shares per group, edges vs baseline
   results/study_replication_tm7.tsv        TM7 vs matched controls for the study-replication ranking
   results/study_replication.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import p9_pipeline as p
import tm7_benchmark as tb

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_SAMPLES, MIN_DET, Z_STUDY = 30, 3, 1.645


def studies_of(S):
    vc = S["study"].value_counts()
    return list(vc[vc >= MIN_SAMPLES].index)


def per_study_counts(X, S, a, b, label_col="stratum_hab_study_rich", log_every=0):
    """For pairs (a[k], b[k]) of row indices into X (bool mOTU x sample): number of studies in
    which the pair is testable, significantly positive, significantly negative."""
    a = np.asarray(a); b = np.asarray(b)
    n_t = np.zeros(len(a), np.int16); n_p = np.zeros(len(a), np.int16); n_n = np.zeros(len(a), np.int16)
    labels_all = S[label_col].values if label_col in S else S["stratum_study_rich"].values
    for k, st in enumerate(studies_of(S)):
        cols = np.where(S["study"].values == st)[0]
        det = X[:, cols].sum(1)
        live = det >= MIN_DET
        sel = np.where(live[a] & live[b])[0]
        if not len(sel):
            continue
        idx = np.where(live)[0]
        loc = np.full(X.shape[0], -1); loc[idx] = np.arange(len(idx))
        Xs = X[np.ix_(idx, cols)].astype(np.float32)
        lab = labels_all[cols]
        strata = pd.unique(lab)
        N = np.stack([Xs[:, lab == s].sum(1) for s in strata], 1).astype(np.float64)
        Ns = np.array([(lab == s).sum() for s in strata], float)
        A = N * (Ns - N) / (Ns * np.sqrt(np.maximum(Ns - 1, 1)))
        for c0 in range(0, len(sel), 20000):                       # chunks keep memory small
            sc = sel[c0:c0 + 20000]
            ia, ib = loc[a[sc]], loc[b[sc]]
            O = np.einsum("ij,ij->i", Xs[ia], Xs[ib]).astype(np.float64)
            E = np.einsum("ij,ij->i", N[ia] / Ns, N[ib])
            V = np.einsum("ij,ij->i", A[ia], A[ib])
            z = (O - E) / np.sqrt(np.maximum(V, 1e-12))
            ok = V > 1e-9
            n_t[sc[ok]] += 1
            n_p[sc[ok & (z >= Z_STUDY)]] += 1
            n_n[sc[ok & (z <= -Z_STUDY)]] += 1
        if log_every and (k + 1) % log_every == 0:
            p.log(f"    {k + 1} studies done")
    return n_t, n_p, n_n


def per_study_rows(X, S, rows, label_col="stratum_hab_study_rich"):
    """Like per_study_counts, for every pair (rows[k], j) with j over all mOTUs: returns the
    number of studies in which each such pair is significantly positive (len(rows) x n_motus)."""
    rows = np.asarray(rows)
    n_p = np.zeros((len(rows), X.shape[0]), np.int16)
    labels_all = S[label_col].values if label_col in S else S["stratum_study_rich"].values
    for st in studies_of(S):
        cols = np.where(S["study"].values == st)[0]
        Xs = X[:, cols].astype(np.float32)
        live = Xs.sum(1) >= MIN_DET
        lab = labels_all[cols]
        strata = pd.unique(lab)
        N = np.stack([Xs[:, lab == s].sum(1) for s in strata], 1).astype(np.float64)
        Ns = np.array([(lab == s).sum() for s in strata], float)
        A = N * (Ns - N) / (Ns * np.sqrt(np.maximum(Ns - 1, 1)))
        O = (Xs[rows] @ Xs.T).astype(np.float64)
        E = (N[rows] / Ns) @ N.T
        V = A[rows] @ A.T
        z = (O - E) / np.sqrt(np.maximum(V, 1e-12))
        okm = (V > 1e-9) & live[rows][:, None] & live[None, :]
        n_p += (okm & (z >= Z_STUDY)).astype(np.int16)
    return n_p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--n-random", type=int, default=300000)
    a = ap.parse_args(argv)
    t0 = time.time()
    pa, S, M = p.load(a.data)
    M = M.loc[pa.index]; S = S.loc[pa.columns]
    X = pa.values.astype(bool)
    pos = pd.Series(np.arange(len(pa)), index=pa.index)
    st = studies_of(S)
    p.log(f"{len(st)} studies with >= {MIN_SAMPLES} samples ({S.study.isin(st).sum()} samples)")

    # 1 network edges of the main model and a same-environment random baseline
    Ed = pd.read_csv(f"{a.out}/network_edges_{p.MAIN}.tsv.gz", sep="\t")
    ea, eb = pos.reindex(Ed["a"]).values.astype(int), pos.reindex(Ed["b"]).values.astype(int)
    rng = np.random.default_rng(1)
    env = M["dom_env"].values
    ra = rng.integers(0, len(pa), a.n_random * 2); rb = rng.integers(0, len(pa), a.n_random * 2)
    keep = (ra != rb) & (env[ra] == env[rb]); ra, rb = ra[keep][: a.n_random], rb[keep][: a.n_random]
    p.log(f"{len(ea)} network edges, {len(ra)} random same-environment pairs")
    n_t, n_p, n_n = per_study_counts(X, S, np.r_[ea, ra], np.r_[eb, rb], log_every=16)
    ne = len(ea)
    Ed["n_testable"], Ed["n_pos"], Ed["n_neg"] = n_t[:ne], n_p[:ne], n_n[:ne]
    Ed.to_csv(f"{a.out}/study_replication_edges.tsv.gz", sep="\t", index=False)

    def shares(t, pz, nz, mask=None):
        m = (t >= 2) if mask is None else (t >= 2) & mask
        return {"n": int(m.sum()), "pos_ge2": float((pz[m] >= 2).mean()) if m.any() else np.nan,
                "pos_ge_half": float((pz[m] >= np.ceil(t[m] / 2)).mean()) if m.any() else np.nan,
                "any_neg": float((nz[m] >= 1).mean()) if m.any() else np.nan,
                "median_testable": float(np.median(t[m])) if m.any() else np.nan}
    rows = [{"set": "network edges", "group": "all", **shares(n_t[:ne], n_p[:ne], n_n[:ne])},
            {"set": "random same-env pairs", "group": "all", **shares(n_t[ne:], n_p[ne:], n_n[ne:])}]
    for e in p.ENVS:
        rows.append({"set": "network edges", "group": e, **shares(n_t[:ne], n_p[:ne], n_n[:ne], env[ea] == e)})
        rows.append({"set": "random same-env pairs", "group": e, **shares(n_t[ne:], n_p[ne:], n_n[ne:], env[ra] == e)})
    T = pd.DataFrame(rows)
    T.to_csv(f"{a.out}/study_replication_summary.tsv", sep="\t", index=False, float_format="%.4g")
    print(T.round(3).to_string(index=False))

    # 3 TM7 benchmark with the study-replication ranking
    D = tb.DEFS
    phylum = M["phylum"].values
    tier1 = np.isin(phylum, D["tier1_phylum"]); tier2 = M["family"].isin(D["tier2_families"]).values
    tier3 = M["genus"].isin(D["tier3_genera"]).values; pates = phylum == "Patescibacteria"
    tm7 = np.where(M["class"].eq("Saccharimonadia").values)[0]
    lp = np.log2(M["prevalence"].values.astype(float)); pool = ~pates & ~tier1
    ctrl = {}
    for i in tm7:
        d = np.abs(lp - lp[i])
        c = np.where(pool & (env == env[i]) & (d <= D["max_log2_prev_ratio"]))[0]
        ctrl[i] = c[np.argsort(d[c])[: D["n_controls_per_query"]]]
    q = np.unique(np.concatenate([tm7] + list(ctrl.values())))
    Rm = tb.r_rows(X.astype(np.float32), q, S[p.STRATA_COL[p.MAIN]].values)
    p.log(f"TM7 benchmark: {len(q)} queries x {len(pa)} partners")
    qp = per_study_rows(X, S, q).astype(float)
    out, Q = [], []
    for k, r in enumerate(q):
        score = qp[k] + 0.5 * (np.clip(Rm[k], -1, 1) + 1) / 2            # studies first, r breaks ties
        ok = ~pates & (phylum != phylum[r]); ok[r] = False
        Q.append({"motu_id": M.index[r], **tb.query_metrics(score, ok, tier1, tier2, tier3)})
    Q = pd.DataFrame(Q).set_index("motu_id")
    rng = np.random.default_rng(D["seed"])
    draws = {i: rng.choice(ctrl[i], D["n_draws"]) for i in tm7 if len(ctrl[i])}
    for gname, members in {"all TM7": tm7, "Human-associated": tm7[env[tm7] == "Human-associated"],
                           "Non-saline": tm7[env[tm7] == "Non-saline"]}.items():
        mem = [i for i in members if i in draws]
        for met, hb in tb.HIGHER_BETTER.items():
            obs = tb.stat(Q.loc[M.index[mem], met].values, met)
            null = tb.stat(np.stack([Q.loc[M.index[draws[i]], met].values for i in mem], 0).T, met)
            better = (null >= obs) if hb else (null <= obs)
            out.append({"ranking": "replicating studies", "group": gname, "n_tm7": len(mem), "metric": met,
                        "tm7": obs, "control_mean": np.nanmean(null), "control_lo": np.nanpercentile(null, 2.5),
                        "control_hi": np.nanpercentile(null, 97.5), "p": (1 + better.sum()) / (1 + len(null))})
    B = pd.DataFrame(out)
    B.to_csv(f"{a.out}/study_replication_tm7.tsv", sep="\t", index=False, float_format="%.4g")
    print(B[B.group == "all TM7"].round(4).to_string(index=False))
    stab = pd.read_csv(f"{a.out}/edge_stability.tsv", sep="\t") if os.path.exists(f"{a.out}/edge_stability.tsv") else None
    half = None
    if stab is not None:
        r_ = stab[(stab.model == p.MAIN) & (stab["class"] == "positive")]
        half = float(r_.replication.iat[0]) if len(r_) else None
    with open(f"{a.out}/study_replication.json", "w") as fh:
        json.dump({"min_samples": MIN_SAMPLES, "min_det": MIN_DET, "z_study": Z_STUDY, "n_studies": len(st),
                   "random_half_replication_main_positive": half, "runtime_min": (time.time() - t0) / 60}, fh, indent=1)
    p.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
