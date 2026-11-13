"""KEGG pilot: does the partner chosen by co-occurrence fill the symbiont's metabolic gaps?

    python kegg_pilot.py [--workers 4]

Run after tm7_benchmark.py. Downloads (resumable) the KEGG annotation of each selected mOTU's
representative genome from the mOTUs API (motus-api.microbiomics.io, CC BY 4.0) into
data/motusdb/kegg/, and KEGG module definitions from the KEGG REST API (rest.kegg.jp, academic use).

Design (written before the result):
  Queries   the 41 TM7 (known host-dependent, hosts Actinomycetota) and, per TM7, its 2 nearest
            matched control mOTUs (tm7_benchmark.py rule: same environment, similar prevalence,
            not TM7, not Actinomycetota).
  Partners  per query q: "chosen" = its top 5 partners by main-model association r (benchmark
            candidate set: not Patescibacteria, not q's phylum); "random" = 10 random mOTUs of the
            same dominant environment from that candidate set (seed fixed).
  Modules   a KEGG module counts as present in a genome when >= 75% of its KOs are annotated
            (approximation, the same for every genome). Only biosynthesis modules (name contains
            "biosynthesis") are used: amino acids, cofactors/vitamins, nucleotides, lipids, ...
  Gap fill  fill(q, H) = number of biosynthesis modules present in H and absent in q.
  Statistic d(q) = mean fill over chosen partners - mean fill over random partners.
            d > 0: co-occurrence picks partners that supply what q cannot make itself.
            Comparing each q only with its own random partners removes q's own genome size.
  Tests     TM7: mean d > 0 (sign-flip permutation, one-sided). Specificity: mean d of TM7 vs
            mean d of their matched controls (label permutation). Expected if the signal is a
            dependency signal: TM7 d > 0 and TM7 d > controls d.
Also reported: modules TM7 lack most often and how often chosen Actinomycetota hosts have them.

-> data/motusdb/kegg/<genome>.kegg.gz       raw annotation files (one per genome)
   results/kegg_pilot_modules.tsv.gz        presence of biosynthesis modules per mOTU
   results/kegg_pilot_queries.tsv           per query: d, chosen and random fill
   results/kegg_pilot_summary.tsv           tests
   results/kegg_pilot_tm7_gaps.tsv          modules missing in TM7, filled by their chosen hosts?
   results/kegg_pilot.json
"""
import argparse
import gzip
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import p9_pipeline as p
import tm7_benchmark as tb

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://motus-api.microbiomics.io/v1/genomes/{g}/download?file_type=kegg"
UA = {"User-Agent": "P9-cooccurrence-analysis/1.0 (python urllib)"}
N_CTRL, N_TOP, N_RAND, MOD_FRAC, SEED = 2, 5, 10, 0.8, 3


def get(url, path, tries=4):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                data = r.read()
            with open(path + ".part", "wb") as fh:
                fh.write(data)
            os.replace(path + ".part", path)
            return True
        except Exception as e:
            if "404" in str(e) or "422" in str(e):
                return False
            time.sleep(5 * (k + 1))
    return False


def kos_of(path):
    kos = set()
    with gzip.open(path, "rt", errors="replace") as fh:
        next(fh, None)
        for line in fh:
            f = line.split("\t")
            if len(f) > 1 and f[1].startswith("K"):
                kos.update(x for x in f[1].split(";") if x.startswith("K"))
    return kos


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    t0 = time.time()
    kdir = os.path.join(a.data, "motusdb", "kegg"); os.makedirs(kdir, exist_ok=True)

    # ---- queries and partners
    pa, S, M = p.load(a.data)
    M = M.loc[pa.index]; S = S.loc[pa.columns]
    X = pa.values.astype(np.float32)
    D = tb.DEFS
    phylum = M["phylum"].values; env = M["dom_env"].values
    lp = np.log2(M["prevalence"].values.astype(float))
    pates = phylum == "Patescibacteria"; tier1 = np.isin(phylum, D["tier1_phylum"])
    tm7 = np.where(M["class"].eq("Saccharimonadia").values)[0]
    pool = ~pates & ~tier1
    ctrl = {}
    for i in tm7:
        d = np.abs(lp - lp[i]); c = np.where(pool & (env == env[i]) & (d <= D["max_log2_prev_ratio"]))[0]
        ctrl[i] = list(c[np.argsort(d[c])[:N_CTRL]])
    queries = list(tm7) + sorted({c for v in ctrl.values() for c in v})
    R = tb.r_rows(X, np.array(queries), S[p.STRATA_COL[p.MAIN]].values)
    rng = np.random.default_rng(SEED)
    plan = []
    for k, q in enumerate(queries):
        ok = ~pates & (phylum != phylum[q]); ok[q] = False
        top = np.argsort(-np.where(ok, R[k], -np.inf))[:N_TOP]
        same = np.where(ok & (env == env[q]))[0]
        rnd = rng.choice(same, min(N_RAND, len(same)), replace=False)
        plan.append({"q": q, "role": "TM7" if q in set(tm7) else "control", "chosen": list(top), "random": list(rnd)})
    need = sorted({q for q in queries} | {h for r in plan for h in r["chosen"] + r["random"]})
    GQ = pd.read_csv(f"{a.data}/motusdb/genome_quality.tsv", sep="\t", index_col="motu_id")["rep_genome"]
    rep = GQ.reindex(M.index[need]).dropna()
    p.log(f"{len(queries)} queries ({len(tm7)} TM7), {len(need)} mOTUs, {len(rep)} with a representative genome")

    # ---- downloads
    todo = [(m, g) for m, g in rep.items() if not os.path.exists(os.path.join(kdir, f"{g}.kegg.gz"))]
    p.log(f"downloading {len(todo)} KEGG files ({len(rep) - len(todo)} already present)")
    fails = []
    with ThreadPoolExecutor(a.workers) as ex:
        fut = {ex.submit(get, API.format(g=g), os.path.join(kdir, f"{g}.kegg.gz")): m for m, g in todo}
        for n, f in enumerate(as_completed(fut), 1):
            if not f.result():
                fails.append(fut[f])
            if n % 100 == 0:
                p.log(f"  {n}/{len(todo)}")
    mdef = os.path.join(kdir, "_kegg_module_ko.tsv"); mlist = os.path.join(kdir, "_kegg_module_list.tsv")
    get("https://rest.kegg.jp/link/ko/module", mdef); get("https://rest.kegg.jp/list/module", mlist)

    # ---- module presence, from module DEFINITIONS (steps with alternatives)
    # First pilot run (2026-11-11) used ">= 75% of all KOs linked to the module", which is wrong:
    # KEGG links every alternative enzyme to a module, so almost no genome reached 75%.
    # Now: a module is a sequence of steps; a step is done if any KO option in it is present
    # (optional '-K' parts ignored); the module counts as present if >= MOD_FRAC of steps are done.
    names = pd.read_csv(mlist, sep="\t", header=None, names=["module", "name"]).set_index("module").name
    names.index = names.index.str.replace("md:", "", regex=False)
    bios = [m for m in names.index if "biosynthesis" in names[m].lower()]
    ddir = os.path.join(kdir, "_module_def"); os.makedirs(ddir, exist_ok=True)
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda m: get(f"https://rest.kegg.jp/get/{m}", os.path.join(ddir, m + ".txt")), bios))
    import re

    def steps_of(defn):
        out, depth, cur = [], 0, ""
        for ch in defn:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == " " and depth == 0:
                if cur.strip():
                    out.append(cur)
                cur = ""
            else:
                cur += ch
        if cur.strip():
            out.append(cur)
        res = []
        for s in out:
            s = re.sub(r"-\(?[^ ,()]*\)?", "", s) if not s.startswith("-") else ""
            k = set(re.findall(r"K\d{5}", s))
            if k:
                res.append(k)
        return res
    mk = {}
    for m in bios:
        fp = os.path.join(ddir, m + ".txt")
        if not os.path.exists(fp):
            continue
        txt = open(fp, encoding="utf-8", errors="replace").read()
        dl = re.search(r"^DEFINITION\s+(.+?)\n(?=[A-Z])", txt, re.S | re.M)
        if dl:
            st = steps_of(" ".join(dl.group(1).split()))
            if len(st) >= 2:
                mk[m] = st
    rows = {}
    for m_id, g in rep.items():
        path = os.path.join(kdir, f"{g}.kegg.gz")
        if os.path.exists(path):
            ks = kos_of(path)
            rows[m_id] = {m: np.mean([bool(ks & s) for s in st]) >= MOD_FRAC for m, st in mk.items()} | {"_n_ko": len(ks)}
    P = pd.DataFrame.from_dict(rows, orient="index")
    P.to_csv(f"{a.out}/kegg_pilot_modules.tsv.gz", sep="\t")
    mods = [m for m in P.columns if not m.startswith("_")]
    B = P[mods].astype(bool)
    p.log(f"{len(P)} genomes parsed, {len(mods)} biosynthesis modules")

    # ---- gap filling
    ids = M.index.values
    out = []
    for r in plan:
        qid = ids[r["q"]]
        if qid not in B.index:
            continue
        lack = ~B.loc[qid].values
        fill = lambda hs: [int((B.loc[ids[h]].values & lack).sum()) for h in hs if ids[h] in B.index]
        fc, fr = fill(r["chosen"]), fill(r["random"])
        if not fc or not fr:
            continue
        # partner-size adjustment: residual of fill after a linear fit on the partner's module count,
        # fitted over all 15 partners of q, so bigger genomes do not win just by having more pathways
        hs = [h for h in r["chosen"] + r["random"] if ids[h] in B.index]
        size = np.array([B.loc[ids[h]].sum() for h in hs], float)
        f_all = np.array(fill(hs), float)
        coef = np.polyfit(size, f_all, 1) if np.ptp(size) > 0 else np.array([0.0, f_all.mean()])
        res = f_all - np.polyval(coef, size)
        is_ch = np.array([h in set(r["chosen"]) for h in hs])
        out.append({"motu_id": qid, "role": r["role"], "species": M.species.iat[r["q"]], "dom_env": env[r["q"]],
                    "n_ko": int(P.loc[qid, "_n_ko"]), "modules_present": int(B.loc[qid].sum()),
                    "fill_chosen": float(np.mean(fc)), "fill_random": float(np.mean(fr)),
                    "d": float(np.mean(fc) - np.mean(fr)),
                    "d_adj": float(res[is_ch].mean() - res[~is_ch].mean()) if is_ch.any() and (~is_ch).any() else np.nan,
                    "size_chosen": float(size[is_ch].mean()), "size_random": float(size[~is_ch].mean()),
                    "n_chosen": len(fc), "n_random": len(fr)})
    Qd = pd.DataFrame(out)
    Qd.to_csv(f"{a.out}/kegg_pilot_queries.tsv", sep="\t", index=False, float_format="%.4g")
    rng = np.random.default_rng(SEED)
    tests = []
    for col in ["d", "d_adj"]:
        for role in ["TM7", "control"]:
            d = Qd[Qd.role == role][col].dropna().values
            null = np.array([(d * rng.choice([-1, 1], len(d))).mean() for _ in range(10000)])
            tests.append({"test": f"{role}: chosen fill more gaps than random ({col} > 0)", "n": len(d), "value": d.mean(),
                          "p": (1 + (null >= d.mean()).sum()) / (1 + len(null))})
        dt, dc = Qd[Qd.role == "TM7"][col].dropna().values, Qd[Qd.role == "control"][col].dropna().values
        allv = np.r_[dt, dc]; obs = dt.mean() - dc.mean()
        null = np.array([(lambda z: z[:len(dt)].mean() - z[len(dt):].mean())(rng.permutation(allv)) for _ in range(10000)])
        tests.append({"test": f"TM7 {col} minus control {col}", "n": len(allv), "value": obs, "p": (1 + (null >= obs).sum()) / (1 + len(null))})
    T = pd.DataFrame(tests); T.to_csv(f"{a.out}/kegg_pilot_summary.tsv", sep="\t", index=False, float_format="%.4g")
    print(T.round(4).to_string(index=False))
    print(Qd.groupby("role")[["n_ko", "modules_present", "size_chosen", "size_random", "fill_chosen", "fill_random", "d", "d_adj"]].median().round(2).to_string())

    # ---- which gaps: modules most TM7 lack, filled by chosen Actinomycetota partners?
    tq = [r for r in plan if r["role"] == "TM7" and ids[r["q"]] in B.index]
    gaps = []
    for m in mods:
        lacking = [r for r in tq if not B.loc[ids[r["q"]], m]]
        ch = [B.loc[ids[h], m] for r in lacking for h in r["chosen"] if ids[h] in B.index]
        ra = [B.loc[ids[h], m] for r in lacking for h in r["random"] if ids[h] in B.index]
        if lacking and ch and ra:
            gaps.append({"module": m, "name": names.get(m, ""), "tm7_lacking": len(lacking),
                         "chosen_have": float(np.mean(ch)), "random_have": float(np.mean(ra))})
    G = pd.DataFrame(gaps).sort_values("tm7_lacking", ascending=False)
    G["chosen_minus_random"] = G.chosen_have - G.random_have
    G.to_csv(f"{a.out}/kegg_pilot_tm7_gaps.tsv", sep="\t", index=False, float_format="%.3g")
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(G.sort_values("chosen_minus_random", ascending=False).head(15).round(2).to_string(index=False))
    with open(f"{a.out}/kegg_pilot.json", "w") as fh:
        json.dump({"n_queries": len(Qd), "n_genomes": int(len(P)), "n_modules": len(mods), "download_failures": fails,
                   "settings": {"n_ctrl": N_CTRL, "n_top": N_TOP, "n_random": N_RAND, "module_frac": MOD_FRAC, "seed": SEED},
                   "runtime_min": (time.time() - t0) / 60}, fh, indent=1)
    p.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
