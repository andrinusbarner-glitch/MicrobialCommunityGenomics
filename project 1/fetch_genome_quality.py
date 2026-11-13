"""Download genome quality for every analysed mOTU from the mOTUs online database.

    python fetch_genome_quality.py [--out data/motusdb/genome_quality.tsv] [--workers 4]

motus-db.org has no bulk file for genome metadata, so this uses the genome table's
own CSV export (the "Export Metadata > Selection > CSV" button), one request per
mOTU, asking only for the quality columns. Each response is reduced to one summary
row per mOTU; raw rows are not kept. Already fetched mOTUs are skipped, so an
interrupted run can simply be restarted.

Columns written (one row per mOTU):
  n_genomes, n_mags, n_isolates     all genomes assigned to the mOTU
  n_mq, n_hq                        medium (>= 50% complete, <= 10% contaminated) and
                                    high quality (>= 90%, <= 5%) genomes (MIMAG)
  size_hq_median                    median assembled size of high-quality genomes (bp)
  est_size_median                   median estimated complete genome size over
                                    medium-quality genomes:
                                    size * (1 - contamination) / completeness
  completeness_max, gc_median
  rep_genome, rep_is_mag, rep_size, rep_completeness, rep_contamination

Data: Dmitrijeva et al., "The mOTUs online database provides web-accessible genomic
context to taxonomic profiling of microbial communities", NAR 2025 (PubMed 39526369);
https://motus-db.org, CC-BY-4.0.
"""
import argparse
import csv
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://motus-db.org/genome-cols/csv/downloads"
COLS = ["genome", "motus4", "is_rep", "is_mag", "completeness", "contamination", "genome_size", "gc_content"]
FIELDS = ["motu_id", "n_genomes", "n_mags", "n_isolates", "n_mq", "n_hq", "size_hq_median", "est_size_median",
          "completeness_max", "gc_median", "rep_genome", "rep_is_mag", "rep_size", "rep_completeness",
          "rep_contamination"]


def fetch(motu, tries=4):
    cols = [{"name": c, "data": c, "title": c} for c in COLS]
    body = {"repo_id": "motus", "ids": "", "invert": True, "global_search_value": "",
            "search_value_pair": [{"key": "motus4", "value": motu, "regex": False}],
            "view_state": {"visibleColumns": cols, "order": [], "orderByNames": [], "globalSearch": "",
                           "perColumnSearch": [{"name": "motus4", "value": motu}]}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "text/csv",
                                          "User-Agent": "P9-cooccurrence-analysis/1.0 (python urllib)"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def summarise(motu, text):
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r.get("motus4") == motu]
    out = dict.fromkeys(FIELDS, "")
    out["motu_id"] = motu
    out["n_genomes"] = len(rows)
    if not rows:
        return out
    comp = np.array([num(r["completeness"]) for r in rows])
    cont = np.array([num(r["contamination"]) for r in rows])
    size = np.array([num(r["genome_size"]) for r in rows])
    gc = np.array([num(r["gc_content"]) for r in rows])
    mag = np.array([str(r["is_mag"]).upper() == "TRUE" for r in rows])
    mq = (comp >= 50) & (cont <= 10)
    hq = (comp >= 90) & (cont <= 5)
    out.update(n_mags=int(mag.sum()), n_isolates=int((~mag).sum()), n_mq=int(mq.sum()), n_hq=int(hq.sum()),
               size_hq_median=float(np.nanmedian(size[hq])) if hq.any() else "",
               est_size_median=float(np.nanmedian(size[mq] * (1 - cont[mq] / 100) / (comp[mq] / 100))) if mq.any() else "",
               completeness_max=float(np.nanmax(comp)) if np.isfinite(comp).any() else "",
               gc_median=float(np.nanmedian(gc)) if np.isfinite(gc).any() else "")
    rep = [i for i, r in enumerate(rows) if str(r["is_rep"]).lower() == "true"]
    if rep:
        i = rep[0]
        out.update(rep_genome=rows[i]["genome"], rep_is_mag=bool(mag[i]), rep_size=size[i],
                   rep_completeness=comp[i], rep_contamination=cont[i])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--motus", default=os.path.join(HERE, "results", "motus.tsv"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "motusdb", "genome_quality.tsv"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--all", action="store_true", help="also mOTUs removed by the prevalence filter")
    a = ap.parse_args(argv)
    mk = pd.read_csv(a.motus, sep="\t", index_col=0)
    todo = list(mk.index if a.all else mk.index[mk.kept])
    done = set()
    if os.path.exists(a.out):
        done = set(pd.read_csv(a.out, sep="\t", usecols=["motu_id"]).motu_id)
    todo = [m for m in todo if m not in done]
    print(f"{len(done)} mOTUs already fetched, {len(todo)} to go", flush=True)
    new = not os.path.exists(a.out)
    t0 = time.time()
    with open(a.out, "a", newline="", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(fh, FIELDS, delimiter="\t")
        if new:
            w.writeheader()
        futs = {ex.submit(fetch, m): m for m in todo}
        for k, f in enumerate(as_completed(futs), 1):
            m = futs[f]
            try:
                w.writerow(summarise(m, f.result()))
            except Exception as e:                       # left out, so a restart retries it
                print(f"failed {m}: {e}", flush=True)
                continue
            if k % 200 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"{k}/{len(todo)}  {el / 60:.1f} min, ~{el / k * (len(todo) - k) / 60:.0f} min left", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
