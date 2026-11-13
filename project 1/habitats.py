"""Which sub-habitat does each cluster come from? (mOTUs-db sample metadata)

    python habitats.py [--data data] [--out results] [--min-samples 30]

Needs data/motusdb/Supplementary_Table_4.tsv.gz (mOTUs-db supplementary table,
Zenodo 10.5281/zenodo.13325008, CC-BY-4.0). Each sample gets a finer habitat
(NCBI metagenome term, e.g. human oral / human gut / wastewater / pig gut).

Writes
  samples_habitat.tsv   per sample: environment, study, biosample, ncbi_environment, habitat
  cluster_habitats.tsv  per cluster of every model: the habitat in which its members are
                        most prevalent, that habitat's share of the cluster's prevalence
                        profile, and the runner-up habitat. Prevalence is computed per
                        habitat (share of the habitat's samples containing the mOTU), so
                        habitats with many samples do not win by size alone.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import p9_pipeline as p

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--min-samples", type=int, default=30, help="habitats with fewer samples are not scored")
    a = ap.parse_args(argv)
    prm = json.load(open(f"{a.out}/diagnostics.json"))["params"]
    pa, samples, _ = p.load(a.data, prm["min_prev"], prm["rich_bins"], prm["min_study"])
    if "habitat" not in samples:
        raise SystemExit("data/motusdb/Supplementary_Table_4.tsv.gz not found")
    samples[["environment", "study", "biosample", "ncbi_environment", "habitat"]].to_csv(
        f"{a.out}/samples_habitat.tsv", sep="\t")
    counts = samples.habitat.value_counts()
    habs = counts[counts >= a.min_samples].index
    X = pa.values.astype(np.float32)
    prev = pd.DataFrame({h: X[:, (samples.habitat == h).values].mean(1) for h in habs}, index=pa.index)
    mk = pd.read_csv(f"{a.out}/motus.tsv", sep="\t", index_col=0).loc[pa.index]
    rows = []
    for m in p.MODELS:
        lab = mk[f"cluster_{m}"].fillna(-1).astype(int)
        for c, idx in lab[lab >= 0].groupby(lab[lab >= 0]).groups.items():
            prof = prev.loc[idx].mean()
            share = prof / prof.sum()
            top = share.sort_values(ascending=False)
            rows.append({"model": m, "cluster": int(c), "n_motus": len(idx),
                         "top_habitat": top.index[0], "top_habitat_share": float(top.iloc[0]),
                         "second_habitat": top.index[1], "second_habitat_share": float(top.iloc[1]),
                         "top_habitat_prevalence": float(prof[top.index[0]])})
    H = pd.DataFrame(rows)
    H.to_csv(f"{a.out}/cluster_habitats.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print("habitats scored:", ", ".join(f"{h} ({counts[h]})" for h in habs))
    cl = pd.read_csv(f"{a.out}/clusters_{p.MAIN}.tsv", sep="\t")[["cluster", "top_family", "dominant_env", "top_study", "top_study_share"]]
    print(cl.merge(H[H.model == p.MAIN], on="cluster").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
