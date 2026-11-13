"""Build genome-scale metabolic models (CarveMe) and score the pairs (SMETANA).

    python gem_build.py --env <venv with carveme+smetana> --diamond <folder with diamond(.exe)>

Run after `gem_pairs.py --stage fetch`. Settings fixed before the result:
  CarveMe 1.6 (BiGG universe), gap-filled on anaerobic and aerobic rich medium (LB[-O2], LB),
  free SCIP solver. SMETANA on each pair in medium LB[-O2] (first run; the reported pilot instead uses
  the primary minimal anaerobic medium and M9[-O2], scored by gem_growth.py, gem_table.py, gem_score.py):
    global   MIP  metabolic interaction potential: how many metabolites the pair can exchange to
                  reduce what it needs from the medium (higher = more potential cooperation)
             MRO  metabolic resource overlap: how much the two compete for the same nutrients
    detailed SCS / MUS / MPS: species coupling, metabolite uptake and production scores, i.e. which
             metabolite would flow from which partner to which
  Positive control: Anaerostipes amylophilus + Bifidobacterium adolescentis should exchange lactate
  and/or acetate (donor Bifidobacterium).
Baseline: each S is also paired with 3 random partners from the shortlist genomes (seed fixed), so a
pair's MIP / MRO can be compared with what the same organism reaches with arbitrary partners.

-> data/motusdb/gem/<genome>.xml        models
   results/gem_smetana_global.tsv, results/gem_smetana_detailed.tsv, results/gem_summary.tsv
"""
import argparse
import gzip
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--diamond", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)
    scripts = os.path.join(a.env, "Scripts" if os.name == "nt" else "bin")
    env = dict(os.environ, PATH=a.diamond + os.pathsep + scripts + os.pathsep + os.environ["PATH"])
    gdir = os.path.join(a.data, "motusdb", "gem")
    P = pd.read_csv(f"{a.out}/gem_pairs.tsv", sep="\t")
    genomes = sorted(set(P.S_genome) | set(P.H_genome))
    for g in genomes:
        xml = os.path.join(gdir, f"{g}.xml")
        if os.path.exists(xml):
            continue
        faa = os.path.join(gdir, f"{g}.faa")
        with gzip.open(os.path.join(gdir, f"{g}.faa.gz"), "rb") as fi, open(faa, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        cmd = [os.path.join(scripts, "carve"), faa, "-o", xml, "--gapfill", "LB[-O2],LB", "-i", "LB[-O2]"]
        print("building", g, flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(xml):
            print("  FAILED:", (r.stderr or r.stdout)[-800:], flush=True)
        os.remove(faa)
    media = os.path.join(a.env, "Lib" if os.name == "nt" else "lib", "site-packages", "carveme", "data", "input", "media_db.tsv")
    if not os.path.exists(media):
        import glob
        media = glob.glob(os.path.join(a.env, "**", "carveme", "data", "input", "media_db.tsv"), recursive=True)[0]
    # communities: the pairs plus 3 random partners per S from the other shortlist genomes
    rng = np.random.default_rng(13)
    comm = [(f"pair|{r.label}|{r.S_species}+{r.H_species}", r.S_genome, r.H_genome) for r in P.itertuples()]
    allg = list(dict.fromkeys(list(P.S_genome) + list(P.H_genome)))
    for r in P.itertuples():
        others = [g for g in allg if g not in (r.S_genome, r.H_genome)]
        # each control needs its own community name, otherwise SMETANA merges S and all 3 partners into one community
        for i, g in enumerate(rng.choice(others, min(3, len(others)), replace=False), 1):
            comm.append((f"random|{r.label}|{r.S_species}+random{i}", r.S_genome, g))
    ctab = os.path.join(gdir, "_communities.tsv")
    ok = {g for g in genomes if os.path.exists(os.path.join(gdir, f"{g}.xml"))}
    with open(ctab, "w") as fh:
        for name, x, y in comm:
            if x in ok and y in ok:
                fh.write(f"{name}\t{x}\n{name}\t{y}\n")
    models = [os.path.join(gdir, f"{g}.xml") for g in sorted(ok)]
    # smetana is installed as a plain Python script (no .exe launcher on Windows): run it with the env's Python
    smet = [os.path.join(scripts, "python.exe" if os.name == "nt" else "python"), os.path.join(scripts, "smetana")]
    for mode, flag in [("global", "-g"), ("detailed", "-d")]:
        pref = os.path.join(gdir, f"_smetana_{mode}")
        cmd = [*smet, *models, "-c", ctab, flag, "--mediadb", media, "-m", "LB[-O2]", "-o", pref, "--solver", "scip"]
        print("smetana", mode, flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print("  FAILED:", (r.stderr or r.stdout)[-1200:], flush=True)
        for f in os.listdir(gdir):
            if f.startswith(f"_smetana_{mode}") and f.endswith(".tsv"):
                shutil.copy(os.path.join(gdir, f), os.path.join(a.out, f"gem_smetana_{mode}.tsv"))
    if os.path.exists(f"{a.out}/gem_smetana_global.tsv"):
        G = pd.read_csv(f"{a.out}/gem_smetana_global.tsv", sep="\t")
        G["kind"] = G.community.str.split("|").str[0]; G["label"] = G.community.str.split("|").str[1]
        print(G.to_string(index=False))
        Sm = G.groupby(["label", "kind"])[[c for c in ["mip", "mro"] if c in G]].mean().unstack()
        Sm.to_csv(f"{a.out}/gem_summary.tsv", sep="\t", float_format="%.3f")
        print(Sm.round(3).to_string())
    if os.path.exists(f"{a.out}/gem_smetana_detailed.tsv"):
        Dd = pd.read_csv(f"{a.out}/gem_smetana_detailed.tsv", sep="\t")
        Dd = Dd[Dd.community.str.startswith("pair|")]
        top = Dd.sort_values("smetana", ascending=False).groupby("community").head(5) if "smetana" in Dd else Dd.head(40)
        with pd.option_context("display.width", 220, "display.max_rows", 80):
            print(top.to_string(index=False))


if __name__ == "__main__":
    main()
