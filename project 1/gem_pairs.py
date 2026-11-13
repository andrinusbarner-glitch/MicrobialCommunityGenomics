"""Systems-biology step: genome-scale metabolic models for the most promising shortlist pairs.

    python gem_pairs.py --stage fetch      # protein FASTA of the representative genomes (mOTUs API)
    (models are built with CarveMe and scored with SMETANA by gem_run.ps1, see README)
    python gem_pairs.py --stage report     # collect SMETANA scores into results/gem_*

Pairs (curated from the non-marine shortlist, using interaction_types.tsv):
  positive control  Anaerostipes amylophilus -> Bifidobacterium adolescentis (documented lactate/acetate
                    cross-feeding for the genus pair)
  others            Turicibacter -> Romboutsia, Lawsonella -> Cutibacterium acnes, Prevotella ->
                    Succiniclasticum, TM7 CALMAK01 -> Scrofimicrobium, TM7 Nanosynbacter -> Leucobacter,
                    Methanobrevibacter smithii + SFEL01, T78 + Methanothrix soehngenii
Caveats, stated before the result: models from MAGs are incomplete and CarveMe gap-fills them, which
can add capabilities the organism lacks; TM7 models are largely gap-filled guesses.

-> data/motusdb/gem/<genome>.faa.gz   protein sequences
   results/gem_pairs.tsv              the pairs with their genomes
"""
import argparse
import os

import pandas as pd

from kegg_pilot import get

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [("Anaerostipes amylophilus", "Bifidobacterium adolescentis", "positive control"),
         ("Turicibacter sanguinis", "Romboutsia timonensis", "one-way"),
         ("Lawsonella clevelandensis_A", "Cutibacterium acnes", "one-way"),
         ("Prevotella sp900316645", "Succiniclasticum sp900315925", "succinate"),
         ("Unknown CALMAK01", "Scrofimicrobium sp014069575", "TM7"),
         ("Nanosynbacter sp002421965", "Leucobacter sp002336855", "TM7"),
         ("Methanobrevibacter_A smithii", "SFEL01 sp004557245", "H2"),
         ("Unknown T78", "Methanothrix soehngenii", "methanogen")]
API = "https://motus-api.microbiomics.io/v1/genomes/{g}/download?file_type=gene_faa"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--stage", choices=["fetch", "report"], default="fetch")
    a = ap.parse_args(argv)
    T = pd.read_csv(f"{a.out}/interaction_types.tsv", sep="\t")
    GQ = pd.read_csv(f"{a.data}/motusdb/genome_quality.tsv", sep="\t", index_col="motu_id")
    gdir = os.path.join(a.data, "motusdb", "gem"); os.makedirs(gdir, exist_ok=True)
    rows = []
    for s, h, tag in PAIRS:
        r = T[(T.S_species == s) & (T.H_species == h)]
        if not len(r):
            print("pair not found:", s, h); continue
        r = r.iloc[0]
        rows.append({"label": tag, "S": r.S, "S_species": s, "H": r.H, "H_species": h,
                     "S_genome": GQ.rep_genome.get(r.S), "H_genome": GQ.rep_genome.get(r.H),
                     "S_completeness": GQ.rep_completeness.get(r.S), "H_completeness": GQ.rep_completeness.get(r.H)})
    P = pd.DataFrame(rows); P.to_csv(f"{a.out}/gem_pairs.tsv", sep="\t", index=False)
    if a.stage == "fetch":
        for g in sorted(set(P.S_genome) | set(P.H_genome)):
            ok = get(API.format(g=g), os.path.join(gdir, f"{g}.faa.gz"))
            print(("ok  " if ok else "FAIL"), g)
        print(P[["label", "S_species", "S_completeness", "H_species", "H_completeness"]].to_string(index=False))


if __name__ == "__main__":
    main()
