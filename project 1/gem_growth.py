"""Which metabolic models grow on which medium? (run with the modelling environment's Python)

    <gemenv>\\Scripts\\python gem_growth.py

Before any SMETANA score is read: a community with a member that cannot grow gives a missing value,
not a zero. Media: complete anaerobic (all exchanges open, no O2; the medium of the primary SMETANA
run), M9[-O2] (minimal anaerobic, glucose; the sensitivity run), LB[-O2] (rich anaerobic, the
gap-filling medium). Growth = biomass flux > 1e-3 with FBA.

-> results/gem_growth.tsv
"""
import glob
import os
import sys

import pandas as pd
from reframed import Environment, FBA, load_cbmodel, set_default_solver

HERE = os.path.dirname(os.path.abspath(__file__))
set_default_solver("scip")
import carveme
media_db = pd.read_csv(os.path.join(os.path.dirname(carveme.__file__), "data", "input", "media_db.tsv"), sep="\t")
P = pd.read_csv(os.path.join(HERE, "results", "gem_pairs.tsv"), sep="\t")
names = dict(zip(P.S_genome, P.S_species)) | dict(zip(P.H_genome, P.H_species))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "data", "motusdb", "gem", "*.xml"))):
    g = os.path.basename(f)[:-4]
    m = load_cbmodel(f, flavor="fbc2")
    out = {"genome": g, "species": names.get(g, ""), "reactions": len(m.reactions)}
    for med in ["complete[-O2]", "M9[-O2]", "LB[-O2]"]:
        if med == "complete[-O2]":
            env = Environment.complete(m, max_uptake=10)
            if "R_EX_o2_e" in m.reactions:
                env["R_EX_o2_e"] = (0, float("inf"))
        else:
            comp = media_db[media_db.medium == med].compound.tolist()
            env = Environment.from_compounds(comp, fmt_func=lambda x: f"R_EX_{x}_e", max_uptake=10).apply(m, inplace=False, warning=False)
        sol = FBA(m, constraints=env)
        out[med] = round(sol.fobj, 4) if sol and sol.fobj is not None else 0.0
    rows.append(out)
    print(out, flush=True)
T = pd.DataFrame(rows)
for med in ["complete[-O2]", "M9[-O2]", "LB[-O2]"]:
    T[f"grows_{med}"] = T[med] > 1e-3
T.to_csv(os.path.join(HERE, "results", "gem_growth.tsv"), sep="\t", index=False)
print(T[["species", "reactions"] + [c for c in T if c.startswith("grows_")]].to_string(index=False))
