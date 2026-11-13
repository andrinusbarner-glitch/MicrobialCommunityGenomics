"""Testability table for the metabolic-model pilot (run with the modelling environment's Python).

    <gemenv>\\Scripts\\python gem_table.py

Rules agreed before any SMETANA score is read:
  - missing is not zero: a member that cannot grow in a medium gives no score there (NaN, left out)
  - growing only together with the partner is a result (clearest dependence signal SMETANA can give),
    but for gap-filled models possibly an artefact, so completeness is shown next to it
  - same rules for random partners: a control whose members cannot grow drops out; a pair with
    fewer than 2 usable controls in a medium is untestable in that medium
  - primary medium: SMETANA's minimal medium derived from the complete anaerobic medium; M9[-O2] is
    the sensitivity check
Growth alone uses gem_growth.tsv; growth of a two-member community = community biomass > 1e-3 (FBA).

-> results/gem_testability.tsv   one row per community (pairs and controls)
   results/gem_testability_pairs.tsv  per pair: testable in each medium, usable controls
"""
import os

import pandas as pd
from reframed import Community, Environment, FBA, load_cbmodel, set_default_solver

HERE = os.path.dirname(os.path.abspath(__file__))
set_default_solver("scip")
import carveme
media_db = pd.read_csv(os.path.join(os.path.dirname(carveme.__file__), "data", "input", "media_db.tsv"), sep="\t")
gdir = os.path.join(HERE, "data", "motusdb", "gem")
Gr = pd.read_csv(os.path.join(HERE, "results", "gem_growth.tsv"), sep="\t").set_index("genome")
P = pd.read_csv(os.path.join(HERE, "results", "gem_pairs.tsv"), sep="\t")
comp = dict(zip(P.S_genome, P.S_completeness)) | dict(zip(P.H_genome, P.H_completeness))
C = pd.read_csv(os.path.join(gdir, "_communities.tsv"), sep="\t", header=None, names=["community", "genome"])
models = {}


def model(g):
    if g not in models:
        models[g] = load_cbmodel(os.path.join(gdir, f"{g}.xml"), flavor="fbc2")
    return models[g]


def grows_together(gs, med):
    com = Community("c", [model(g) for g in gs], copy_models=True)
    mm = com.merged_model
    if med == "minimal":
        env = Environment.complete(mm, max_uptake=20)
        if "R_EX_M_o2_e_pool" in mm.reactions:
            env["R_EX_M_o2_e_pool"] = (0, float("inf"))
    else:
        cpds = media_db[media_db.medium == med].compound.tolist()
        env = Environment.from_compounds(cpds, fmt_func=lambda x: f"R_EX_M_{x}_e_pool", max_uptake=20).apply(mm, inplace=False, warning=False)
    sol = FBA(mm, constraints=env)
    return bool(sol and sol.fobj is not None and sol.fobj > 1e-3)


rows = []
for name, d in C.groupby("community", sort=False):
    gs = list(d.genome)
    kind, label = name.split("|")[0], name.split("|")[1]
    for med, col in [("minimal", "grows_complete[-O2]"), ("M9[-O2]", "grows_M9[-O2]")]:
        alone = [bool(Gr.loc[g, col]) for g in gs]
        together = grows_together(gs, med)
        rows.append({"community": name, "kind": kind, "label": label, "S": name.split("|")[2].split("+")[0], "medium": med,
                     "members": " + ".join(f"{g[:22]} ({comp.get(g, float('nan')):.0f}%)" for g in gs),
                     "grow_alone": " / ".join("yes" if a else "no" for a in alone), "grow_together": together,
                     "usable": all(alone) and together,
                     "grows_only_together": together and not all(alone)})
T = pd.DataFrame(rows)
T.to_csv(os.path.join(HERE, "results", "gem_testability.tsv"), sep="\t", index=False)
summ = []
for (label, s, med), d in T.groupby(["label", "S", "medium"], sort=False):
    pr = d[d.kind == "pair"].iloc[0]
    nctrl = int(d[(d.kind == "random")].usable.sum())
    summ.append({"pair": f"{label}: {pr.community.split('|')[2]}", "medium": med, "pair_usable": bool(pr.usable), "pair_grows_only_together": bool(pr.grows_only_together),
                 "usable_controls": nctrl, "testable": bool(pr.usable) and nctrl >= 2})
S = pd.DataFrame(summ)
S.to_csv(os.path.join(HERE, "results", "gem_testability_pairs.tsv"), sep="\t", index=False)
with pd.option_context("display.width", 220, "display.max_colwidth", 80):
    print(T[T.kind == "pair"][["label", "medium", "members", "grow_alone", "grow_together", "grows_only_together"]].to_string(index=False))
    print()
    print(S.to_string(index=False))
    print("\ntestable pairs: minimal", int(S[S.medium == "minimal"].testable.sum()), "of", (S.medium == "minimal").sum(),
          "| M9[-O2]", int(S[S.medium == "M9[-O2]"].testable.sum()), "of", (S.medium == "M9[-O2]").sum())
