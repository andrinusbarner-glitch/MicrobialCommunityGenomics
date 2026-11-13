"""Score the metabolic-model pilot under the rules fixed before reading (run after gem_table.py).

    python gem_score.py

Inputs: SMETANA global output for the 8 pairs and the 24 random-partner controls (3 per pair), each in
the primary medium (SMETANA's minimal media derived from the complete anaerobic medium) and in M9[-O2]
(sensitivity), plus results/gem_testability.tsv.
Rules: a community whose members cannot all grow in a medium gets no score there (NaN, left out, never
counted as 'no cooperation'); SMETANA 'n/a' is missing; a pair with fewer than 2 usable controls for a
score is untestable for that score.
Reading rules: MRO higher than controls = similar nutrient requirements, consistent with shared niche or
relatedness; it does not distinguish interaction. Controls are unmatched random partners from the
shortlist genomes (other environments included), so they are a weak baseline. 8 pairs = pilot.

-> results/gem_smetana_global.tsv  every community x medium with usable flag and cleaned scores
   results/gem_scores.tsv          per pair x medium: pair score, control mean/range, n usable controls
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
g = os.path.join(HERE, "data", "motusdb", "gem")
out = os.path.join(HERE, "results")
parts = []
# pair rows from the main run; its control rows merged S with all 3 partners (shared community name) and
# are discarded; controls come from the separate run on correctly named two-member communities
for f, med in [("_smetana_minimal_global.tsv", "minimal"), ("_smetana_M9_global.tsv", "M9[-O2]"),
               ("_ctrl_minimal_global.tsv", "minimal"), ("_ctrl_M9_global.tsv", "M9[-O2]")]:
    d = pd.read_csv(os.path.join(g, f), sep="\t", na_values=["n/a"])
    if f.startswith("_smetana"):
        d = d[d.community.str.startswith("pair|")]
    assert (d["size"] == 2).all(), f
    d["medium"] = med
    parts.append(d)
G = pd.concat(parts, ignore_index=True)
T = pd.read_csv(os.path.join(out, "gem_testability.tsv"), sep="\t")[["community", "medium", "usable", "grow_alone", "members"]]
G = G.merge(T, on=["community", "medium"], how="left")
G["usable"] = G.usable.fillna(False).astype(bool)
for c in ["mip", "mro"]:
    G[c] = pd.to_numeric(G[c], errors="coerce").where(G.usable)
G["kind"] = G.community.str.split("|").str[0]
G["label"] = G.community.str.split("|").str[1]
G["S"] = G.community.str.split("|").str[2].str.split("+").str[0]
G[["community", "kind", "label", "S", "medium", "members", "grow_alone", "usable", "mip", "mro"]].to_csv(
    os.path.join(out, "gem_smetana_global.tsv"), sep="\t", index=False)
rows = []
for (lab, s, med), d in G.groupby(["label", "S", "medium"], sort=False):
    pr = d[d.kind == "pair"].iloc[0]
    rc = d[d.kind == "random"]
    r = {"pair": pr.community.split("|")[2], "group": lab, "medium": med, "pair_usable": bool(pr.usable)}
    for c in ["mip", "mro"]:
        v = rc[c].dropna()
        r[c] = pr[c]
        r[f"{c}_ctrl_n"] = len(v)
        r[f"{c}_ctrl_mean"] = v.mean() if len(v) >= 2 else np.nan
        r[f"{c}_ctrl_min"] = v.min() if len(v) >= 2 else np.nan
        r[f"{c}_ctrl_max"] = v.max() if len(v) >= 2 else np.nan
        r[f"{c}_testable"] = bool(pr.usable) and not np.isnan(pr[c]) and len(v) >= 2
        r[f"{c}_above_all_ctrl"] = bool(r[f"{c}_testable"] and pr[c] > v.max())
    rows.append(r)
S = pd.DataFrame(rows)
S.to_csv(os.path.join(out, "gem_scores.tsv"), sep="\t", index=False, float_format="%.3f")
with pd.option_context("display.width", 250):
    print(S.round(3).to_string(index=False))
for med in ["minimal", "M9[-O2]"]:
    d = S[S.medium == med]
    print(f"\n{med}: MIP testable {d.mip_testable.sum()}/{len(d)}, above all controls {d.mip_above_all_ctrl.sum()};"
          f" MRO testable {d.mro_testable.sum()}/{len(d)}, above all controls {d.mro_above_all_ctrl.sum()}")
    t = d[d.mro_testable]
    if len(t):
        print(f"  MRO pair mean {t.mro.mean():.3f} vs control mean {t.mro_ctrl_mean.mean():.3f}; pair > control mean in {(t.mro > t.mro_ctrl_mean).sum()}/{len(t)}")
    t = d[d.mip_testable]
    if len(t):
        print(f"  MIP pair mean {t.mip.mean():.2f} vs control mean {t.mip_ctrl_mean.mean():.2f}; pair > control mean in {(t.mip > t.mip_ctrl_mean).sum()}/{len(t)}")
