"""Run the co-occurrence analysis of the report in order.

    python run_all.py [--from STEP]

Each step reads the previous steps' files in results/; --from restarts at a step
(e.g. --from shortlist) after a crash or a change to one script.

Not included here (see README): the one-off downloads fetch_genome_quality.py and
marine_metadata.py, the optional R check host_regression_lme4.R, and the metabolic-model
pilot (gem_*.py), which needs a separate environment with CarveMe and SMETANA.
phylogeny.py, abundance.py and kegg_pilot.py are imported as helper modules and are not run.
"""
import argparse
import subprocess
import sys
import time

STEPS = [
    ("pipeline", ["p9_pipeline.py"]),              # five nested null models
    ("habitats", ["habitats.py"]),                 # sub-habitat of every sample
    ("env_networks", ["env_networks.py"]),         # per-environment networks, partner relatedness
    ("context", ["context.py"]),                   # specific / shared / reversed pairs
    ("tm7_benchmark", ["tm7_benchmark.py"]),       # positive control
    ("host_regression", ["host_regression.py"]),   # abundance regression for the positive control
    ("study_replication", ["study_replication.py"]),  # replication in 64 independent studies
    ("shortlist", ["shortlist.py"]),               # held-out shortlist
    ("marine_retest", ["marine_retest.py"]),       # marine pairs with depth, latitude, size fraction
    ("interaction_types", ["interaction_types.py"]),  # pair types; gem_pairs.py reads its table
    ("report_figures", ["report_figures.py"]),     # Figures 1-5
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0], choices=[s for s, _ in STEPS])
    a = ap.parse_args()
    names = [s for s, _ in STEPS]
    for name, cmd in STEPS[names.index(a.start):]:
        t0 = time.time()
        print(f"\n===== {name} =====", flush=True)
        r = subprocess.run([sys.executable, "-u", "-W", "ignore"] + cmd)
        print(f"===== {name}: exit {r.returncode}, {(time.time() - t0) / 60:.1f} min =====", flush=True)
        if r.returncode != 0:
            sys.exit(f"step {name} failed; fix it and rerun with --from {name}")
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
