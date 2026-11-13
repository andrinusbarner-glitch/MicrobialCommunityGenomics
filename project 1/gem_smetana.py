"""SMETANA runs of the metabolic-model pilot as reported (run after gem_build.py).

    python gem_smetana.py --env <venv with carveme + smetana>

gem_build.py builds the CarveMe models and writes data/motusdb/gem/_communities.tsv (the 8 pairs and
3 random partners per S, seed 13). Its own SMETANA run on LB[-O2] is the first, withdrawn run; the
report uses the four runs below, all with the SCIP solver:

  primary medium   SMETANA's minimal medium derived from the complete anaerobic medium
                   (no -m, --anaerobic)
  sensitivity      M9 without oxygen (-m "M9[-O2]", CarveMe's media_db.tsv)

each for the pairs (_pairs_only.tsv) and for the random-partner controls (_controls.tsv); every control
has its own community name, so SMETANA scores two-member communities.

-> data/motusdb/gem/_smetana_minimal_global.tsv, _smetana_M9_global.tsv,
   data/motusdb/gem/_ctrl_minimal_global.tsv,   _ctrl_M9_global.tsv
Then: gem_growth.py, gem_table.py (modelling env), gem_score.py.
"""
import argparse
import glob
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="virtual environment with carveme and smetana")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    a = ap.parse_args(argv)
    gdir = os.path.join(a.data, "motusdb", "gem")
    scripts = os.path.join(a.env, "Scripts" if os.name == "nt" else "bin")
    media = glob.glob(os.path.join(a.env, "**", "carveme", "data", "input", "media_db.tsv"), recursive=True)[0]
    models = sorted(glob.glob(os.path.join(gdir, "*.xml")))

    # split the community table into pairs and controls (same rows as in _communities.tsv)
    rows = open(os.path.join(gdir, "_communities.tsv")).read().splitlines()
    for name, keep in [("_pairs_only.tsv", "pair|"), ("_controls.tsv", "random|")]:
        with open(os.path.join(gdir, name), "w") as fh:
            fh.write("\n".join(r for r in rows if r.startswith(keep)) + "\n")

    # smetana is a plain Python script (no .exe launcher on Windows): run it with the env's Python
    smet = [os.path.join(scripts, "python.exe" if os.name == "nt" else "python"), os.path.join(scripts, "smetana")]
    runs = [("_pairs_only.tsv", "_smetana_minimal", ["--anaerobic"]),
            ("_pairs_only.tsv", "_smetana_M9", ["--mediadb", media, "-m", "M9[-O2]"]),
            ("_controls.tsv", "_ctrl_minimal", ["--anaerobic"]),
            ("_controls.tsv", "_ctrl_M9", ["--mediadb", media, "-m", "M9[-O2]"])]
    for comm, prefix, medium in runs:
        print("smetana", prefix, flush=True)
        cmd = [*smet, *models, "-c", os.path.join(gdir, comm), "-g", *medium,
               "-o", os.path.join(gdir, prefix), "--solver", "scip"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
