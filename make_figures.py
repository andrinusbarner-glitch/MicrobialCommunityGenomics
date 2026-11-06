"""Write every P3 figure to results/figures/ (run after p3_pipeline.py).

    python make_figures.py [--data data] [--out results]
"""
import argparse
import os
import sys

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import p3_pipeline as p
import p3_viz as v

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)
    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
    save = lambda fig, name: (fig.savefig(os.path.join(fig_dir, name), dpi=180, bbox_inches="tight"),
                              plt.close(fig), print("wrote", name))

    R = v.load_results(a.out)
    pa, _, _ = p.load(a.data, R["diag"]["params"]["min_prev"])

    fig, R2 = v.plot_environment_overview(pa, R); save(fig, "01_environment_overview.png")
    save(v.plot_richness_diagnostics(R), "02_richness_depth_diagnostics.png")
    save(v.plot_effect_sizes(R), "03_effect_sizes.png")
    ax = v.plot_edge_fate(R, ("global->env", "env->env_rich")); save(ax.figure, "04_edge_fate.png")
    save(v.plot_envpair_density(R), "05_envpair_density.png")

    nets = {m: v.network(a.out, R, m) for m in ["global", p.MAIN]}
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    v.plot_network(nets["global"], R, "env", ax=ax[0, 0])
    v.plot_network(nets[p.MAIN], R, "env", ax=ax[0, 1])
    v.plot_network(nets[p.MAIN], R, "cluster", ax=ax[1, 0])
    v.plot_network(nets[p.MAIN], R, "phylum", ax=ax[1, 1])
    fig.tight_layout(); save(fig, "06_networks.png")
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    v.plot_network(nets[p.MAIN], R, "breadth", ax=ax[0])
    v.plot_network(nets[p.MAIN], R, "prevalence", ax=ax[1])
    fig.tight_layout(); save(fig, "07_network_breadth_prevalence.png")
    save(v.plot_cluster_env_heatmap(R, p.MAIN), "08_cluster_env_profile.png")
    save(v.plot_partition_comparison(R, "env", p.MAIN), "09_clusters_env_vs_env_rich.png")
    v.export_gephi(nets[p.MAIN], R, os.path.join(fig_dir, f"network_{p.MAIN}.graphml"))
    print(f"environment explains {R2:.1%} of Jaccard variation between samples")


if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
