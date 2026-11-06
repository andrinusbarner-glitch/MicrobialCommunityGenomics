# P3 co-occurrence: streamlined pipeline

```
P3_streamlined/
  data/              raw input (extracted from "Raw Data.tar")
  p3_pipeline.py     statistics, networks, clusters -> results/
  p3_viz.py          figure functions; only reads results/
  make_figures.py    writes all figures to results/figures/
  results/           output of the run (see below)
```

Run:

```bash
python p3_pipeline.py
```

```bash
python make_figures.py
```

On the course server, pass the masterdata folder: `python p3_pipeline.py --data /nfs/.../motus_small_sample/`.

## What changed compared with `cooccurrence_clusters_abundance.py`

| Before | Now | Why |
|---|---|---|
| All 10,183 mOTUs | mOTUs with ≥ 50 detections **in this subset** (`--min-prev`) | Some mOTUs occur in only 1–5 of these samples; they gave z ≈ 100 and a 378-mOTU fake cluster |
| Strata = environment | Main model strata = environment × within-environment **richness quartile** (16 strata) | Rich/deep samples detect many mOTUs at once, which creates positive co-occurrence that is not biology |
| 6 analyses (PA/abundance/combined × global/stratified) | 3 PA models: `global` (baseline), `env`, `env_rich` (main) | Combined ≈ PA; abundance global ≈ stratified |
| Edge weight = \|z\| | Significance from z (BH-FDR); weight = effect size r, edges need r ≥ `--min-r` | z grows with sample size; r does not |
| Positive and negative networks clustered | Positive network only; negative edges are counted | Clustering mutual-avoidance edges has no meaning |
| `top_phylum` always NaN | Taxonomy read from the plain rank columns | Bug fix |
| Viz re-ran Louvain | Viz reads the pipeline's cluster labels | Figures and tables now agree |
| Full significant edge lists (35–40M rows) | Network edges + summary statistics | Only these were used |

Effect size r: the correlation of two mOTUs' 0/1 vectors after centring each vector within each stratum (pooled phi coefficient).
For the global model, z = r·√(N−1) exactly.

## Results files

- `run_summary.tsv`: per model: z threshold, significant +/− pairs, network size, clusters, modularity (Louvain vs environment-only partition), AMI with environment, degree–prevalence correlation, Louvain seed stability (main model).
- `edge_fate.tsv`: what happens to significant pairs going global → env → env_rich (kept / lost / flipped / new; how many lost pairs are within one environment).
- `motus.tsv`: one row per mOTU: prevalence, filter flag, habitat profile, taxonomy, cluster and degree in each model.
- `samples.tsv`: environment, richness, read depth and stratum for every sample.
- `clusters_<model>.tsv`: one row per cluster: size, environment profile, top phylum/family/genus, median prevalence.
- `network_edges_<model>.tsv.gz`: the clustered network (a, b, r, z).
- `envpair_density_<model>_<pos|neg>.tsv`: share of possible pairs between two environments that are significant.
- `effect_size_hist.tsv`, `diagnostics.json`, `run_log.txt`.
