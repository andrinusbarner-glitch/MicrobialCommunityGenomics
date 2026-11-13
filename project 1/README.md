# Co-occurrence structure of the global prokaryotic community

Code and result tables for the report *Investigating Co-Occurrence Structures of the Global
Prokaryotic Community* (Microbial Community Genomics, D-BIOL, ETH Zürich).

Pairwise associations among 9,634 mOTUs detected in at least 50 of 13,310 metagenomic samples are
tested on presence–absence data with stratified Cochran–Mantel–Haenszel tests under five nested null
models (none; + environment; + sample-richness quartile; + study; sub-habitat × study × richness, the
main model with 244 strata). The method is checked on a positive control (Saccharibacteria/TM7 and
their Actinomycetota hosts), pairs selected in one half of 64 independent studies are tested in the
other half, and a pilot with genome-scale metabolic models (CarveMe + SMETANA) scores eight shortlist
pairs against random partners.

## Where each result in the report comes from

| Report section | Script(s) | Result table(s) in `results/` |
|---|---|---|
| Methods: nested null models, CMH test, BH-FDR, effect size r | `p9_pipeline.py` | `run_summary.tsv`, `diagnostics.json` |
| 3.1 Habitat filtering (30.0 M / 10.1 M pairs, 95 % / 66 % lost or reversed, 81 %, 0.34 → 0.94) | `p9_pipeline.py` | `run_summary.tsv`, `edge_fate.tsv` |
| 3.1 Replication in 64 independent studies (93 % vs 31 %) | `study_replication.py` | `study_replication_summary.tsv` |
| 3.2 Partners are relatives (79–91 % of null distance) | `env_networks.py` (uses `phylogeny.py`) | `env_network_phylo.tsv` |
| 3.2 Environment comparison at equal sample size (5 % Human … 25 % Saline) | `env_networks.py` | `env_network_summary.tsv` |
| 3.2 Specific / shared / reversed pairs (88.7 / 11.0 / 0.4 %) | `context.py` | `context_pair_classes.tsv` |
| 3.3 TM7 positive control (32 % vs 10 %, 51 % vs 15 %) | `tm7_benchmark.py` | `tm7_benchmark_summary.tsv`, `tm7_benchmark_queries.tsv` |
| 3.3 Abundance regression and mixed model (34 % vs 15 %) | `host_regression.py`, `host_regression_lme4.R` | `host_regression_summary.tsv`, `host_regression_lme4.tsv` |
| 3.4 Shortlist (93.0 / 93.4 % confirmed, 583 pairs) | `shortlist.py` | `shortlist_summary.tsv`, `shortlist_final.tsv` |
| 3.4 Marine re-test with depth, latitude, size fraction (94.1 → 90.4 %) | `marine_metadata.py`, `marine_retest.py` | `marine_retest_summary.tsv`, `marine_retest_pairs.tsv` |
| 3.5 Metabolic-model pilot (MIP, MRO) | `gem_pairs.py`, `gem_build.py`, `gem_smetana.py`, `gem_growth.py`, `gem_table.py`, `gem_score.py` | `gem_pairs.tsv`, `gem_growth.tsv`, `gem_testability*.tsv`, `gem_smetana_global.tsv`, `gem_scores.tsv` |
| Figures 1–5 | `report_figures.py` | `figures/report/fig{1..5}_*_col.pdf` |

`report_figures.py` prints every number it draws, so the text can be checked against the tables.

**Interactive network explorer.** `results/network_explorer.html` is a self-contained page (download
it and open it in any browser; no installation). It shows the positive co-occurrence network of each
of the five null models, so you can see the same mOTUs move as controls are added. Nodes can be
coloured by environment, sub-habitat, phylum or cluster; hovering shows a node's taxon and prevalence;
clicking a cluster highlights it and lists its dominant family, environment and top study. It is a
finished view of the pipeline's networks and clusters (`network_edges_*.tsv.gz`, `clusters_*.tsv`,
`motus.tsv`).

Supporting files: `habitats.py` assigns each sample its sub-habitat (needed by the main model's
strata); `interaction_types.py` classifies the non-marine shortlist pairs, and `gem_pairs.py` reads its
table to choose the modelled pairs; `phylogeny.py`, `abundance.py` and `kegg_pilot.py` are imported as
helper modules (tree distances, CLR transform, mOTUs API download).

## Installation

Co-occurrence analysis (Python 3.13):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip install -r requirements.txt
```

Mixed-model check (optional): R 4.4 with `lme4` 1.1-37.

Metabolic-model pilot, in a separate environment (Python 3.13):

```bash
python -m venv gemenv
gemenv/bin/pip install -r requirements-gem.txt
```

plus DIAMOND 2.2.8 on the `PATH` (needed by CarveMe).

## Input data

The input data are not redistributed here; see [`data/README.md`](data/README.md) for what goes in
`data/` and where to get it.

## Running

1. One-off downloads (resumable; need internet):
   ```bash
   python fetch_genome_quality.py      # genome size and quality per mOTU -> data/motusdb/genome_quality.tsv
   python marine_metadata.py           # depth, latitude, size fraction of marine samples (already included)
   ```
2. The co-occurrence analysis, in order (`interaction_types.py` downloads KEGG annotations from the
   mOTUs API, so this step needs internet):
   ```bash
   python run_all.py                   # or: python run_all.py --from shortlist
   ```
3. Optional mixed-model check, after `host_regression.py`:
   ```bash
   Rscript host_regression_lme4.R
   ```
4. Metabolic-model pilot. `gem_pairs.py` uses the main environment; the next three steps use
   `gemenv`:
   ```bash
   python gem_pairs.py --stage fetch                                  # protein FASTA from the mOTUs API
   gemenv/bin/python gem_build.py --env gemenv --diamond <diamond folder>   # CarveMe models (+ withdrawn LB run)
   gemenv/bin/python gem_smetana.py --env gemenv                      # SMETANA: minimal anaerobic and M9[-O2]
   gemenv/bin/python gem_growth.py
   gemenv/bin/python gem_table.py
   python gem_score.py                                                # -> results/gem_scores.tsv
   python report_figures.py                                           # redraw Figure 5
   ```
   The community tables and SMETANA outputs used in the report are included in
   `data/motusdb/gem/`, so `gem_score.py` can be rerun without building the models.

## Notes on the analysis

- **Significance and strength.** Significance comes from the CMH z (Benjamini–Hochberg); association
  strength is the within-stratum phi coefficient r, because z grows with sample size and r does not.
- **Frozen definitions.** The TM7 benchmark (`tm7_benchmark.py`, v1), the shortlist rules
  (`shortlist.py`) and the metabolic-model rules (`gem_table.py`, `gem_score.py`) were fixed before the
  results were read.
- **Metabolic pilot.** The first SMETANA run (rich medium, in `gem_build.py`) was withdrawn: its control
  communities shared one name, so SMETANA merged them. The reported scores come from
  `gem_smetana.py`, where every control is its own two-member community. Controls are random partners
  from the other modelled genomes, not matched for environment or genome completeness.
- **Data limits.** "Non-saline" is mostly wastewater, not freshwater; freshwater has too few testable
  mOTUs. Marine samples lack depth and size fraction in the course metadata, which is why they were
  re-tested with metadata recovered from BioSample and ENA.

## Software versions

Python 3.13.7, NumPy 2.5.4, pandas 3.1.1, SciPy 1.18.1, scikit-learn 1.9.1, NetworkX 3.7,
matplotlib 3.11.2; R 4.4.3, lme4 1.1.37; CarveMe 1.6.6, DIAMOND 2.2.8, SMETANA 1.2.1, reframed 1.6.0,
COBRApy 0.32.1, PySCIPOpt 6.3.0 (SCIP solver).

## Citation of data sources

- mOTUs: Sunagawa et al. 2013, *Metagenomic species profiling using universal phylogenetic marker
  genes*; Dmitrijeva et al., *The mOTUs online database provides web-accessible genomic context to
  taxonomic profiling of microbial communities* (PubMed 39526369); supplementary table 4 via Zenodo,
  doi:10.5281/zenodo.13325008 (CC-BY-4.0).
- Marine sample metadata: NCBI BioSample and ENA public records.
