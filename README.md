# BC2026 — mOTUs working dataset

Derived from `~/Desktop/Github/Data/mOTUsMatrix/` and `~/Desktop/Github/mOTUs_tree/`,
filtered and preprocessed for the 2026 Block Course projects. **All matrices are gzipped TSV** — load with `pd.read_csv(path, sep="\t", index_col=0)`.

## Files

### Working subset — 10,183 mOTUs × 13,310 samples
Filter: mOTUs detected in ≥500 samples, samples containing ≥500 mOTUs, and samples belonging to one of four broad environmental groups (see below).

- **`motus_subset_matrix.tsv.gz`** (11 MB) — Dense read-count matrix (int32).
- **`motus_subset_pa.tsv.gz`** (4 MB) — Dense presence/absence matrix (int8). Start here for most analyses.
- **`motus_subset_motu_index.tsv`** / **`motus_subset_sample_index.tsv`** — Row / column IDs.
- **`motus_subset_taxonomy.tsv`** — GTDB taxonomy for the 10,183 filtered mOTUs.
- **`motus_subset_environment.tsv`** — Two columns: `sample_id`, `environment`. Values:

  | environment | n samples |
  |---|---|
  | Saline | 4,334 |
  | Animal-associated | 3,715 |
  | Human-associated | 3,444 |
  | Non-saline | 1,817 |

### Phylogeny (P2, P3, P4)
- **`mOTUs_tree_bacteria.nwk`** — FastTree bacterial phylogeny from concatenated single-copy markers.

### Function (P4)
- **`motus_subset_cog_matrix.tsv.gz`** (2 MB) — COG-count matrix (int16) for the 2,094 mOTUs whose representative genome maps to a GTDB reference. Rows = mOTU, columns = 4,789 COGs. *Taxonomically biased — isolates only.*
- **`cog_annotations.tsv`** — COG → functional category + gene name + pathway.

## Which files per project

| Project | Needs |
|---|---|
| **P1** Specialists vs generalists | `motus_subset_pa.tsv.gz` · `motus_subset_taxonomy.tsv` · `motus_subset_environment.tsv` |
| **P2** Phylogenetic signal | `motus_subset_pa.tsv.gz` · `motus_subset_taxonomy.tsv` · `motus_subset_environment.tsv` · `mOTUs_tree_bacteria.nwk` |
| **P3** Co-occurrence | `motus_subset_pa.tsv.gz` · `motus_subset_taxonomy.tsv` · `motus_subset_environment.tsv` · `mOTUs_tree_bacteria.nwk` |
| **P4** Streamlining | `motus_subset_cog_matrix.tsv.gz` · `cog_annotations.tsv` · `motus_subset_taxonomy.tsv` · `motus_subset_environment.tsv` · `mOTUs_tree_bacteria.nwk` |

## Loading examples

```python
import pandas as pd

pa   = pd.read_csv("motus_subset_pa.tsv.gz",             sep="\t", index_col=0)
tax  = pd.read_csv("motus_subset_taxonomy.tsv",          sep="\t")
env  = pd.read_csv("motus_subset_environment.tsv",       sep="\t")
cog  = pd.read_csv("motus_subset_cog_matrix.tsv.gz",     sep="\t", index_col=0)
```

## Filter rationale
The raw atlas is 117k × 112k with median mOTU prevalence 40 and median sample richness 158 — the vast majority is sparse, rare mOTUs. Filtering to `prev ≥ 500`, `richness ≥ 500`, and one of the four well-populated L2 environments yields a subset that:

- fits comfortably in pandas,
- retains the ecologically well-supported core of the atlas,
- covers four biologically meaningful, well-populated environmental groups (marine, freshwater, animal, human) for cross-biome comparisons.
