# Input data

The input data are not part of this repository. Put them here with these names:

| File | Content | Source |
|---|---|---|
| `motus_subset_pa.tsv.gz` | presence/absence of mOTUs per sample (13,310 samples) | course data set (mOTUs subset) |
| `motus_subset_matrix.tsv.gz` | read counts per mOTU and sample | course data set |
| `motus_subset_environment.tsv` | environment label per sample (Human, Animal, Saline, Non-saline) | course data set |
| `motus_subset_taxonomy.tsv` | taxonomy per mOTU | course data set |
| `mOTUs_tree_bacteria.nwk` | mOTUs bacterial tree (patristic distances) | course data set |
| `motusdb/Supplementary_Table_4.tsv.gz` | sample → BioSample and NCBI environment term (sub-habitat) | mOTUs-db, Zenodo doi:10.5281/zenodo.13325008 (CC-BY-4.0) |
| `motusdb/genome_quality.tsv` | genome size and quality per mOTU | `python fetch_genome_quality.py` (motus-db.org) |
| `motusdb/marine_metadata.tsv` | depth, latitude, size fraction of marine samples | **included**; rebuilt by `python marine_metadata.py` (NCBI BioSample, ENA) |
| `motusdb/gem/_*.tsv` | community tables and SMETANA outputs of the metabolic pilot | **included**; rebuilt by `gem_build.py` and `gem_smetana.py` |
| `motusdb/gem/*.faa.gz`, `*.xml` | protein FASTA and CarveMe models | `gem_pairs.py --stage fetch`, `gem_build.py` |

On the course server the course files can also be read in place:
`python p9_pipeline.py --data /path/to/motus_small_sample/`.
