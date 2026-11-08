# Endpoint definition, not holdout design, shapes the cross-study transfer of drug-response model rankings: a multi-cohort meta-evaluation of rank agreement and selection regret

Code, configurations, run manifests, result tables and figures for the manuscript *Endpoint definition, not holdout design, shapes the cross-study transfer of drug-response model rankings: a multi-cohort meta-evaluation of rank agreement and selection regret* (submitted to BMC Bioinformatics).

## Repository layout
- `research_common/`, `topic*/`, `data_preparation/`: analysis package (installable with `pip install -e .`)
- `modal_jobs/`: cloud job scripts (Modal), aggregation scripts, the manuscript builder and the independent audit
- `results/figures/`: every manuscript figure (PNG and LZW TIFF at 600 dpi, vector PDF), numbered as in the manuscript and Additional file 1; redrawn by `modal_jobs/figures_paper*.py`
- `results/tables/main`, `results/tables/supplementary`: every manuscript table as CSV, numbered as in the manuscript
- `results/tables/source_data/`: the result tables the manuscript was built from
- `results/provenance/`: run manifests (package versions, SHA-256 hashes of inputs, code and outputs) and the audit report (PASS 128 | FAIL 0)

## Data
The input data are the public cross-study drug-response benchmark of Partin et al., deposited at Zenodo (record 15258883, https://doi.org/10.5281/zenodo.15258883). They are not redistributed here. `data_preparation/prepare_gdsc2_ctrp.py` builds the analysis table and its provenance record from the extracted archive.

## Reproducing the analysis
Python 3.12 was used (see the run manifests for exact package versions).

```
pip install -e ".[chemistry]"
python data_preparation/prepare_gdsc2_ctrp.py <extracted_zenodo_dir> <prepared_dir>
context-audit --help                         # primary pipeline (configuration: topic1_cell_context/configs/research.json)
python modal_jobs/build_final.py             # rebuild the manuscript tables/figures from result files
python modal_jobs/audit_final.py             # independent audit (needs the raw run outputs)
```

## Figure mapping (manuscript number: original result file)
- Fig1: figures_pub\Fig1_study_design.png
- Fig2: figures_pub\Fig2_primary_analysis.png
- Fig3: figures_pub\Fig3_meta_evaluation_seeds.png
- Fig4: figures_pub\Fig4_endpoint_cutoff.png
- Fig5: figures_pub\Fig5_prevalence_adjusted.png
- Fig6: figures_pub\Fig6_cellline_overlap.png
- Fig7: figures_pub\Fig7_replication.png
- FigS1: figures_pub\FigS1_fusion_weights.png
- FigS2: external_validity.png

## Tables
- main: Paper1_Table_1.csv, Paper1_Table_2.csv, Paper1_Table_3.csv, Paper1_Table_4.csv, Paper1_Table_5.csv, Paper1_Table_6.csv
- supplementary: Paper1_Table_S1.csv, Paper1_Table_S2.csv, Paper1_Table_S3.csv, Paper1_Table_S4.csv, Paper1_Table_S5.csv, Paper1_Table_S6.csv, Paper1_Table_S7.csv, Paper1_Table_S8.csv, Paper1_Table_S9.csv, Paper1_Table_S10.csv, Paper1_Table_S11.csv

## Citation
If you use this code, please cite the article (details to be added on publication) and this archive (see `CITATION.cff`).

## Licence
Code is released under the MIT licence (see `LICENSE`). Result tables and figures may be reused with attribution.
