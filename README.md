# Endpoint definition, not holdout design, shapes the cross-study transfer of drug-response model rankings: a multi-cohort meta-evaluation of rank agreement and selection regret

Code, configurations, run manifests, result tables and figures for the manuscript *Endpoint definition, not holdout design, shapes the cross-study transfer of drug-response model rankings: a multi-cohort meta-evaluation of rank agreement and selection regret* (submitted to BMC Bioinformatics).

## Repository layout
- `research_common/`, `topic*/`, `data_preparation/`: analysis package (installable with `pip install -e .`)
- `modal_jobs/`: cloud job scripts (Modal), aggregation scripts, the manuscript builder and the independent audit
- `results/figures/`: every manuscript figure (PNG and SVG), numbered as in the manuscript and Additional file 1
- `results/tables/main`, `results/tables/supplementary`: every manuscript table as CSV, numbered as in the manuscript
- `results/tables/source_data/`: the result tables the manuscript was built from
- `results/provenance/`: run manifests (package versions, SHA-256 hashes of inputs, code and outputs) and the audit report (PASS 118 | FAIL 0)

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
- Fig1: figures\fig1_internal_vs_external_brier.png
- Fig2: figures\fig3_meta_evaluation.png
- Fig3: sensitivity\figures\figS10_seeds50_meta.png
- Fig4: sensitivity\figures\figS1_cutoff_sensitivity.png
- Fig5: sensitivity\figures\figS2_skill_and_recalibration.png
- Fig6: sensitivity\figures\figS9_cellline_subsets.png
- Fig7: sensitivity\figures\figS11_replication_cohorts.png
- FigS1: figures\fig2_external_brier_by_model.png
- FigS2: sensitivity\figures\figS3_rank_agreement_by_metric.png
- FigS3: figures\fig4_fusion_weights.png
- FigS4: external_validity.png

## Tables
- main: Paper1_Table_1.csv, Paper1_Table_2.csv, Paper1_Table_3.csv, Paper1_Table_4.csv, Paper1_Table_5.csv, Paper1_Table_6.csv, Paper1_Table_7.csv, Paper1_Table_8.csv
- supplementary: Paper1_Table_S1.csv, Paper1_Table_S2.csv, Paper1_Table_S3.csv, Paper1_Table_S4.csv, Paper1_Table_S5.csv, Paper1_Table_S6.csv, Paper1_Table_S7.csv, Paper1_Table_S8.csv, Paper1_Table_S9.csv, Paper1_Table_S10.csv

## Citation
If you use this code, please cite the article (details to be added on publication) and this archive (see `CITATION.cff`).

## Licence
Code is released under the MIT licence (see `LICENSE`). Result tables and figures may be reused with attribution.
