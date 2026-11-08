# Topic 1 results (Modal run r1 + sensitivity analyses)

Final manuscript: BMC_Bioinformatics_Topic1_v4.docx (older drafts in archive_drafts/).
- Primary analysis (cutoff 0.7): scores.csv, meta_evaluation.csv, fusion_weights.csv, run_manifest.json, tables/, figures/ (equal-compound Brier columns)
- Sensitivity: sensitivity/tables (S1-S6), sensitivity/figures (S1-S3, S9), sensitivity/cutoff_runs/, prevalence-adjusted metrics, cell-line subset scores
- Large raw files (predictions, split assignments) remain in outputs/modal_t1 and outputs/sens_t1.
- Regenerate: python modal_jobs/make_report.py; PYTHONPATH=. python modal_jobs/analyze_cellline_novel.py; python modal_jobs/make_sensitivity_report.py 1; python modal_jobs/build_topic1_v4.py (generated from build_topic1_v3.py by make_v4.py; run analyze_extra.py t1 first) (run from modal_jobs/)
- References: modal_jobs/references.json (compiled by modal_jobs/compile_refs.py from Crossref records resolved by exact DOI).
- Extra analyses (Modal account yadalam1980): modal_jobs/modal_extra.py, submit_extra.py, analyze_extra.py; outputs in outputs/extra_t1 and outputs/extra_t2.

## Final submission files (real-data study only)
- BMC_Bioinformatics_Topic1_FINAL.docx — manuscript (4,518 words main text; Times New Roman 12 pt, justified, 1.15 spacing; 7 figures, 8 tables, 37 verified Vancouver references)
- BMC_Bioinformatics_Topic1_FINAL_Additional_file_1.docx — 4 supplementary figures, 10 supplementary tables
- AUDIT_REPORT_FINAL.txt — independent double audit (111 checks)
- Regenerate: cd modal_jobs; python build_final.py; cd ..; python modal_jobs/audit_final.py
