"""Per-row AUC lookup so the endpoint cutoff can be varied without re-featurising.
Verified: every prepared row matches, and (auc < 0.7) reproduces the stored label with 0 mismatches."""
import pandas as pd

RAW = "F:/bioinfo_data/topic1_extract/csa_data/raw_data/y_data/response.tsv"
TABLE = "F:/bioinfo_data/topic1_prepared/compound_context.csv"
OUT = "F:/bioinfo_data/topic1_prepared/auc_lookup.csv"

r = pd.read_csv(RAW, sep="\t", usecols=["source", "improve_sample_id", "improve_chem_id", "auc"])
r = r[r.source.isin(["GDSCv2", "CTRPv2"])]
g = r.groupby(["source", "improve_chem_id", "improve_sample_id"], as_index=False).auc.median()
g["sample_id"] = g.source + "|" + g.improve_chem_id + "|" + g.improve_sample_id
t = pd.read_csv(TABLE, usecols=["sample_id", "label"])
m = t.merge(g[["sample_id", "auc"]], on="sample_id", how="left")
assert m.auc.notna().all() and ((m.auc < 0.7).astype(int) == m.label).all()
m[["sample_id", "auc"]].to_csv(OUT, index=False)
print("rows", len(m), "written", OUT)
