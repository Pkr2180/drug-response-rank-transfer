"""Deploy bioinfo-extra and spawn jobs (survive client exit).
Usage: MODAL_PROFILE=yadalam1980 MODAL_IMAGE_BUILDER_VERSION=2025.06 python modal_jobs/submit_extra.py <t2|t1seeds|t1rep>"""
import os
import subprocess
import sys

import modal

group = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, "-m", "modal", "deploy", os.path.join(here, "modal_extra.py")], check=True)
app = "bioinfo-extra"
calls = []
if group == "t2":
    for n in ["fn_interaction", "fn_ptm_dominant", "donor_interaction", "donor_ptm_dominant", "fn_linear_b", "donor_linear_b"]:
        calls.append((n, modal.Function.from_name(app, "t2_run").spawn(n)))
elif group == "t1seeds":
    sets = {"A": [103, 107, 109, 113, 127, 131, 137, 139, 149, 151], "B": [157, 163, 167, 173, 179, 181, 191, 193, 197, 199],
            "C": [211, 223, 227, 229, 233, 239, 241, 251, 257, 263], "D": [269, 271, 277, 281, 283, 293, 307, 311, 313, 317]}
    for k, seeds in sets.items():
        calls.append((f"seeds_{k}", modal.Function.from_name(app, "t1_run").spawn("/data/inputs/compound_context.csv", "/data/inputs/provenance.json", f"seeds_{k}", seeds)))
elif group == "t1rep":
    for name in ["rev_ctrp_gdsc2", "gdsc2_gdsc1"]:
        calls.append((name, modal.Function.from_name(app, "t1_run").spawn(f"/data/inputs/{name}/compound_context.csv", f"/data/inputs/{name}/provenance.json", name)))
for n, c in calls:
    print("SPAWNED", n, c.object_id)
