"""Deploy the sensitivity app for one topic and spawn its jobs (survive client exit).
Usage: MODAL_PROFILE=<acct> TOPIC=<1|2> MODAL_IMAGE_BUILDER_VERSION=2025.06 python modal_jobs/submit_sensitivity.py"""
import os
import subprocess
import sys

import modal

topic = os.environ["TOPIC"]
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, "-m", "modal", "deploy", os.path.join(here, "modal_sensitivity.py")], check=True)
app = f"bioinfo-sens-topic{topic}"
calls = []
only = os.environ.get("ONLY_PREVALENCE")  # respawn only the analysis jobs
if topic == "1":
    calls.append(("t1_prevalence", modal.Function.from_name(app, "t1_prevalence").spawn()))
    for c in [0.5, 0.6, 0.7, 0.8, 0.9]:
        if not only:
            calls.append((f"t1_cutoff_{c}", modal.Function.from_name(app, "t1_cutoff").spawn(c)))
else:
    calls.append(("t2_prevalence", modal.Function.from_name(app, "t2_prevalence").spawn()))
    for n in ["fn_sweep", "donor_sweep", "annotation_sweep", "nsize_20", "nsize_80", "nsize_160"]:
        calls.append((n, modal.Function.from_name(app, "t2_sweep").spawn(n)))
for name, c in calls:
    print("SPAWNED", name, c.object_id)
