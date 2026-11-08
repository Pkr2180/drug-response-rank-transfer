"""Deploy the app, then spawn the job so it survives this client exiting.
Usage: MODAL_PROFILE=<acct> TOPIC=<1|2> RUN_NAME=<new unique name> python modal_jobs/submit.py"""
import os
import subprocess
import sys

import modal

topic, run = os.environ["TOPIC"], os.environ["RUN_NAME"]
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, "-m", "modal", "deploy", os.path.join(here, "modal_run.py")], check=True)
call = modal.Function.from_name(f"bioinfo-topic{topic}", "run").spawn(topic, run)
print("SPAWNED", call.object_id)
