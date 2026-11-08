"""Run one research workflow on Modal.  Select the project with env TOPIC=1|2.

  Topic 1 (account pradeepaiperio): MODAL_PROFILE=pradeepaiperio TOPIC=1 modal run --detach modal_jobs/modal_run.py
  Topic 2 (account pradeep210981):  MODAL_PROFILE=pradeep210981 TOPIC=2 modal run --detach modal_jobs/modal_run.py

Results are written to the Modal volume "bioinfo-results" under /<run name>.
Topic 1 reads /inputs/{compound_context.csv,provenance.json} from the same volume.
"""
import os
import subprocess

import modal

TOPIC = os.environ.get("TOPIC", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.environ.get("RUN_NAME", "modal_20261105")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.2.5", "pandas==2.3.1", "scipy==1.15.2",
                 "scikit-learn==1.8.0", "matplotlib==3.10.1")
    .add_local_dir(f"{ROOT}/research_common", "/app/research_common", ignore=["__pycache__"])
    .add_local_dir(f"{ROOT}/topic1_cell_context", "/app/topic1_cell_context", ignore=["__pycache__", "results"])
    .add_local_dir(f"{ROOT}/topic2_uncertainty", "/app/topic2_uncertainty", ignore=["__pycache__", "results"])
)
vol = modal.Volume.from_name("bioinfo-results", create_if_missing=True)
app = modal.App(f"bioinfo-topic{TOPIC}", image=image)


@app.function(volumes={"/data": vol}, cpu=8, memory=32768, timeout=24 * 3600)
def run(topic: str, run_name: str):
    out = f"/data/{'context_real_gdsc2_ctrp' if topic == '1' else 'uncertainty_research_sim'}_{run_name}"
    if topic == "1":
        cmd = ["python", "-m", "topic1_cell_context.cli", "run",
               "--config", "/app/topic1_cell_context/configs/research.json",
               "--input", "/data/inputs/compound_context.csv",
               "--provenance", "/data/inputs/provenance.json", "--output", out]
    else:
        cmd = ["python", "-m", "topic2_uncertainty.cli", "demo",
               "--config", "/app/topic2_uncertainty/configs/research.json", "--output", out]
    rc = subprocess.run(cmd, cwd="/app", env={**os.environ, "PYTHONPATH": "/app"}).returncode
    vol.commit()
    if rc:
        raise RuntimeError(f"workflow exited {rc}")
    return out


@app.local_entrypoint()
def main():
    print("result:", run.remote(TOPIC, RUN))
