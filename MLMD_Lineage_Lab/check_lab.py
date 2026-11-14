"""End-to-end self-check: run the demo scenario against a throwaway store and assert the recorded metadata.

    python check_lab.py

Uses a temporary directory, so it never touches metadata/ or artifacts/.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp(prefix="mlmd_lab_"))
os.environ["MLMD_DB"] = str(tmp / "mlmd.sqlite")
os.environ["ARTIFACT_DIR"] = str(tmp / "artifacts")

from store import connect, mlmd  # noqa: E402  (must import after MLMD_DB is set)

STATE = mlmd.Execution.State


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=LAB_DIR, capture_output=True, text=True)


# Same scenario as the README.
for args, expect_ok in [
    (["--data-version", "v1"], True),
    (["--data-version", "v2"], True),
    (["--data-version", "v2"], True),                        # identical inputs -> cache hit
    (["--data-version", "v2", "--n-estimators", "25"], True),  # new params -> retrain
    (["--data-version", "v3"], False),                       # corrupted data -> validation fails
]:
    r = run("pipeline.py", *args)
    assert (r.returncode == 0) == expect_ok, (args, r.stdout, r.stderr)

store = connect()
trainers = sorted(store.get_executions_by_type("Trainer"), key=lambda e: e.id)
assert [t.last_known_state for t in trainers] == [STATE.COMPLETE, STATE.COMPLETE, STATE.CACHED, STATE.COMPLETE], trainers
validations = [v.last_known_state for v in store.get_executions_by_type("DataValidation")]
assert validations.count(STATE.FAILED) == 1, validations
assert len(store.get_artifacts_by_type("Model")) == 3, "the cache hit must not create a 4th model"
assert len(store.get_artifacts_by_type("DataSet")) == 3, "re-running v2 must reuse its DataSet artifact"
assert len(store.get_artifacts_by_type("Schema")) == 1, "the schema is learned once"

impact = run("lineage.py", "impact", "v2")
assert impact.returncode == 0 and "Models built from v2: #7, #10" in impact.stdout, impact.stdout
impact_bad = run("lineage.py", "impact", "v3")
assert "Models built from v3: none" in impact_bad.stdout, impact_bad.stdout
trace = run("lineage.py", "trace", "7")
assert "<- Trainer" in trace.stdout and "version=v2" in trace.stdout and "SchemaGen" in trace.stdout, trace.stdout

print(f"ALL CHECKS PASSED (scratch store: {tmp})")
