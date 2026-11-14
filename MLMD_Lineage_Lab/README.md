# ML Metadata Lab: Persistent Lineage, Cached Steps and Failure Records for a Training Pipeline

Modified version of the MLOps course ML Metadata lab (`Labs/MLMD_Labs/Lab1`, `C2_W3_Lab_1_MLMetadata.ipynb`). The original lab uses [ML Metadata (MLMD)](https://github.com/google/ml-metadata) by hand on a single step. It creates an in-memory store, records one TFDV schema-generation execution on the Chicago Taxi data, then traces the schema back to its dataset in a few manual queries.

This version uses MLMD the way a real pipeline does. A three-step training pipeline records every run in a **persistent** store. The Trainer **asks the store** whether it has already done the same work and reuses the earlier model. A run that fails validation is **recorded as FAILED**. A small CLI answers the lineage questions from the course README across all runs: *which dataset trained this model?* and *which models were built from this (bad) dataset?*

## What changed from the original lab

| Area | Original Lab1 | This lab |
|---|---|---|
| Metadata store | `fake_database`: in memory, gone when the notebook stops | SQLite file `metadata/mlmd.sqlite`: keeps every run, so queries can span runs |
| What is tracked | 1 execution (schema generation) | a 3-step pipeline: `SchemaGen` → `DataValidation` → `Trainer`, run many times |
| Writing metadata | separate `put_artifacts`, `put_executions`, `put_events`, `put_attributions_and_associations` calls | one `put_execution()` per step, which writes the execution, its artifacts, input/output events and context links **in one transaction** |
| Execution state | custom `state` string property (`"RUNNING"`, `"COMPLETED"`) | MLMD's built-in `last_known_state` (`COMPLETE`, `FAILED`, `CACHED`) |
| Contexts | one hand-made `Experiment` context | one `PipelineRun` context per run (`run-001`, `run-002`, …), with the run's data version and params |
| Caching | none | the Trainer looks up earlier `COMPLETE` Trainer executions with the **same input artifacts and params**. If one exists and its model file is still on disk, it records a `CACHED` execution and reuses that model instead of training again |
| Dataset identity | a new artifact every time | DataSet artifacts are deduplicated by SHA-256 of the file. The same data is the same artifact, which is what makes caching and impact analysis work |
| Schema | inferred and printed | inferred **once** from the first dataset, stored as a `Schema` artifact, then loaded from the store by every later run to validate new data |
| Failures | not covered | corrupted data produces a `FAILED` `DataValidation` execution and a `ValidationReport` artifact that lists the anomalies, and the run stops before training |
| Lineage queries | three manual cells for one artifact | `lineage.py` CLI: `runs` (history), `trace` (upstream: what produced a model), `impact` (downstream: what a dataset fed into) |
| Dataset / model | Chicago Taxi CSV (download) + TFDV | scikit-learn breast-cancer data (bundled, no download) in 3 simulated versions + a `RandomForestClassifier` |
| Dependencies | TensorFlow + TFDV | `ml-metadata`, scikit-learn, pandas, all pinned. No TensorFlow; MLMD works on its own |
| Testing | none | `check_lab.py` runs the whole scenario against a throwaway store and asserts what MLMD recorded |

Two more things worth knowing:

- **`filter_query` no longer works.** The course README's `ListOptions(filter_query=...)` examples fail in ml-metadata 1.21.0 with `UnimplementedError: filter_query is not supported - ZetaSQL dependency removed`. This lab queries by type (`get_executions_by_type`, `get_artifacts_by_type`) and filters in Python instead.
- **TFDV is replaced by a small pandas schema check.** The original lab used TFDV only to make the schema. The schema here records, for each column, its dtype, whether nulls are allowed and whether values must be non-negative. That keeps the install small and the focus on metadata.

### The pipeline and what each run records

```
            DataSet (v1 | v2 | v3, deduplicated by file hash)
               │
  first run ───┼──▶ SchemaGen ──▶ Schema   (learned once, reused by every later run)
               │                    │
               ├────────────────────┴──▶ DataValidation ──▶ ValidationReport
               │                    │       └─ FAILED on anomalies → run stops here
               └────────────────────┴──▶ Trainer ──▶ Model
                                            └─ CACHED if same DataSet + Schema + n_estimators
                                               were already trained and the model file exists
  All of a run's executions and artifacts are grouped under one PipelineRun context.
```

The three dataset versions simulate data arriving over time:

| Version | Rows | Content |
|---|---|---|
| `v1` | 341 | first data drop: a random 60% of the 569 rows |
| `v2` | 569 | all rows |
| `v3` | 569 | all rows, but 5 `mean radius` values set to -1 and 5 `mean texture` values set to missing (simulated bad upstream data) |

## Requirements

- **OS:** one of
  - **Linux x86_64 with glibc 2.39 or newer**, for example Ubuntu 24.04+, Debian 13+, Fedora 40+ or Arch. The `ml-metadata` wheel is built for `manylinux_2_39`. Ubuntu 22.04 (glibc 2.35) cannot install it; use the Docker option below.
  - **macOS on Apple Silicon** (M1 or newer).
  - **Windows:** `ml-metadata` has no Windows build. Use WSL2 with Ubuntu 24.04 (install with `wsl --install -d Ubuntu-24.04` in an admin PowerShell, then follow the Linux steps inside Ubuntu), or the Docker option below.
- **Python 3.10 to 3.13.** The steps below use **Python 3.12** through [uv](https://docs.astral.sh/uv/), which downloads it for you, so you do not need to install Python yourself.
- About 300 MB of disk for the virtual environment, and an internet connection for the first install.
- Pinned packages (`requirements.txt`): `ml-metadata==1.21.0`, `scikit-learn==1.9.1`, `pandas==3.1.1`, `numpy==2.5.4`, `protobuf==6.33.6`.

Tested on Arch Linux (CachyOS) x86_64 with Python 3.12.15 and uv.

## Setup

### 1. Install uv (one time)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal, then check that it works with `uv --version`.

### 2. Get the code

```bash
git clone https://github.com/ScradOWO/MLOPs_Labs.git
cd MLOPs_Labs/MLMD_Lineage_Lab
```

Run every command below from inside the `MLMD_Lineage_Lab` folder.

### 3. Create the virtual environment and install dependencies

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Check the install:

```bash
python -c "import ml_metadata; print(ml_metadata.__version__)"    # 1.21.0
```

Everything goes into `.venv/`. Nothing is installed system-wide.

<details>
<summary>Without uv (plain pip)</summary>

With Python 3.12 already installed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
</details>

<details>
<summary>Docker option (any OS with Docker, including Windows and Ubuntu 22.04)</summary>

This runs the self-check in a `python:3.12-slim` container. The lab folder is mounted read-only and copied inside the container, so nothing is written to your machine and nothing is installed on it apart from Docker.

```bash
docker run --rm -v "$PWD":/src:ro python:3.12-slim sh -c \
  "cp -r /src /lab && cd /lab && pip install -q --root-user-action=ignore -r requirements.txt && python check_lab.py"
```

For the step-by-step demo, open a shell in the container instead (`-it` and `bash` at the end) and run the commands from the next section:

```bash
docker run --rm -it -v "$PWD":/src:ro python:3.12-slim sh -c \
  "cp -r /src /lab && cd /lab && pip install -q --root-user-action=ignore -r requirements.txt && bash"
```

Everything the demo writes disappears when you `exit` the shell.
</details>

## Running the lab

### Quick check (one command)

```bash
python check_lab.py
```

This runs the full scenario below against a temporary store, then asserts what MLMD recorded: the Trainer states, the cache hit, the single failed validation, artifact counts, and the `trace` and `impact` output. It takes about 5 seconds. Success is one line:

```
ALL CHECKS PASSED (scratch store: /tmp/mlmd_lab_...)
```

### Step-by-step demo

Run these five pipeline passes in order. Each one is a new `PipelineRun` in `metadata/mlmd.sqlite`.

```bash
python pipeline.py --data-version v1                    # run-001: learns the schema, trains Model #4
python pipeline.py --data-version v2                    # run-002: new data, trains Model #7
python pipeline.py --data-version v2                    # run-003: same inputs -> Trainer CACHED, reuses #7
python pipeline.py --data-version v2 --n-estimators 25  # run-004: new params -> trains Model #10
python pipeline.py --data-version v3                    # run-005: bad data -> FAILED, exits with code 1
```

The IDs above (`#4`, `#7`, `#10`) are what you get on a fresh store. To start over, delete `metadata/` and `artifacts/` together:

```bash
rm -rf metadata artifacts
```

**What success looks like.** Run 3 does not train:

```
== run-003: data v2, n_estimators=100
Schema         reusing Schema #2 from the store
DataValidation passed (DataSet #5, 569 rows)
Trainer        CACHED: same inputs and params already trained, reusing Model #7
Result         Model #7 test accuracy 0.958
```

Run 5 stops at validation. The `Pipeline stopped` line and exit code 1 are the expected result:

```
== run-005: data v3, n_estimators=100
Schema         reusing Schema #2 from the store
DataValidation FAILED with 2 anomalies:
                 - mean radius: 5 negative values
                 - mean texture: 5 missing values
Pipeline stopped: run-005 recorded as FAILED, no model trained.
```

### Querying the lineage

**Run history:** every run, each step's state, and the model it produced.

```bash
python lineage.py runs
```

```
run-001  data=v1  n_estimators=100  SchemaGen:COMPLETE, DataValidation:COMPLETE, Trainer:COMPLETE -> Model #4 acc=0.9767
run-002  data=v2  n_estimators=100  DataValidation:COMPLETE, Trainer:COMPLETE               -> Model #7 acc=0.958
run-003  data=v2  n_estimators=100  DataValidation:COMPLETE, Trainer:CACHED                 -> Model #7 acc=0.958
run-004  data=v2  n_estimators=25   DataValidation:COMPLETE, Trainer:COMPLETE               -> Model #10 acc=0.951
run-005  data=v3  n_estimators=100  DataValidation:FAILED                                   -> no model
```

**Upstream trace:** *which data and schema produced this model?* With no ID, it traces the newest model.

```bash
python lineage.py trace        # or: python lineage.py trace 7
```

```
Model #10 accuracy=0.951 n_estimators=25  [artifacts/models/run-004.joblib]
  <- Trainer #9 COMPLETE in run-004
    DataSet #5 rows=569 version=v2  [artifacts/data/v2.csv]
    Schema #2 columns=31  [artifacts/schema/schema.json]
      <- SchemaGen #1 COMPLETE in run-001
        DataSet #1 rows=341 version=v1  [artifacts/data/v1.csv]
```

This shows something the original single-step lab could not: Model #10 was trained on `v2`, but it was validated against a schema learned from `v1` three runs earlier.

**Downstream impact:** *if `v2` turns out to be bad, which models must be recalled?*

```bash
python lineage.py impact v2
```

The output ends with:

```
Models built from v2: #7, #10
```

`python lineage.py impact v3` ends with `Models built from v3: none`, which shows the validation gate kept the corrupted data out of every model.

## Files

| File | Purpose |
|---|---|
| `store.py` | Opens the SQLite store and registers the artifact, execution and context types |
| `pipeline.py` | Runs one pipeline pass (`--data-version`, `--n-estimators`) and records every step in MLMD |
| `lineage.py` | Lineage CLI: `runs`, `trace [MODEL_ID]`, `impact DATA_VERSION` |
| `check_lab.py` | End-to-end self-check on a temporary store |
| `requirements.txt` | Pinned dependencies |
| `metadata/`, `artifacts/` | Created on first run (git-ignored): the MLMD database, and the datasets, schema, validation reports and models it points to |

Two environment variables move the outputs, for example to keep separate experiments: `MLMD_DB` (store file, default `metadata/mlmd.sqlite`) and `ARTIFACT_DIR` (default `artifacts/`).

## Troubleshooting

- **`No matching distribution found for ml-metadata==1.21.0`**: your platform has no wheel. That happens on Windows, Intel Macs, Linux with glibc older than 2.39 (check with `ldd --version`), or Python outside 3.10–3.13. Use WSL2 with Ubuntu 24.04 or the Docker option.
- **`... is recorded in the store but missing on disk`**: `artifacts/` was deleted but `metadata/` was not, so the store points to files that no longer exist. Run `rm -rf metadata artifacts` and start again.
- **A deleted model file is not reused:** this is intended. The cache only reuses a model whose file still exists, so `rm artifacts/models/run-002.joblib` followed by a v2 run retrains.
- **Different IDs from the README:** the store already has earlier runs. Reset with `rm -rf metadata artifacts`.
