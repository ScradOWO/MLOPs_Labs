# GitHub Actions Lab: Tested, Gated and Released ML Models with CI/CD

Modified version of the MLOps course GitHub Actions lab (`Labs/Github_Labs/Lab2`, "Model Training and Versioning" + "Model Calibration"). In the original, every push to `main` trains a random forest on random synthetic data, scores it, and commits the model file and its metrics back into the repository.

This version rebuilds the **workflow itself** as a CI/CD pipeline with three jobs:

1. **Tests** run on a Python matrix.
2. **Train and gate:** the model is trained and calibrated, then compared with the **currently released model**. The run fails if the new model is worse.
3. **Release:** a manual promotion step publishes a passing model as a **GitHub Release**. The latest release is the production model and the baseline the next run must match.

The workflow is [`.github/workflows/github_actions_lab.yml`](../.github/workflows/github_actions_lab.yml) at the repository root (GitHub only runs workflows from there). The lab's code is in this folder.

## What changed from the original lab

| Area | Original Lab2 | This lab |
|---|---|---|
| Jobs | 1 job doing everything | 3 jobs chained with `needs:`: `test` → `train-and-gate` → `release` |
| Tests | none in Lab2 | `pytest` (9 tests) on a **matrix of Python 3.12, 3.13 and 3.14**, with JUnit reports uploaded as artifacts |
| Triggers | every push to `main` | push to `main` **and** pull requests, both only when this lab's files change (`paths:` filter), plus a manual **`workflow_dispatch`** with inputs (`n_estimators`, `min_f1`, `release`) |
| Quality gate | none: any model is committed | `src/gate.py` fails the job if macro F1 < `min_f1`, or if the model is worse than the **latest released model** (F1 drop > 0.01 or log-loss rise > 0.02) |
| Gate report | none | Markdown table written to the **job summary** (`$GITHUB_STEP_SUMMARY`) on the run page, and reused as the release notes |
| Model storage / versioning | `git commit` + `git push` of `.joblib` files into the repo from CI (the repo grows on every push) | **workflow artifact** on every run, and a **GitHub Release** `model-<run number>` with `model.joblib` + `metrics.json` only when you choose to promote |
| Baseline | none | downloaded at run time from the latest `model-*` release with `gh release download` |
| Permissions | default token, pushes to `main` | `contents: read` for everything, `contents: write` only in the `release` job |
| Repeated runs | every push runs to the end | `concurrency:` cancels an older push/PR run on the same branch; manual runs share one queue and run in order, so a release always finishes before the next manual run reads its baseline |
| Speed | dependencies reinstalled each time | pip cache keyed on `requirements.txt` |
| Action versions | `checkout@v2/v4`, `setup-python@v2/v4`, Python 3.9 | `checkout@v7`, `setup-python@v7`, `upload-artifact@v7`, `download-artifact@v8`, Python 3.12 |
| Data | `make_classification` with `n_samples=random.randint(0, 2000)`, which can be 0 and crash | scikit-learn's bundled **digits** dataset (1,797 images of handwritten 0–9), fixed stratified split |
| Evaluation | F1 measured on the **same** synthetic data the model trained on (`random_state=0` both times) | metrics measured on a **held-out** 25% split (450 images) the model never saw |
| Calibration | described in the README as a separate workflow | built in: trains a raw forest and a sigmoid-calibrated one (`CalibratedClassifierCV`) and ships the one with the lower log loss |
| Model | `RandomForestClassifier` (binary) | `RandomForestClassifier` (10 classes) + probability calibration |

### Pipeline

```
push / pull request / manual run
        │
        ▼
 test  (Python 3.12 │ 3.13 │ 3.14 in parallel)   pytest, JUnit report artifact
        │ all pass
        ▼
 train-and-gate                                  python src/train.py  → build/model.joblib, metrics.json
   ├─ fetch baseline: metrics.json of latest model-* release (if any)
   ├─ python src/gate.py  → job summary table, exit 1 on FAIL
   └─ upload build/ as artifact "model-<run number>"
        │ gate passed AND manual run with release = true
        ▼
 release                                         gh release create model-<run number>
                                                 (becomes the next run's baseline)
```

## Requirements

**To run the CI/CD pipeline** (the core of the lab), you need nothing installed locally:

- A **GitHub account** (free). GitHub-hosted runners are free for public repositories.
- A copy of this repository in your account (fork or push, see below).

**To run the same steps on your own machine** (optional):

- **Git**: <https://git-scm.com/downloads>
- **uv**, which also installs Python 3.12 for you: <https://docs.astral.sh/uv/getting-started/installation/>
- Optional: **GitHub CLI** `gh` 2.40+ (<https://cli.github.com>) to start runs and download artifacts from the terminal.
- Pinned packages (`requirements.txt`): `scikit-learn==1.9.1`, `joblib==1.6.0`, `numpy==2.5.4`, `pytest==9.2.0`. They work on Python 3.12–3.14 (numpy 2.5 dropped 3.11).

Tested locally on Arch Linux: the test suite passes on Python 3.12.15, 3.13 and 3.14.10, and `train.py` + `gate.py` were run with the same commands and `GITHUB_STEP_SUMMARY` set as in CI. The workflow itself has to run on GitHub to be checked (see Running B).

## Setup

### 1. Get the repository into your GitHub account

Either **fork** `https://github.com/ScradOWO/MLOPs_Labs` (button at the top right of the repo page), or clone it and push it to a new repo of yours:

```bash
git clone https://github.com/ScradOWO/MLOPs_Labs.git
cd MLOPs_Labs
git remote set-url origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### 2. Enable Actions

In your copy on GitHub, open the **Actions** tab. On a fork, GitHub asks you to confirm before running workflows: click **"I understand my workflows, go ahead and enable them"**. No secrets or other settings are needed. The workflow only uses the built-in `GITHUB_TOKEN`.

### 3. (Optional) Local environment

```bash
cd GitHub_Actions_Lab
uv venv --python 3.12 .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish   Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Running the lab

### A. Run it locally (same commands CI runs)

From `GitHub_Actions_Lab/` with the venv active:

```bash
pytest -q                                   # 9 passed
python src/train.py --out build             # trains, prints metrics, writes build/
python src/gate.py build/metrics.json       # QUALITY GATE PASSED
```

Expected training result (every run is identical, all seeds are fixed):

| Candidate | Accuracy | Macro F1 | Log loss |
|---|---|---|---|
| raw random forest | 0.9667 | 0.9664 | 0.3302 |
| **calibrated** (shipped) | **0.9711** | **0.9709** | **0.1620** |

Calibration halves the log loss, so the probabilities the model reports are much more trustworthy, and accuracy goes up slightly too.

To see the gate fail, train a 1-tree forest and compare it against the good model:

```bash
python src/train.py --out build_weak --n-estimators 1
python src/gate.py build_weak/metrics.json --baseline build/metrics.json   # exit code 1
```

```
| macro_f1 | 0.8835 | 0.9709 | drop <= 0.01 vs baseline | FAIL |
| log_loss | 0.7125 | 0.162 | rise <= 0.02 vs baseline | FAIL |
QUALITY GATE FAILED
```

### B. Run it on GitHub Actions

**1. Automatic run on push.** Any commit that touches `GitHub_Actions_Lab/` or the workflow file starts it, for example:

```bash
echo "# trigger" >> GitHub_Actions_Lab/src/train.py
git commit -am "Trigger the GitHub Actions lab" && git push
```

Open **Actions → "GitHub Actions Lab - test, train, gate, release"**. A successful run shows three green test jobs (one per Python version) and a green **Train, compare with released model, gate** job. **Release model** is shown as skipped, because releases are manual. The run page's summary shows the gate table, and the **Artifacts** section lists `model-<N>` and the three `test-results-py3.1x` files.

**2. Release the model (manual promotion).** **Actions → the workflow → Run workflow**, tick **"Publish the model as a GitHub Release if the gate passes"**, and click **Run workflow**. Or from a terminal:

```bash
gh workflow run github_actions_lab.yml -f release=true
```

When it finishes, the repo's **Releases** page has `model-<N>` with `model.joblib`, `metrics.json` and the gate report as release notes. From now on, every run is compared against this release.

**3. Watch the gate block a regression.** Start a run with a deliberately weak model. You can start it right after step 2: manual runs queue, so it waits for the release to finish and then compares against it.

```bash
gh workflow run github_actions_lab.yml -f n_estimators=1 -f release=true
```

The **train-and-gate** job fails (red) with `QUALITY GATE FAILED`, and the summary table shows which checks failed against the released baseline. **Release model** never runs, so the bad model cannot reach Releases even though `release=true` was requested.

From a terminal you can follow runs with:

```bash
gh run list --workflow github_actions_lab.yml
gh run watch                     # live view of the latest run
gh run view <run-id> --log-failed
```

## Results

Training is deterministic (fixed seeds and split), so CI produces exactly the local numbers above. The three runs from "Run it on GitHub Actions" should end like this:

| Run | `test` (×3) | `train-and-gate` | `release` | Outcome |
|---|---|---|---|---|
| 1. push to `main` | pass | pass: macro F1 0.9709 ≥ 0.90, no baseline yet | skipped (not a manual run) | artifact `model-<N>` |
| 2. manual, `release=true` | pass | pass | **runs** | Release `model-<N>` with `model.joblib`, `metrics.json`, gate report |
| 3. manual, `n_estimators=1`, `release=true` | pass | **fails**: F1 0.8835 vs released 0.9709, log loss 0.7125 vs 0.162 | skipped (gate failed) | no new release; production model unchanged |

Run 3 is the point of the lab. A model that is 9 F1 points worse is stopped automatically, even though the person starting the run asked for it to be released.

## Files

| Path | Purpose |
|---|---|
| `../.github/workflows/github_actions_lab.yml` | The CI/CD workflow (must live at the repo root) |
| `src/train.py` | Loads digits, trains raw + calibrated forests, writes `model.joblib` and `metrics.json` |
| `src/gate.py` | Quality gate: absolute F1 floor + no-regression check against the released baseline; writes the Markdown report |
| `tests/test_pipeline.py` | 9 unit tests: split, metrics, model selection, gate rules |
| `requirements.txt` | Pinned dependencies |

## Troubleshooting

- **The workflow doesn't appear in the Actions tab:** workflows only load from `.github/workflows/` at the repository root, and on forks Actions must be enabled first (Setup step 2).
- **"Run workflow" button missing:** `workflow_dispatch` only shows up once the workflow file is on the default branch (`main`).
- **A push didn't trigger a run:** the `paths:` filter only starts runs for changes under `GitHub_Actions_Lab/` or to the workflow file itself.
- **`release` job fails with `HTTP 403: Resource not accessible by integration`:** your organization caps the token at read-only. In the repo, go to **Settings → Actions → General → Workflow permissions** and choose **Read and write permissions**.
- **Gate fails on the first ever run:** with no release yet, only the absolute check (`macro F1 >= 0.90`) applies. Check `min_f1` wasn't set above what the model reaches (about 0.97).
