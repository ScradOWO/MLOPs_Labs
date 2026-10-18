# Data Labeling Lab: Labeling-Function Quality Gate for Snorkel

Modified version of the MLOps course Data Labeling lab (`Data_Labs/Data_Labeling_Labs/01_spam_tutorial.ipynb`). That lab uses [Snorkel](https://snorkel.ai) weak supervision to label YouTube comments as SPAM or HAM.

In the original pipeline, a hand-picked list of labeling functions (LFs) goes straight into Snorkel's `LabelModel`. This lab adds a **quality-gate stage** in between. The gate screens a larger pool of LFs, drops the ones that would hurt the training labels, and passes only the rest to the `LabelModel`.

## What changed from the original lab

This lab adds a labeling-function quality gate to the Snorkel pipeline. The gate does three things before any LF reaches the `LabelModel`:

- it drops LFs that are inaccurate on a small hand-labeled dev set,
- it drops LFs that have too little dev evidence to judge,
- it drops LFs that duplicate another LF.

To exercise the gate, the candidate pool grows from the original 10 LFs to 20. The results section below compares the gated pipeline against the original lab.

Sections 1–5 of the notebook are still the original tutorial, which serves as the baseline. The change is all in Section 6. Smaller cleanups: `utils.py` is trimmed to the one function the notebook uses, so TensorFlow is no longer needed. The notebook's `pip install` cells are removed, and all dependencies are pinned in `requirements.txt`.

### How the quality gate works

`lf_quality_gate()` (notebook Section 6b) visits the LFs best-first, sorted by dev accuracy and then train coverage. It keeps an LF only if it passes all 3 checks:

1. **Enough evidence:** it votes on at least `min_votes = 3` of the 100 labeled dev comments.
2. **Accurate enough:** its accuracy on those dev votes is at least `min_acc = 0.65`. This threshold was chosen by sweeping it on the valid split (Section 6d).
3. **Not a duplicate:** it is dropped if an LF already kept gives the same vote on every comment they both label **and** the Jaccard overlap of their covered comments is at least `max_jaccard = 0.8`. This check uses only the unlabeled train set.

Why check 3 matters: `LabelModel` assumes LFs are conditionally independent. Without the gate, `keyword_http` and the new `regex_url` fire on almost the same comments. `LabelModel` then gives **both** of them a weight of 0, which drags its accuracy down.

The gate also returns the reason for every LF it drops, so the pipeline is auditable:

```
Kept:    ['keyword_my', 'keyword_subscribe', 'keyword_please', 'keyword_song', 'regex_check_out', 'short_comment', 'textblob_polarity', 'regex_url', 'keyword_channel', 'keyword_money', 'regex_like_my_comment', 'keyword_video', 'keyword_views']
Dropped:
  keyword_http             duplicate of regex_url
  regex_social             only 2 dev votes
  textblob_subjectivity    dev accuracy 0.59
  has_person_nlp           dev accuracy 0.47
  keyword_love             dev accuracy 0.40
  regex_exclaim            dev accuracy 0.33
  regex_timestamp          only 0 dev votes
```

## Requirements

- **OS:** Linux, macOS or Windows. CPU only; no GPU needed.
- **Python 3.12.** snorkel 0.10.0 does not install on Python 3.13 or newer. The steps below use [uv](https://docs.astral.sh/uv/), which downloads Python 3.12 for you, so you do not need to install Python separately.
- About 2 GB of disk space for the virtual environment, and an internet connection for the first install.
- All Python packages are pinned in `requirements.txt`. Key versions: `snorkel==0.10.0`, `torch==2.14.1` (CPU build), `spacy==3.8.16` with `en_core_web_sm` 3.8.0, `textblob==0.20.1`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.4`.

## Setup

### 1. Install uv (one time)

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen the terminal, then check that it works with `uv --version`.

### 2. Create the virtual environment and install dependencies

Run these commands from inside the `Snorkel_Data_Labeling_Lab` folder:

```bash
cd Snorkel_Data_Labeling_Lab

# Create a venv with Python 3.12 (uv downloads 3.12 if you don't have it)
uv venv --python 3.12 .venv

# Activate it
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell or cmd)

# Install the CPU-only PyTorch build first (snorkel needs torch; this avoids ~5 GB of CUDA packages on Linux)
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# Install everything else, including the spaCy English model
uv pip install -r requirements.txt
```

Everything is installed inside `.venv/`. Nothing is installed system-wide. To remove the lab, delete the folder.

<details>
<summary>Without uv (plain pip)</summary>

Install Python 3.12 from [python.org](https://www.python.org/downloads/). Then:

```bash
python3.12 -m venv .venv            # Windows: py -3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
</details>

## Run

With the venv activated, run the whole notebook headless. It takes about 30–60 seconds on a laptop CPU:

```bash
jupyter nbconvert --to notebook --execute --inplace spam_labeling.ipynb
```

Or open it and run all cells interactively:

```bash
jupyter notebook spam_labeling.ipynb      # then Kernel > Restart & Run All
```

## What a successful run looks like

- `nbconvert` finishes with `[NbConvertApp] Writing ... bytes to spam_labeling.ipynb` and shows no traceback.
- The progress bars printed while the LFs are applied are normal.
- **Section 4/5 (baseline, original tutorial)** prints:
  ```
  Majority Vote Accuracy:   83.6%
  Label Model Accuracy:     86.4%
  Test Accuracy: 93.2%
  ```
- **Section 6d** prints the threshold sweep on the valid split. 0.60 and 0.65 tie for best, and anything above 0.65 makes coverage collapse.
- **Section 6** prints the Kept/Dropped list shown above. The two `assert` self-checks pass silently.
- **Section 6e** prints the final comparison table below. Every run gives the same numbers because all seeds are fixed.

## Results

| LF set | # LFs | train coverage | **train label acc** | MV valid | LM valid | MV test | LM test | LogReg test |
|---|---|---|---|---|---|---|---|---|
| original 10 LFs (baseline) | 10 | 0.887 | 0.770 | 0.833 | 0.825 | 0.836 | **0.864** | **0.932** |
| all 20 candidates, no gate | 20 | 0.940 | 0.623 | 0.867 | 0.675 | **0.896** | 0.732 | 0.832 |
| **gated candidates** | 13 | 0.839 | **0.803** | **0.925** | **0.858** | 0.892 | 0.852 | 0.892 |

MV = Snorkel `MajorityLabelVoter`. LM = Snorkel `LabelModel`. LogReg = the final bag-of-n-grams logistic regression trained on the LM labels. "Train label acc" is the accuracy of the generated training labels against the true labels of the train comments. The tutorial hides those labels, and the 100 dev comments are left out of this metric.

What the table shows:

- **Adding LFs blindly hurts.** The pool of 20 LFs labels more comments, but training-label accuracy falls from 77.0% to 62.3%, and `LabelModel` test accuracy falls from 86.4% to 73.2%.
- **The gate fixes that, and beats the original hand-picked set on label quality.** Training labels are 80.3% accurate, against 77.0% for the baseline. Valid-split accuracy goes up for both majority vote (83.3% to 92.5%) and `LabelModel` (82.5% to 85.8%). Majority-vote test accuracy also goes up, from 83.6% to 89.2%.
- **Trade-off:** the gate favors precision over coverage, labeling 83.9% of train comments instead of 88.7%. On the 250-comment test set, `LabelModel` accuracy is about the same (86.4% vs 85.2%). The final classifier is lower (93.2% vs 89.2%) because it trains on fewer comments. Note that the original 10 LFs were hand-tuned by the Snorkel authors on this exact dataset. The gate exists for the realistic case where new, unchecked LFs keep getting added, and the "no gate" row shows the damage it prevents.

## Files

| File | Purpose |
|---|---|
| `spam_labeling.ipynb` | The lab notebook, saved with outputs. Sections 1–5 are the original tutorial, Section 6 is the quality gate |
| `utils.py` | `load_spam_dataset()`: loads the 5 YouTube CSVs and builds the train/dev/valid/test splits |
| `data/Youtube0*.csv` | [YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection) (comments from 5 videos, included) |
| `requirements.txt` | Pinned dependencies |

## Troubleshooting

- **`snorkel` fails to install, or there are numpy/torch build errors:** you are not on Python 3.12. Check with `python --version` inside the venv. Recreate the venv with `uv venv --python 3.12 .venv`.
- **`OSError: [E050] Can't find model 'en_core_web_sm'`:** the spaCy model was not installed. Re-run `uv pip install -r requirements.txt`.
- **The notebook runs on a different kernel than the venv:** start Jupyter from the activated venv. Alternatively, register the venv as a kernel with `python -m ipykernel install --user --name snorkel-lab` and pick `snorkel-lab` in Jupyter.
