"""Quality gate: fail the CI job if the new model is too weak or worse than the released one.

    python src/gate.py build/metrics.json [--baseline baseline/metrics.json] [--min-f1 0.90]

Prints a Markdown report. Inside GitHub Actions it is also appended to the job
summary ($GITHUB_STEP_SUMMARY). Exits 1 if any rule fails.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def check(new, baseline=None, min_f1=0.90, max_f1_drop=0.01, max_log_loss_rise=0.02):
    """Return rows of (metric, new value, baseline value, rule, passed)."""
    rows = [("macro_f1", new["macro_f1"], None, f">= {min_f1}", new["macro_f1"] >= min_f1)]
    if baseline:
        rows.append(("macro_f1", new["macro_f1"], baseline["macro_f1"], f"drop <= {max_f1_drop} vs baseline",
                     new["macro_f1"] >= baseline["macro_f1"] - max_f1_drop))
        rows.append(("log_loss", new["log_loss"], baseline["log_loss"], f"rise <= {max_log_loss_rise} vs baseline",
                     new["log_loss"] <= baseline["log_loss"] + max_log_loss_rise))
    return rows


def report(rows, new, baseline):
    lines = [
        "## Model quality gate",
        "",
        f"Shipped model: **{new['shipped']}** (n_estimators={new['n_estimators']}), "
        f"accuracy {new['accuracy']}, macro F1 {new['macro_f1']}, log loss {new['log_loss']}",
        "",
        f"Baseline: {'latest model release' if baseline else 'none yet (first release), absolute checks only'}",
        "",
        "| Metric | New | Baseline | Rule | Result |",
        "|---|---|---|---|---|",
    ]
    for metric, value, base, rule, ok in rows:
        lines.append(f"| {metric} | {value} | {'-' if base is None else base} | {rule} | {'PASS' if ok else 'FAIL'} |")
    lines += ["", "| Candidate | Accuracy | Macro F1 | Log loss |", "|---|---|---|---|"]
    for name, m in new["candidates"].items():
        lines.append(f"| {name} | {m['accuracy']} | {m['macro_f1']} | {m['log_loss']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--baseline", type=Path, help="metrics.json of the current production model (optional)")
    parser.add_argument("--min-f1", type=float, default=0.90)
    args = parser.parse_args()

    new = json.loads(args.metrics.read_text())
    baseline = json.loads(args.baseline.read_text()) if args.baseline and args.baseline.exists() else None
    rows = check(new, baseline, min_f1=args.min_f1)
    text = report(rows, new, baseline)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)

    if not all(ok for *_, ok in rows):
        sys.exit("QUALITY GATE FAILED")
    print("QUALITY GATE PASSED")


if __name__ == "__main__":
    main()
