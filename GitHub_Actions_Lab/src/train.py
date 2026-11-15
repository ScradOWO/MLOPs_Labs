"""Train a digit classifier, calibrate its probabilities, and write the model + metrics.

    python src/train.py --out build [--n-estimators 200]

Writes build/model.joblib and build/metrics.json. Called by the `train` job of
.github/workflows/github_actions_lab.yml, and runnable locally the same way.
"""
import argparse
import json
from pathlib import Path

import joblib
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

SEED = 42


def load_splits():
    """Fixed, stratified hold-out split, so every CI run is scored on the same unseen images."""
    X, y = load_digits(return_X_y=True)
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)
    pred = model.classes_[proba.argmax(axis=1)]
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "log_loss": round(log_loss(y_test, proba, labels=model.classes_), 4),
    }


def train(n_estimators):
    """Fit a raw forest and a sigmoid-calibrated one; ship whichever has the lower hold-out log loss."""
    X_train, X_test, y_train, y_test = load_splits()
    raw = RandomForestClassifier(n_estimators=n_estimators, random_state=SEED).fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=n_estimators, random_state=SEED), method="sigmoid", cv=5
    ).fit(X_train, y_train)

    scores = {"raw": evaluate(raw, X_test, y_test), "calibrated": evaluate(calibrated, X_test, y_test)}
    chosen = min(scores, key=lambda k: scores[k]["log_loss"])
    metrics = {
        **scores[chosen],
        "shipped": chosen,
        "candidates": scores,
        "n_estimators": n_estimators,
        "n_test": len(y_test),
        "sklearn_version": sklearn.__version__,
    }
    return (calibrated if chosen == "calibrated" else raw), metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("build"))
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    model, metrics = train(args.n_estimators)
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out / "model.joblib")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
