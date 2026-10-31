"""Train a wine-cultivar classifier and refuse to save it if it is not good enough.

Runs inside the `trainer` stage of the Dockerfile (and the `train` compose service).
Hyperparameters and the accuracy threshold come from environment variables,
which the Dockerfile fills from build args.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", 200))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", 0)) or None  # 0 means no limit
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", 0.90))
OUT_DIR = Path(os.environ.get("MODEL_DIR", "model"))


def main():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, stratify=wine.target, random_state=42
    )
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)

    # Gate on 5-fold CV accuracy: the 36-row test split alone is too small to trust.
    cv_acc = cross_val_score(model, X_train, y_train, cv=5).mean()
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    metrics = {
        "dataset": "sklearn wine (178 samples, 13 features, 3 cultivars)",
        "model": "RandomForestClassifier",
        "params": {"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH},
        "cv_accuracy": round(cv_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "min_accuracy": MIN_ACCURACY,
        "sklearn_version": sklearn.__version__,
        "feature_names": list(wine.feature_names),
        "class_names": [f"cultivar_{c}" for c in wine.target_names],
    }
    print(json.dumps(metrics, indent=2))

    if cv_acc < MIN_ACCURACY:
        # Non-zero exit fails `docker build` / the compose service, so a weak model never ships.
        sys.exit(f"QUALITY GATE FAILED: cv_accuracy {cv_acc:.4f} < MIN_ACCURACY {MIN_ACCURACY}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "model.joblib")
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"QUALITY GATE PASSED: model saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
