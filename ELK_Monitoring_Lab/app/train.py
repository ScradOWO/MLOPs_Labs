"""Train the diabetes-progression regressor and save the drift reference.

Runs once at image build time. Writes:
  model/model.joblib      the regressor
  model/reference.json    per-feature quantile bins + training proportions, used by drift_monitor.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

OUT = Path("model")
N_BINS = 10


def bins_and_proportions(values):
    """Decile edges from training data; open-ended outer bins so live values outside the range still count."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1))[1:-1])  # unique: binary `sex` has 1 edge
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return edges.tolist(), (counts / counts.sum()).tolist()


def main():
    data = load_diabetes(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    # numpy, not DataFrame: the API predicts from plain lists, and a name-less fit avoids a warning per request
    model = GradientBoostingRegressor(random_state=42).fit(X_train.to_numpy(), y_train)
    pred = model.predict(X_test.to_numpy())

    reference = {name: dict(zip(["edges", "proportions"], bins_and_proportions(X_train[name].to_numpy())))
                 for name in data.feature_names}
    reference["prediction"] = dict(zip(["edges", "proportions"], bins_and_proportions(model.predict(X_train.to_numpy()))))

    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")
    (OUT / "reference.json").write_text(json.dumps(reference))
    print(f"test MAE {mean_absolute_error(y_test, pred):.1f}, R2 {r2_score(y_test, pred):.3f}")


if __name__ == "__main__":
    main()
