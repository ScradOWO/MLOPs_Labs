import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gate import check  # noqa: E402
from train import evaluate, load_splits, train  # noqa: E402


def test_split_is_stratified_and_deterministic():
    X_train, X_test, y_train, y_test = load_splits()
    X_train2, _, _, _ = load_splits()
    assert np.array_equal(X_train, X_train2)
    assert len(y_test) == 450
    # every digit keeps roughly its share (~10%) of the hold-out set
    assert set(np.bincount(y_test)) <= set(range(43, 48))


def test_evaluate_on_a_perfect_model():
    class Perfect:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.eye(2)[X[:, 0]] * 0.98 + 0.01

    m = evaluate(Perfect(), np.array([[0], [1], [1]]), np.array([0, 1, 1]))
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["log_loss"] < 0.02


@pytest.fixture(scope="module")
def trained():
    return train(n_estimators=50)


def test_train_ships_the_lower_log_loss_candidate(trained):
    _, metrics = trained
    cands = metrics["candidates"]
    assert metrics["log_loss"] == min(c["log_loss"] for c in cands.values())
    assert metrics["shipped"] in cands


def test_trained_model_is_good_enough(trained):
    model, metrics = trained
    assert metrics["macro_f1"] >= 0.90
    _, X_test, _, _ = load_splits()
    assert model.predict(X_test[:5]).shape == (5,)


GOOD = {"macro_f1": 0.95, "log_loss": 0.20}


def test_gate_passes_without_baseline():
    assert all(ok for *_, ok in check(GOOD))


def test_gate_fails_below_min_f1():
    assert not all(ok for *_, ok in check({"macro_f1": 0.80, "log_loss": 0.2}))


@pytest.mark.parametrize("new, ok", [
    ({"macro_f1": 0.945, "log_loss": 0.21}, True),   # within tolerance
    ({"macro_f1": 0.93, "log_loss": 0.20}, False),   # F1 dropped 0.02
    ({"macro_f1": 0.95, "log_loss": 0.25}, False),   # log loss rose 0.05
])
def test_gate_compares_against_baseline(new, ok):
    assert all(r[-1] for r in check(new, baseline=GOOD)) is ok
