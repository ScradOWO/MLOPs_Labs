"""End-to-end check of the running API. Standard library only, so it runs in any Python image.

Usage: python smoke_test.py [base_url]   (default http://localhost:8000)
"""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

# Row 0 of the sklearn wine dataset; its true label is class 0.
SAMPLE = dict(zip(
    ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols",
     "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue",
     "od280/od315_of_diluted_wines", "proline"],
    [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065],
))


def call(path, body=None):
    req = urllib.request.Request(BASE + path)
    if body is not None:
        req.data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


status, body = call("/health")
assert (status, body) == (200, {"status": "ok"}), (status, body)

status, info = call("/model-info")
assert status == 200 and info["cv_accuracy"] >= info["min_accuracy"], info

status, pred = call("/predict", SAMPLE)
assert status == 200, pred
assert pred["predicted_class"] == "cultivar_class_0", pred
assert abs(sum(pred["probabilities"].values()) - 1) < 1e-3, pred

status, err = call("/predict", {"alcohol": 13.0})
assert status == 400 and "missing features" in err["error"], err

print(f"model-info: cv_accuracy={info['cv_accuracy']} test_accuracy={info['test_accuracy']}")
print(f"predict(row 0) -> {pred}")
print("SMOKE TEST PASSED")
