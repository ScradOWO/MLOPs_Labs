"""Every INTERVAL seconds, measure input and prediction drift on the last WINDOW seconds of traffic.

Reads recent prediction logs back out of Elasticsearch, compares each feature's
distribution with the training reference using the Population Stability Index
(PSI), and writes one `drift` JSON line per check to /logs/drift.jsonl. Logstash
ships that to the `ml-drift` index, where the Kibana dashboard plots it.

PSI rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, >= 0.2 significant shift (alert).
"""
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import numpy as np

ES_URL = os.environ.get("ES_URL", "http://elasticsearch:9200")
INTERVAL = int(os.environ.get("INTERVAL", 15))
WINDOW = int(os.environ.get("WINDOW", 60))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", 100))
PSI_ALERT = float(os.environ.get("PSI_ALERT", 0.2))
DRIFT_LOG = os.environ.get("DRIFT_LOG", "/logs/drift.jsonl")
EPS = 1e-4  # keeps empty bins from producing log(0)

reference = json.loads(open("model/reference.json").read())


def psi(ref, values):
    counts = np.bincount(np.searchsorted(ref["edges"], values, side="right"), minlength=len(ref["proportions"]))
    actual = counts / counts.sum()
    return float(sum((a - e) * math.log((a + EPS) / (e + EPS)) for a, e in zip(actual, ref["proportions"])))


def recent_predictions():
    body = {"size": 10000, "_source": ["features", "prediction"],
            "query": {"range": {"@timestamp": {"gte": f"now-{WINDOW}s"}}}}
    req = urllib.request.Request(f"{ES_URL}/ml-predictions/_search", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return [h["_source"] for h in json.load(r)["hits"]["hits"]]
    except urllib.error.HTTPError as e:
        if e.code == 404:  # index not created yet: no traffic so far
            return []
        raise


def check():
    docs = recent_predictions()
    if len(docs) < MIN_SAMPLES:
        print(f"{len(docs)} predictions in the last {WINDOW}s (< {MIN_SAMPLES}), skipping", flush=True)
        return None
    scores = {f: round(psi(reference[f], [d["features"][f] for d in docs]), 4)
              for f in reference if f != "prediction"}
    drifted = sorted(f for f, s in scores.items() if s >= PSI_ALERT)
    record = {
        "event_type": "drift",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_s": WINDOW,
        "n": len(docs),
        "psi": scores,
        "prediction_psi": round(psi(reference["prediction"], [d["prediction"] for d in docs]), 4),
        "max_psi": max(scores.values()),
        "drifted_features": drifted,
        "alert": bool(drifted),
    }
    with open(DRIFT_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    status = f"ALERT drift in {drifted}" if drifted else "ok"
    print(f"n={len(docs)} max_psi={record['max_psi']} prediction_psi={record['prediction_psi']} {status}", flush=True)
    return record


if __name__ == "__main__":
    # Self-check: a sample from the reference itself has ~0 PSI, a shifted one does not.
    ref = {"edges": [-1, 0, 1], "proportions": [0.25, 0.25, 0.25, 0.25]}
    same = np.random.default_rng(0).choice([-1.5, -0.5, 0.5, 1.5], 4000)
    assert psi(ref, same) < 0.01 and psi(ref, same + 2) > PSI_ALERT
    while True:
        try:
            check()
        except (urllib.error.URLError, OSError) as e:  # ES restarting: retry next tick instead of dying
            print(f"check failed: {e}", flush=True)
        time.sleep(INTERVAL)
