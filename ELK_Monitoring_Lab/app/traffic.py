"""Send simulated production traffic to the API: a normal phase, then a drifted one.

    PHASES="normal:60,drift:60" RATE=10 python traffic.py

normal: real rows resampled from the diabetes dataset, i.e. the distribution the model was trained on.
        (No added noise: `sex` and `s4` take few distinct values, and noise would itself look like drift.)
drift:  the same, but `bmi` shifted up by about one standard deviation and `bp` spread 1.8x wider,
        e.g. a new clinic whose patients are heavier and whose blood-pressure cuff is miscalibrated.
"""
import json
import os
import time
import urllib.request

import numpy as np
from sklearn.datasets import load_diabetes

API = os.environ.get("API_URL", "http://model-api:8000")
RATE = float(os.environ.get("RATE", 10))
PHASES = [(name, int(sec)) for name, sec in
          (p.split(":") for p in os.environ.get("PHASES", "normal:60,drift:60").split(","))]

data = load_diabetes(as_frame=True).data
rng = np.random.default_rng(0)


def make_row(phase):
    row = data.iloc[rng.integers(len(data))].copy()
    if phase == "drift":
        row["bmi"] += 0.05
        row["bp"] *= 1.8
    return {k: float(v) for k, v in row.items()}


def send(row):
    req = urllib.request.Request(f"{API}/predict", data=json.dumps(row).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)


def main():
    for phase, seconds in PHASES:
        if phase not in ("normal", "drift"):
            raise SystemExit(f"unknown phase {phase!r}: use normal or drift")
        end, sent = time.monotonic() + seconds, 0
        print(f"phase {phase}: {seconds}s at {RATE} req/s", flush=True)
        while time.monotonic() < end:
            send(make_row(phase))
            sent += 1
            time.sleep(1 / RATE)
        print(f"phase {phase}: sent {sent} requests", flush=True)
    print("TRAFFIC DONE", flush=True)


if __name__ == "__main__":
    main()
