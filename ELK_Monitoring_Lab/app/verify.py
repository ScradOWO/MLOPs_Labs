"""End-to-end check of the monitoring stack. Polls for up to TIMEOUT seconds until every check passes.

    docker compose run --rm verify
"""
import json
import os
import time
import urllib.error
import urllib.request

ES = os.environ.get("ES_URL", "http://elasticsearch:9200")
KB = os.environ.get("KIBANA_URL", "http://kibana:5601")
TIMEOUT = int(os.environ.get("TIMEOUT", 300))


def get(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def count(index, query=None):
    r = get(f"{ES}/{index}/_count", {"query": query} if query else None)
    return r["count"] if r else 0


def checks():
    drift_docs = (get(f"{ES}/ml-drift/_search", {"size": 1000, "sort": ["@timestamp"]}) or {"hits": {"hits": []}})
    drift = [h["_source"] for h in drift_docs["hits"]["hits"]]
    alerts = [d for d in drift if d["alert"]]
    return {
        "Kibana dashboard 'ML model monitoring' imported": get(f"{KB}/api/saved_objects/dashboard/ml-monitoring") is not None,
        "predictions indexed (>= 1000)": count("ml-predictions") >= 1000,
        "no unparsed log lines": count("ml-unparsed") == 0,
        "a stable window was measured (no alert)": any(not d["alert"] for d in drift),
        "drift alert raised for bmi and bp": any({"bmi", "bp"} <= set(d["drifted_features"]) for d in alerts),
        "no false alarm on the untouched features": all(set(d["drifted_features"]) <= {"bmi", "bp"} for d in alerts),
    }


deadline = time.monotonic() + TIMEOUT
while True:
    try:
        result = checks()
    except (urllib.error.URLError, OSError) as e:
        result = {f"Elasticsearch/Kibana reachable ({e})": False}
    if all(result.values()) or time.monotonic() > deadline:
        break
    print(f"waiting: {sum(result.values())}/{len(result)} checks pass", flush=True)
    time.sleep(10)

for name, ok in result.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print(f"predictions: {count('ml-predictions')}, drift checks: {count('ml-drift')}, "
      f"alerts: {count('ml-drift', {'term': {'alert': True}})}")
if not all(result.values()):
    raise SystemExit("VERIFY FAILED")
print("VERIFY PASSED")
