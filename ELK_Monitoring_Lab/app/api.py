"""Model-serving API that writes one structured JSON log line per request.

Logstash tails /logs/predictions.jsonl and ships every line to Elasticsearch,
so the log line *is* the monitoring record: inputs, output, latency, status.
"""
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_VERSION = os.environ.get("MODEL_VERSION", "gbr-v1")
LOG_PATH = os.environ.get("PREDICTION_LOG", "/logs/predictions.jsonl")

model = joblib.load("model/model.joblib")

log = logging.getLogger("predictions")
log.setLevel(logging.INFO)
handler = logging.FileHandler(LOG_PATH)
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)


class Features(BaseModel):
    """The 10 (already standardized) features of sklearn's diabetes dataset."""
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


app = FastAPI(title="Diabetes progression model")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(features: Features):
    start = time.perf_counter()
    row = features.model_dump()
    prediction = float(model.predict([list(row.values())])[0])
    latency_ms = (time.perf_counter() - start) * 1000
    request_id = str(uuid.uuid4())
    log.info(json.dumps({
        "event_type": "prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": MODEL_VERSION,
        "features": row,
        "prediction": round(prediction, 2),
        "latency_ms": round(latency_ms, 3),
    }))
    return {"request_id": request_id, "prediction": round(prediction, 2), "model_version": MODEL_VERSION}
