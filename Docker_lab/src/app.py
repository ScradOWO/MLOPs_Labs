"""Flask API that serves the wine classifier saved by train.py."""
import json
import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))

app = Flask(__name__)
model = joblib.load(MODEL_DIR / "model.joblib")
metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
FEATURES = metrics["feature_names"]
CLASSES = metrics["class_names"]


@app.get("/")
def index():
    return render_template("index.html", features=FEATURES, metrics=metrics)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return metrics


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or request.form
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify(error=f"missing features: {missing}"), 400
    try:
        row = [float(data[f]) for f in FEATURES]
    except (TypeError, ValueError) as e:
        return jsonify(error=f"features must be numbers: {e}"), 400

    proba = model.predict_proba([row])[0]
    best = int(proba.argmax())
    return jsonify(
        predicted_class=CLASSES[best],
        probabilities={c: round(float(p), 4) for c, p in zip(CLASSES, proba)},
    )


if __name__ == "__main__":
    # Local debugging only; the container runs gunicorn.
    app.run(host="0.0.0.0", port=8000)
