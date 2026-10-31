# syntax=docker/dockerfile:1
# Four stages: base -> deps -> trainer -> serving.
# Only `serving` is the final image; the training script and pip cache never reach it.

ARG PYTHON_VERSION=3.12

# ---- base: shared slim runtime + a non-root user ---------------------------------
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH
RUN useradd --uid 10001 --create-home app
WORKDIR /app

# ---- deps: install pinned packages into a venv ------------------------------------
# requirements.txt is copied alone so code edits don't invalidate this layer.
FROM base AS deps
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv && pip install -r requirements.txt

# ---- trainer: train at build time, fail the build if the model is too weak --------
FROM deps AS trainer
ARG N_ESTIMATORS=200
ARG MAX_DEPTH=0
ARG MIN_ACCURACY=0.90
ENV N_ESTIMATORS=${N_ESTIMATORS} \
    MAX_DEPTH=${MAX_DEPTH} \
    MIN_ACCURACY=${MIN_ACCURACY}
COPY src/train.py .
RUN mkdir model && chown app:app model
USER app
RUN python train.py

# ---- serving: venv + model artifact + API code, nothing else -----------------------
FROM base AS serving
COPY --from=deps /opt/venv /opt/venv
COPY --from=trainer --chown=app:app /app/model ./model
COPY src/app.py .
COPY src/templates ./templates
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --start-interval=1s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--access-logfile", "-", "app:app"]
