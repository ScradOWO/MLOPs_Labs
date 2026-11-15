# One image for the three Python services: model-api, traffic and drift-monitor.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --uid 10001 --create-home app \
 && mkdir /logs && chown app:app /logs
WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY train.py .
RUN python train.py && chown -R app:app model
COPY api.py traffic.py drift_monitor.py verify.py ./

USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --start-interval=1s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
