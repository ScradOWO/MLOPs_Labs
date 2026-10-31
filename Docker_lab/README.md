# Docker Lab: Build-Time Quality Gate, Slim Non-Root Serving, Health-Checked Compose Pipeline

Modified version of the MLOps course Docker lab (`Labs/Docker_Labs/Lab2`). The original lab trains a small TensorFlow model on Iris in one Docker stage and serves it with Flask from a second stage, with a Docker Compose file that does the same in two containers.

This version keeps the same idea (train in a container, serve from a container) and reworks the Docker side. The model can only reach the serving image if it passes an accuracy check during `docker build`. The serving image is slim, runs as a non-root user, and reports its own health. The Compose stack chains training, serving and an end-to-end test using health and completion conditions. It also uses a different dataset (Wine) and model (scikit-learn random forest).

## What changed from the original lab

| Area | Original Lab2 | This lab |
|---|---|---|
| Dockerfile stages | 2 (`model_training`, `serving`) | 4 (`base` → `deps` → `trainer` → `serving`) |
| Quality gate | none; any model is shipped | `docker build` **fails** if 5-fold CV accuracy < `MIN_ACCURACY` |
| Hyperparameters | hard-coded | build args / env vars: `N_ESTIMATORS`, `MAX_DEPTH`, `MIN_ACCURACY` |
| Base image | `python:3.10` (full Debian) | `python:3.12-slim` |
| Image size (content) | **1.64 GB** | **131 MB** |
| User inside container | `root` | `app` (uid 10001) |
| Server | Flask dev server, `debug=True` | gunicorn, 2 workers |
| Port | `EXPOSE 80` but app listens on 4000, so `-p 80:80` reaches nothing | `EXPOSE 8000`, app listens on 8000 |
| Health | none | `HEALTHCHECK` on `/health`; Docker shows `(healthy)` |
| Build context | everything in the folder | `.dockerignore` allow-list (only `requirements.txt` and `src/`) |
| pip | reinstalled on every build | BuildKit cache mount; dependency layer only rebuilds when `requirements.txt` changes |
| Dependencies | unpinned except TensorFlow | all pinned |
| Compose | `pip install` at every container start, model copied by `cp` | services use the built images; model shared through a named volume |
| Compose ordering | `serving` waits for training to finish | `api` waits for `train`; `smoke-test` waits for `api` to be **healthy** |
| Testing | manual | `smoke-test` service checks `/health`, `/model-info`, `/predict` and a bad request |
| Dataset / model | Iris, Keras dense net (TensorFlow) | Wine (178 samples, 13 features, 3 cultivars), `RandomForestClassifier` |
| API | `/predict` form → JSON | `/` web form, `POST /predict` (JSON or form), `/model-info`, `/health`; returns class probabilities and 400 on bad input |

The original also standardized features at training time but fed raw inputs to the model at serving time. Here the whole model is one saved artifact, and a random forest does not need scaling, so training and serving see the same inputs.

### How the pieces fit

```
Dockerfile
  base     python:3.12-slim + non-root user "app"
   └─ deps      pip install -r requirements.txt into /opt/venv   (cached layer)
       └─ trainer   python train.py  → model/model.joblib + metrics.json
                    exits 1 if cv_accuracy < MIN_ACCURACY  → build fails
  serving  base + /opt/venv + model/ + app.py + templates   (no train.py, no pip cache)
           USER app, HEALTHCHECK /health, gunicorn :8000

compose.yaml
  train ──(completed successfully)──▶ api ──(healthy)──▶ smoke-test
     └──── writes ──▶ [volume: model] ◀── reads (read-only) ──┘
```

Because `api` reads the model from the `model` volume, you can retrain with new hyperparameters by re-running only the `train` service. The API image does not need to be rebuilt. If a retrain fails the quality gate, `train.py` exits before saving, so the API keeps the previous good model.

### Files

```
Docker_lab/
├── Dockerfile             4-stage build (base, deps, trainer, serving)
├── .dockerignore          allow-list for the build context
├── compose.yaml           train → api → smoke-test
├── requirements.txt       pinned Python packages (installed inside the image only)
├── src/
│   ├── train.py           trains the model, applies the quality gate, writes model/ + metrics.json
│   ├── app.py             Flask API served by gunicorn
│   └── templates/index.html   web form for manual predictions
└── tests/
    └── smoke_test.py      end-to-end API check (Python standard library only)
```

## Requirements

You only need **Docker**. Python, scikit-learn and Flask are installed inside the image. You do not need them on your machine.

- **Docker Engine 25 or newer** with the **Compose v2 plugin** (`docker compose`, with a space). Engine 25 is the minimum because the `HEALTHCHECK --start-interval` option needs it. BuildKit (the default builder since Engine 23) is required for the pip cache mount.
  - Tested with Docker Engine 29.9.2, Docker Compose v5.6.1 and buildx v0.38.1 on Linux x86_64.
- About 1 GB of free disk space (base image, build cache and the two images), and an internet connection for the first build.
- Port **8000** free on your machine. If it is taken, see [Troubleshooting](#troubleshooting).
- `curl` for the manual checks. It ships with macOS, Linux and Windows 10/11. On Windows PowerShell, type `curl.exe`, not `curl`.

Images pulled and packages installed during the build:

| Item | Version |
|---|---|
| Base image | `python:3.12-slim` |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| Flask | 3.1.3 |
| gunicorn | 26.2.0 |

## Setup

### 1. Install Docker

**Windows or macOS:** install [Docker Desktop](https://docs.docker.com/desktop/) (4.27 or newer includes Engine 25+ and Compose v2). Start Docker Desktop and wait until it says the engine is running.

**Linux (Ubuntu, Debian, Fedora, and others):** install Docker Engine with Docker's convenience script:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER     # lets you run docker without sudo
```

Log out and back in (or run `newgrp docker`) so the group change takes effect. On Arch-based distros, use `sudo pacman -S docker docker-compose docker-buildx`, then `sudo systemctl enable --now docker`.

### 2. Check the install

```bash
docker version --format '{{.Server.Version}}'   # must print 25.0.0 or newer
docker compose version                          # must print v2.x or newer
docker run --rm hello-world                     # must print "Hello from Docker!"
```

### 3. Get the code

```bash
git clone https://github.com/ScradOWO/MLOPs_Labs.git
cd MLOPs_Labs/Docker_lab
```

Run every command below from inside the `Docker_lab` folder.

## Running the lab

There are two ways to run it. Option A is the full pipeline and is the recommended one.

### Option A: Compose pipeline (train → api → smoke test)

```bash
# 1. Build both images and start the stack in the background
docker compose up -d --build

# 2. Run the end-to-end test against the running API
docker compose run --rm smoke-test
```

The first build takes 1–3 minutes, mostly to download the base image and install packages. Later builds take a few seconds because the layers are cached.

**What success looks like.** Step 1 ends with:

```
 Container docker_lab-train-1 Exited
 Container docker_lab-api-1 Started
```

A few seconds later, `docker compose ps -a` shows the training container finished with code 0 and the API as healthy (right after startup it says `health: starting`):

```
SERVICE   STATUS
api       Up 3 seconds (healthy)
train     Exited (0) 1 second ago
```

Step 2 prints:

```
model-info: cv_accuracy=0.9862 test_accuracy=1.0
predict(row 0) -> {'predicted_class': 'cultivar_class_0', 'probabilities': {'cultivar_class_0': 1.0, 'cultivar_class_1': 0.0, 'cultivar_class_2': 0.0}}
SMOKE TEST PASSED
```

Then open **http://localhost:8000** in a browser. The form is pre-filled with a real wine sample. Click **Predict** to see the predicted cultivar and the class probabilities.

To see the training log (metrics JSON and the gate result), run `docker compose logs train`.

### Option B: plain `docker build` / `docker run`

```bash
docker build -t wine-api .
docker run -d --name wine-api -p 8000:8000 wine-api
docker ps          # wait until STATUS shows "(healthy)", about 2–5 seconds
```

The build log includes the training output and ends the trainer step with:

```
QUALITY GATE PASSED: model saved to model/
```

### Calling the API by hand

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/model-info
# {"cv_accuracy":0.9862, "test_accuracy":1.0, "min_accuracy":0.9, "params":{...}, ...}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol":12.37,"malic_acid":0.94,"ash":1.36,"alcalinity_of_ash":10.6,"magnesium":88,"total_phenols":1.98,"flavanoids":0.57,"nonflavanoid_phenols":0.28,"proanthocyanins":0.42,"color_intensity":1.95,"hue":1.05,"od280/od315_of_diluted_wines":1.82,"proline":520}'
# {"predicted_class":"cultivar_class_1","probabilities":{"cultivar_class_0":0.0,"cultivar_class_1":0.865,"cultivar_class_2":0.135}}

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"alcohol":13}'
# HTTP 400  {"error":"missing features: ['malic_acid', 'ash', ...]"}
```

The sample above is row 59 of the dataset, a real class-1 wine. On Windows PowerShell, `curl.exe` needs the JSON quotes escaped, so the browser form at http://localhost:8000 is easier there.

All 13 feature names are listed under `feature_names` in `/model-info`.

## Trying the changes

### 1. The quality gate stops a weak model at build time

Build with a deliberately weak forest (1 tree of depth 1):

```bash
docker build --build-arg N_ESTIMATORS=1 --build-arg MAX_DEPTH=1 -t wine-api:weak .
```

Expected: the build **fails**, and no `wine-api:weak` image is created.

```
QUALITY GATE FAILED: cv_accuracy 0.5847 < MIN_ACCURACY 0.9
ERROR: failed to build: failed to solve: process "/bin/sh -c python train.py" did not complete successfully: exit code: 1
```

You can also tighten or loosen the threshold, for example `--build-arg MIN_ACCURACY=0.99` (fails, since CV accuracy is 0.9862) or `--build-arg MIN_ACCURACY=0.5` (passes even with the weak forest).

### 2. Retrain without rebuilding the API image (Compose)

With the stack from Option A running:

```bash
N_ESTIMATORS=50 MAX_DEPTH=3 docker compose run --rm train
docker compose restart api
curl http://localhost:8000/model-info      # "params": {"max_depth": 3, "n_estimators": 50}
```

A retrain that fails the gate leaves the current model in place:

```bash
N_ESTIMATORS=1 MAX_DEPTH=1 docker compose run --rm train   # QUALITY GATE FAILED ..., exit code 1
docker compose restart api
curl http://localhost:8000/model-info      # still the 50-tree, depth-3 model
```

### 3. Non-root and image size

```bash
docker compose exec api id          # uid=10001(app) gid=10001(app) groups=10001(app)
docker image ls wine-api            # CONTENT SIZE about 131MB
```

For comparison, building the original Lab2 on the same machine gave a 1.64 GB image (5.34 GB on disk once unpacked) that runs as `uid=0(root)`.

## Configuration

| Variable | Where | Default | Meaning |
|---|---|---|---|
| `N_ESTIMATORS` | build arg / compose env | `200` | number of trees |
| `MAX_DEPTH` | build arg / compose env | `0` | max tree depth; `0` means no limit |
| `MIN_ACCURACY` | build arg / compose env | `0.90` | minimum 5-fold CV accuracy for the model to be saved |
| `API_PORT` | compose env | `8000` | host port the API is published on |
| `PYTHON_VERSION` | build arg | `3.12` | Python version of the `python:*-slim` base image |

## Cleaning up

```bash
docker compose down -v                    # stop the stack and delete the model volume
docker rm -f wine-api                     # if you used Option B
docker image rm wine-api wine-trainer     # remove the images
docker builder prune                      # optional: free the build cache
```

## Troubleshooting

- **`port is already allocated`** or `address already in use`: something else uses port 8000. Use another port, for example `API_PORT=8080 docker compose up -d` (then browse to http://localhost:8080), or `docker run -p 8080:8000 ...` for Option B. The smoke test talks to the API over the Compose network, so it does not need a change.
- **`permission denied while trying to connect to the Docker daemon socket`** (Linux): your user is not in the `docker` group yet. Run `sudo usermod -aG docker $USER` and log out and back in.
- **`unknown flag: --start-interval`** or a `HEALTHCHECK` parse error: your Docker Engine is older than 25. Upgrade Docker.
- **`docker: 'compose' is not a docker command`**: install the Compose v2 plugin (it is included in Docker Desktop, and in the `docker-compose-plugin` package on Ubuntu/Debian).
- **Code changes don't show up**: rebuild with `docker compose up -d --build`. `docker compose up` without `--build` reuses the existing images.
