# TensorBoard Lab: Hyperparameter Tuning a CIFAR-10 CNN with HParams

Modified version of the MLOps course TensorBoard Lab 4 ("Hyperparameter Tuning with the HParams Dashboard").

## What changed from the original lab

| | Original lab | This lab |
|---|---|---|
| Dataset | Fashion-MNIST (grayscale 28x28) | **CIFAR-10** (color 32x32x3) |
| Model | Flatten + 1 Dense layer MLP | **2-block CNN** (Conv2D + MaxPool) with random-flip augmentation |
| Hyperparameters | units, dropout, optimizer (8 runs) | **conv filters, dropout, optimizer, learning rate** (16 runs) |
| Training | 1 epoch, no validation | 5 epochs with a held-out validation set |
| Metrics | test accuracy only | **per-epoch train/val loss and accuracy**, test loss and accuracy |
| Dashboards | HParams | HParams, **Scalars, Graphs, Images** (confusion matrix of the best run) |
| Misc | `!rm -rf` (Unix only), debug dump enabled (slow) | cross-platform log cleanup, debug dump removed |

## Run

```bash
cd TensorBoard_HParams_Lab
pip install -r requirements.txt
jupyter notebook hparams_cifar10.ipynb   # run all cells
```

The grid search (16 runs x 5 epochs on a 20k-image subset) takes a few minutes on a CPU. Then view the results:

```bash
tensorboard --logdir logs/hparam_tuning
```

Open http://localhost:6006 and check these tabs:
- **HParams**: table, parallel-coordinates, and scatter-plot views of all 16 runs. Sort by test accuracy.
- **Time Series / Scalars**: training and validation curves per run.
- **Graphs**: the conceptual Keras model graph (select the `keras` tag).
- **Images**: normalized confusion matrix of the best run.

## Results (5 epochs, 20k training images, CPU)

| filters | dropout | optimizer | lr | test acc |
|---|---|---|---|---|
| **32** | **0.2** | **adam** | **0.001** | **0.630** (best) |
| 32 | 0.4 | adam | 0.001 | 0.597 |
| 16 | 0.4 | adam | 0.001 | 0.574 |
| 16 | 0.2 | adam | 0.001 | 0.549 |
| 16 | 0.2 | adam | 0.01 | 0.525 |
| ... | | sgd | 0.01 | 0.29 - 0.34 |
| ... | | sgd | 0.001 | 0.14 - 0.17 |

Observations:
- Adam beat SGD in every configuration. With only 5 epochs, plain SGD at lr=0.001 barely learns, with test accuracy near chance.
- For Adam, lr=0.001 beat lr=0.01 in every case.
- More filters (32) helped. Higher dropout helped the small model but hurt the larger one.

Exact numbers vary slightly between runs.
