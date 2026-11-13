# Model training

Dataset preparation and training for the drawing classifier. The API in `src/` only consumes
the result (`model/artifacts/`).

## How it works

1. **`prepare.py`** downloads Quick Draw *strokes* (not the 28x28 bitmaps) and renders them with
   **`render.py`**, which mirrors `renderForModel()` on the website: crop to the drawing, centre it,
   fixed margin and stroke width, then shrink to the model size exactly like the API does. The model
   is trained on the same kind of picture it will get in production. Training images get a random
   stroke width; validation and test images use the website's fixed width.
2. **`train.py`** trains a small residual CNN (`network.py`) with mixed precision, AdamW, warm-up +
   cosine learning rate, label smoothing and random rotation, shift and zoom.
   It saves `last.keras` every epoch and resumes with `--resume`.
3. **`evaluate.py`** measures the held-out test split: top-1, top-3, macro F1, per-class accuracy,
   most confused pairs, calibration and a confusion matrix.

Every setting is in [`config.toml`](config.toml).

## Train on a free GPU

Open [`train.ipynb`](train.ipynb) on **Kaggle** (recommended: free GPU, more RAM, stable weekly
quota) or **Google Colab**, turn on a GPU and run all cells. The notebook clones this repository,
runs the three scripts and hands you `best.keras` plus the metrics. If the session is interrupted,
run it again: preparation is skipped when done and training resumes from the last epoch.

## Run it on Kaggle from your terminal

[`kaggle_run.py`](kaggle_run.py) drives the same notebook through the Kaggle CLI, so the notebook in
this repository stays the single source of truth. There is **one** private Kaggle notebook,
`<you>/doodleai-train`; every `push` saves a new *version* of it (GPU T4 x2, internet on) with
`QUICK`, the git `REF` and a `RUN_ID` filled in, and starts it. A version is one training run, and the
`run_id` (date, time and mode, e.g. `20261111-102305-full`) identifies it everywhere: in its results,
in the experiment record and in the model card. Training takes hours, so this runs on your machine,
not in CI.

One-time setup:
1. kaggle.com → Settings → API → **Generate New Token**, saved in `~/.kaggle/access_token`
   (`chmod 600`), or `export KAGGLE_API_TOKEN=...`.
2. `export KAGGLE_USERNAME=<your username>`.

```bash
python model/training/kaggle_run.py run --mode quick --ref main   # push, wait, download: minutes
python model/training/kaggle_run.py run --mode full  --ref main   # the same for the real run: hours
```

Or step by step: `push`, `status`, `wait`, `collect --dir out` (downloads only if the run has
finished) and `output --dir out`. The latest run is remembered in `.kaggle-run.json` (not
committed), so the steps after `push` need no arguments; `--kernel <notebook>/<version>` targets
another one. Every step can be repeated, so if the terminal closes or the laptop sleeps during a long
run, start again with `wait` or `collect`: the run keeps going on Kaggle. Status checks that fail for
a moment (network, API) are retried. The free GPU quota is about 30 hours a week.

## Which model goes to production

Training many models is easy; making sure the best one is what the app serves is the point of this
section. Three pieces work together:

1. **A fixed, auditable test set.** `prepare.py` stores the Quick Draw `key_id` of every test drawing
   and a **data fingerprint**: a hash of the classes, split sizes, image size, rendering settings and
   those drawings. Two models can be compared only if their fingerprints are equal.
2. **Experiment records.** `output` (and `run`) saves `model/experiments/<run_id>.json` for every
   evaluated run: metrics, fingerprint, git commit, Kaggle notebook and version. Commit it to keep
   the run in the history. `python model/training/kaggle_run.py compare` ranks all records and marks
   those evaluated on a different test set.
3. **One way to production, guarded by a gate.** The model that is served lives in
   `model/artifacts/` with `model_card.json` (metrics, fingerprint, origin and the hash of the model
   file). It can be replaced only by a model that passes `registry.check_promotion`, with the limits in
   [`config.toml`](config.toml) (`[promotion]`): evaluated on the **same test set**, top-1 better by at
   least 0.5 points (test noise is about ±0.15), no class more than 3 points worse, calibration not
   clearly worse. Promotion also exports the ONNX model the API serves (`export.py`, checked against
   Keras on random input) and records its hash in the card; if the export fails nothing is replaced.
   The card also carries the **calibration temperature** that `evaluate.py` fits on the validation
   split; the API applies it so the percentages it shows match how often the model is right (a run
   evaluated before this existed has none, which means no change).
   It runs locally and needs the `training` group (TensorFlow runs on the CPU there).

```bash
python model/training/promote.py --run-dir kaggle-output/<path to the run> --dry-run   # only the gate
python model/training/promote.py --run-dir kaggle-output/<path to the run>             # installs it
git checkout -b model/<run_id> && git add model/artifacts model/experiments && git commit
```

Then open a pull request. CI repeats the check: it compares `model_card.json` with the one on the
target branch and fails if the new model is not better, or if the card does not match the model file.
The Hugging Face Space deploys from `main` only, so what runs in production is always a model that
passed the gate. To roll back, revert the pull request. If the data changes (new classes, other
rendering), the fingerprint changes and the gate refuses the comparison on purpose: evaluate the model
in production on the new test set first.

## Find what limits the speed

`bench.py` times a few steps of the real model under variants that each change one thing
(precision, XLA, the augmentation layers, and with `--data-dir` the real input pipeline instead of
one batch in memory). The variant that moves ms/step most is what to optimise. Run it on the GPUs
you train on, e.g. in a Kaggle notebook cell after cloning the repo:

```bash
python model/training/bench.py --data-dir /kaggle/working/work/data   # data dir only for real-data
```

## Run locally

The scripts need TensorFlow, Pillow, numpy, requests and matplotlib. A GPU is strongly advised.

```bash
uv sync --group training
WORK=/some/big/disk/doodleai      # ~15 GB for the default 30k drawings per class

uv run python model/training/prepare.py --work-dir $WORK              # for a quick trial add: --classes 5 --samples 4000
uv run python model/training/train.py --work-dir $WORK --name main   # quick trial: --epochs 8 --batch-size 64
uv run python model/training/evaluate.py --model $WORK/runs/main/best.keras \
    --data-dir $WORK/data --out $WORK/runs/main/report
```

If training segfaults on a CPU, set `TF_ENABLE_ONEDNN_OPTS=0` (a oneDNN bug; GPUs are not affected).

## Output

- `<work>/data/`: `X_*.npy`, `y_*.npy`, `meta.json`, `samples.png` (a quick look at the training pictures)
- `<work>/runs/<name>/`: `best.keras` (the model to ship), `last.keras`, `history.csv`, `config.json`, `report/`
