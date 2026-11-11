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
this repository stays the single source of truth. It uploads a copy (with `QUICK` and the git
`REF` filled in) as the private Kaggle notebook `<you>/doodleai-train` with a GPU and internet.
Training takes hours, so this runs on your machine, not in CI (a GitHub job may not live that long).

One-time setup:
1. kaggle.com → Settings → API → **Generate New Token**.
2. `export KAGGLE_API_TOKEN=...` and `export KAGGLE_USERNAME=<your username>`
   (or save the token in `~/.kaggle/access_token`).

```bash
python model/training/kaggle_run.py run --mode quick --ref main   # push, wait, download: minutes
python model/training/kaggle_run.py run --mode full  --ref main   # the same for the real run: hours
```

Or step by step: `push`, `status`, `wait`, `collect --dir out` (downloads only if the run has
finished) and `output --dir out`. Every step can be repeated, so if the terminal closes or the laptop
sleeps during a long run, just start with `wait` or `collect` again: the run keeps going on Kaggle.
While waiting, status checks that fail for a moment (network, API) are retried.
The free GPU quota is about 30 hours a week, so a full run is worth a deliberate decision.

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
