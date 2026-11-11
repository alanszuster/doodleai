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
