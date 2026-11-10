# Model training

Dataset preparation and training for the drawing classifier. The API in `src/` only consumes the
result: `model/artifacts/best_model.keras` and `model/artifacts/classes.json`.

## Files

- `prepare_data.py` - download and preprocess the Quick Draw! dataset
- `train_model.py` - train the CNN
- `check_classes_mapping.py` - verify the number of output classes in a trained model

## Usage

Run from the repository root:

```bash
uv sync --group training

# 1. Download and preprocess data (~2GB download)
uv run python model/training/prepare_data.py

# 2. Train the model
uv run python model/training/train_model.py
```

## Output

- `prepare_data.py` writes the raw data to `model/dataset/quickdraw/`, processed splits to
  `model/dataset/processed/` and the class mapping to `model/artifacts/classes.json`
- `train_model.py` saves the best model to `model/artifacts/best_model.keras`
- `model/reports/` holds training plots
