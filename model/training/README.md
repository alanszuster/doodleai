# Scripts

Data preparation and model training scripts.

## Files

- `prepare_data.py` - Download and preprocess the Quick Draw! dataset
- `train_model.py` - Train the CNN model
- `check_classes_mapping.py` - Verify the number of output classes in a trained model

## Usage

```bash
# 1. Download and preprocess data (~2GB download, configurable sample count)
python scripts/prepare_data.py

# 2. Train the model
python scripts/train_model.py
```

## Configuration

Paths and training parameters are defined in `config.py` at the project root.

## Output

- `prepare_data.py` writes processed splits to `dataset/processed/` and class mappings to `model/classes.json`
- `train_model.py` saves the best model to `model/best_model.keras` and training plots to `outputs/`
