"""Experiment records, model cards and the gate that decides what may reach production.

Every evaluated training run leaves a small record in ``model/experiments/<run_id>.json``.
The model that production serves lives in ``model/artifacts/`` together with
``model_card.json`` (its metrics, the test-set fingerprint, where it came from and the hash of
the model file). A new model may replace it only through ``check_promotion``. Standard library
only, so it can run in CI and anywhere else.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tomllib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "model" / "experiments"
ARTIFACTS_DIR = ROOT / "model" / "artifacts"
CONFIG_PATH = Path(__file__).with_name("config.toml")
CARD_NAME = "model_card.json"
MODEL_NAME = "best_model.keras"  # the Keras model (source of truth, used to evaluate)
ONNX_NAME = "best_model.onnx"  # what the API serves
METRIC_KEYS = ("samples", "top1", "top3", "macro_f1", "ece")


def promotion_rules() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)["promotion"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _history_summary(path: Path) -> tuple[int | None, bool]:
    """(epochs trained, whether the loss ever became NaN/inf); (None, False) without history."""
    if not path.exists():
        return None, False
    with path.open(encoding="utf-8", newline="") as handle:
        losses = [float(row["loss"]) for row in csv.DictReader(handle)]
    return len(losses), not all(math.isfinite(loss) for loss in losses)


def build_record(run_dir: Path, kaggle: dict | None = None) -> dict:
    """A record from a downloaded run directory (``config.json`` + ``report/metrics.json``)."""
    metrics = json.loads((run_dir / "report" / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    run_id = metrics.get("run_id") or config.get("run_id")
    if not run_id:
        raise ValueError(f"{run_dir} has no run_id: it was not trained through kaggle_run.py")
    epochs_run, diverged = _history_summary(run_dir / "history.csv")
    return {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "epochs_run": epochs_run,
        "diverged": diverged,
        "kaggle": kaggle or {},
        "git_commit": config.get("git_commit"),
        "data_fingerprint": metrics.get("data_fingerprint"),
        "image_size": metrics.get("image_size"),
        "temperature": metrics.get("temperature", 1.0),  # calibration, applied by the API
        "metrics": {
            key: metrics[key]
            for key in (*METRIC_KEYS, "log_loss", "ece_calibrated", "log_loss_calibrated")
            if key in metrics
        },
        "per_class_accuracy": metrics["per_class_accuracy"],
        "train_config": config["config"]["train"],
    }


def save_record(record: dict, directory: Path | None = None) -> Path:
    directory = directory or EXPERIMENTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def load_records(directory: Path | None = None) -> list[dict]:
    directory = directory or EXPERIMENTS_DIR
    if not directory.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]


def comparison_table(records: list[dict]) -> str:
    """Records ranked by top-1; ``*`` marks runs whose test set differs from the best one's."""
    if not records:
        return "No experiment records yet."
    ranked = sorted(records, key=lambda r: r["metrics"]["top1"], reverse=True)
    reference = ranked[0]["data_fingerprint"]
    lines = [
        f"{'run':<24} {'top-1':>7} {'top-3':>7} {'F1':>6} {'ECE':>6}  {'data':<16} commit",
        "-" * 78,
    ]
    for r in ranked:
        m = r["metrics"]
        flag = "" if r["data_fingerprint"] == reference else " *"
        lines.append(
            f"{r['run_id']:<24} {m['top1']:>7.2%} {m['top3']:>7.2%} {m['macro_f1']:>6.3f} "
            f"{m['ece']:>6.3f}  {str(r['data_fingerprint']):<16} "
            f"{str(r.get('git_commit') or '')[:7]}{flag}"
        )
    if any(r["data_fingerprint"] != reference for r in ranked):
        lines.append("* different test set than the top run: not directly comparable")
    return "\n".join(lines)


def make_card(record: dict, model_path: Path, onnx_path: Path | None = None) -> dict:
    """The model card: the record plus the files it describes."""
    card = {**record, "model_file": model_path.name, "model_sha256": sha256_file(model_path)}
    if onnx_path is not None:
        card |= {"onnx_file": onnx_path.name, "onnx_sha256": sha256_file(onnx_path)}
    return card


def check_promotion(candidate: dict, champion: dict | None, rules: dict) -> tuple[bool, list[str]]:
    """Decide whether ``candidate`` may replace ``champion``. Returns (ok, reasons)."""
    if candidate.get("diverged"):
        return False, [
            "training diverged (NaN loss): the saved model is only from the last good epoch "
            "and is under-trained; fix the run and train again"
        ]
    if champion is None:
        return True, ["no model in production yet: the first one is accepted"]
    if candidate["data_fingerprint"] != champion["data_fingerprint"]:
        return False, [
            "test sets differ (data fingerprint "
            f"{candidate['data_fingerprint']} vs {champion['data_fingerprint']}): "
            "the models are not comparable; evaluate both on the same data first"
        ]

    reasons, ok = [], True
    gain = candidate["metrics"]["top1"] - champion["metrics"]["top1"]
    if gain < rules["margin"]:
        ok = False
        reasons.append(
            f"top-1 {gain * 100:+.2f} points is below the required +{rules['margin'] * 100:.2f}"
        )
    else:
        reasons.append(f"top-1 {gain * 100:+.2f} points")

    worse = {
        name: champion["per_class_accuracy"][name] - accuracy
        for name, accuracy in candidate["per_class_accuracy"].items()
        if name in champion["per_class_accuracy"]
        and champion["per_class_accuracy"][name] - accuracy > rules["max_class_drop"]
    }
    if worse:
        ok = False
        detail = ", ".join(f"{n} -{d * 100:.1f}" for n, d in sorted(worse.items()))
        reasons.append(f"classes got clearly worse (points): {detail}")

    ece_rise = candidate["metrics"]["ece"] - champion["metrics"]["ece"]
    if ece_rise > rules["max_ece_rise"]:
        ok = False
        reasons.append(f"calibration (ECE) got worse by {ece_rise:.3f}")
    return ok, reasons
