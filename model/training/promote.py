"""Promote a trained model to production, behind the quality gate.

    python model/training/promote.py --run-dir kaggle-output/runs/main [--dry-run]

``--run-dir`` is the downloaded run (it holds ``best.keras``, ``config.json`` and
``report/metrics.json``). The model replaces the one in ``model/artifacts/`` only if
``registry.check_promotion`` agrees: same test set, clearly better, no class clearly worse.
On success it copies the Keras model, exports the ONNX model the API serves, writes
``model_card.json`` next to them, saves the run record and copies the training-history plot to
``model/reports/`` (shown in the README).
Open a pull request with the result; CI repeats the check against the model on ``main``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import registry

RUN_MODEL = "best.keras"  # what train.py saves in a run directory
REPORTS_DIR = registry.ROOT / "model" / "reports"  # the production model's plots, for the README
REPORT_FILES = ("training_history.png",)


def _export_onnx(keras_path: Path, onnx_path: Path) -> None:
    from export import export_onnx  # needs TensorFlow: only the training machine has it

    export_onnx(keras_path, onnx_path)


def current_card(artifacts_dir: Path) -> dict | None:
    path = artifacts_dir / registry.CARD_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def classes_in_order(artifacts_dir: Path) -> list[str]:
    mapping = json.loads((artifacts_dir / "classes.json").read_text(encoding="utf-8"))
    return [mapping[str(i)] for i in range(len(mapping))]


def promote(
    run_dir: Path,
    *,
    artifacts_dir: Path = registry.ARTIFACTS_DIR,
    experiments_dir: Path = registry.EXPERIMENTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    dry_run: bool = False,
    export=_export_onnx,
) -> tuple[bool, list[str]]:
    """Run the gate and, if it passes, install the model. Returns (promoted, reasons)."""
    record = registry.build_record(run_dir)
    if list(record["per_class_accuracy"]) != classes_in_order(artifacts_dir):
        return False, ["the model's classes differ from model/artifacts/classes.json (or order)"]
    ok, reasons = registry.check_promotion(
        record, current_card(artifacts_dir), registry.promotion_rules()
    )
    if not ok:
        return False, reasons
    if dry_run:
        return True, [*reasons, "dry run: nothing changed"]

    model_path = artifacts_dir / registry.MODEL_NAME
    onnx_path = artifacts_dir / registry.ONNX_NAME
    with tempfile.TemporaryDirectory() as scratch:
        # Convert first: if the export fails, nothing in model/artifacts has been touched.
        staged = Path(scratch) / registry.ONNX_NAME
        export(run_dir / RUN_MODEL, staged)
        shutil.copyfile(run_dir / RUN_MODEL, model_path)
        shutil.copyfile(staged, onnx_path)
    card = registry.make_card(record, model_path, onnx_path)
    (artifacts_dir / registry.CARD_NAME).write_text(json.dumps(card, indent=2) + "\n", "utf-8")
    registry.save_record(record, experiments_dir)
    for name in REPORT_FILES:
        if (run_dir / "report" / name).exists():
            reports_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_dir / "report" / name, reports_dir / name)
    return True, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="only run the gate")
    args = parser.parse_args()
    ok, reasons = promote(args.run_dir, dry_run=args.dry_run)
    for reason in reasons:
        print(("ok:       " if ok else "rejected: ") + reason)
    if not ok:
        sys.exit(1)
    if not args.dry_run:
        print("Promoted. Now: git checkout -b model/<run_id>, commit model/artifacts/ and")
        print("model/experiments/, open a pull request. CI checks it against main.")


if __name__ == "__main__":
    main()
