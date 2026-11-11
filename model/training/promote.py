"""Promote a trained model to production, behind the quality gate.

    python model/training/promote.py --run-dir kaggle-output/runs/main [--dry-run]

``--run-dir`` is the downloaded run (it holds ``best.keras``, ``config.json`` and
``report/metrics.json``). The model replaces the one in ``model/artifacts/`` only if
``registry.check_promotion`` agrees: same test set, clearly better, no class clearly worse.
On success it copies the model, writes ``model_card.json`` next to it and saves the run record.
Open a pull request with the result; CI repeats the check against the model on ``main``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import registry


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
    dry_run: bool = False,
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
    shutil.copyfile(run_dir / registry.MODEL_NAME, model_path)
    card = registry.make_card(record, model_path)
    (artifacts_dir / registry.CARD_NAME).write_text(json.dumps(card, indent=2) + "\n", "utf-8")
    registry.save_record(record, experiments_dir)
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
