"""Publish the promoted model to the Hugging Face Hub as a versioned model repository.

    python model/training/publish_hub.py [--repo alanoee/doodleai] [--dry-run]

Takes what ``promote.py`` installed in model/artifacts (the ONNX and Keras models, the card, the
class list) plus the training-history plot, writes a model card (README) from
model_card_template.md, uploads it in one commit and tags the commit with the run id. Every
promoted model stays retrievable that way (``revision="<run_id>"``), and the model is discoverable
on the Hub. Needs ``hf auth login``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import registry

TEMPLATE = Path(__file__).with_name("model_card_template.md")
DEFAULT_REPO = "alanoee/doodleai"
REPORTS_DIR = registry.ROOT / "model" / "reports"
FILES = (registry.MODEL_NAME, registry.ONNX_NAME, registry.CARD_NAME, "classes.json")


def render_card(card: dict, template: str | None = None) -> str:
    """The model card README for the promoted model described by ``card``."""
    per_class = card["per_class_accuracy"]
    hardest = sorted(per_class.items(), key=lambda item: item[1])[:5]
    metrics = card["metrics"]
    values = {
        "CLASSES": str(len(per_class)),
        "SIZE": str(card["image_size"]),
        "RUN_ID": card["run_id"],
        "TEST_SAMPLES": f"{metrics['samples']:,}".replace(",", " "),
        "TOP1": f"{metrics['top1']:.1%}",
        "TOP3": f"{metrics['top3']:.1%}",
        "F1": f"{metrics['macro_f1']:.3f}",
        "ECE": f"{metrics['ece']:.3f}",
        "HARDEST": ", ".join(f"{name} ({accuracy:.0%})" for name, accuracy in hardest),
    }
    text = template if template is not None else TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    if "{{" in text:
        raise ValueError("The model card template has a placeholder that is not filled in")
    return text


def stage(artifacts_dir: Path, reports_dir: Path, out_dir: Path) -> list[str]:
    """Copy what gets published into ``out_dir``; returns the file names."""
    card = json.loads((artifacts_dir / registry.CARD_NAME).read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for name in FILES:
        shutil.copyfile(artifacts_dir / name, out_dir / name)
        names.append(name)
    plot = reports_dir / "training_history.png"
    if plot.exists():
        shutil.copyfile(plot, out_dir / plot.name)
        names.append(plot.name)
    (out_dir / "README.md").write_text(render_card(card), encoding="utf-8")
    return [*names, "README.md"]


def publish(repo: str, artifacts_dir: Path, reports_dir: Path, *, dry_run: bool = False) -> str:
    """Upload the staged files to ``repo`` and tag the commit with the run id."""
    card_path = artifacts_dir / registry.CARD_NAME
    if not card_path.exists():
        raise SystemExit("No model_card.json: promote a model first (promote.py).")
    run_id = json.loads(card_path.read_text(encoding="utf-8"))["run_id"]
    with tempfile.TemporaryDirectory() as scratch:
        names = stage(artifacts_dir, reports_dir, Path(scratch))
        print(f"{repo} @ {run_id}: " + ", ".join(names))
        if dry_run:
            print("dry run: nothing uploaded")
            return run_id
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=scratch,
            repo_id=repo,
            repo_type="model",
            commit_message=f"Model {run_id}",
        )
        api.create_tag(repo, tag=run_id, repo_type="model", exist_ok=True)
    print(f"Published: https://huggingface.co/{repo}/tree/{run_id}")
    return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_REPO, help="the model repo on the Hub")
    parser.add_argument("--dry-run", action="store_true", help="stage the files, upload nothing")
    args = parser.parse_args()
    publish(args.repo, registry.ARTIFACTS_DIR, REPORTS_DIR, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
