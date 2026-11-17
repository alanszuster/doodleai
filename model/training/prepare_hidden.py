"""Prepare a separate binary drawing dataset without changing the public class list."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import tempfile
from pathlib import Path

import requests

from prepare import fetch_class, load_classes, load_config, prepare

SOURCE = "https://raw.githubusercontent.com/studiomoniker/Quickdraw-appendix/master/penis-simplified.ndjson"
NEGATIVE = "other"
POSITIVE = "target"


def fetch_positive(needed: int, raw_dir: Path) -> Path:
    if needed <= 0:
        raise ValueError("samples must be positive")
    target = raw_dir / f"{POSITIVE}.jsonl.gz"
    if target.exists():
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            if sum(1 for _ in handle) >= needed:
                return target

    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=raw_dir) as temp:
        partial = Path(temp) / "drawings.jsonl.gz"
        count = 0
        ids = set()
        with requests.get(SOURCE, stream=True, timeout=60) as response:
            response.raise_for_status()
            with gzip.open(partial, "wt", encoding="utf-8") as output:
                for line in response.iter_lines():
                    if not line:
                        continue
                    item = json.loads(line)
                    if not item.get("key_id") or not item.get("drawing"):
                        raise ValueError("Invalid source drawing: missing key_id or drawing")
                    if item["key_id"] in ids:
                        raise ValueError("Duplicate source drawing ID")
                    ids.add(item["key_id"])
                    output.write(json.dumps({"id": item["key_id"], "d": item["drawing"]}) + "\n")
                    count += 1
                    if count == needed:
                        break
        if count < needed:
            raise ValueError(f"Dataset contains {count} drawings; {needed} requested")
        partial.replace(target)
    return target


def prepare_hidden(
    cfg: dict,
    work_dir: Path,
    *,
    samples: int,
    negative_samples: int | None = None,
    baseline_samples: int | None = None,
    workers: int | None = None,
) -> Path:
    negative_samples = samples if negative_samples is None else negative_samples
    if samples <= cfg["data"]["val_per_class"] + cfg["data"]["test_per_class"]:
        raise ValueError("samples must exceed the combined validation and test sizes")
    if negative_samples <= cfg["data"]["val_per_class"] + cfg["data"]["test_per_class"]:
        raise ValueError("negative samples must exceed the combined validation and test sizes")
    if baseline_samples is not None and not (
        cfg["data"]["val_per_class"] + cfg["data"]["test_per_class"]
        < baseline_samples
        <= min(samples, negative_samples)
    ):
        raise ValueError("baseline samples must fit both classes and their validation/test splits")
    classes = load_classes(cfg)
    if not classes:
        raise ValueError("No negative source classes")
    raw = work_dir / "raw"
    positive = fetch_positive(samples, raw)
    per_class = math.ceil(negative_samples / len(classes))
    sources = [fetch_class(name, per_class, raw / "negative_sources") for name in classes]
    raw.mkdir(parents=True, exist_ok=True)
    negative = raw / f"{NEGATIVE}.jsonl.gz"
    with tempfile.TemporaryDirectory(dir=raw) as temp:
        partial = Path(temp) / "other.jsonl.gz"
        with gzip.open(partial, "wt", encoding="utf-8") as output:
            if baseline_samples is None:
                batches = [(0, negative_samples)]
            else:
                batches = [(0, baseline_samples), (baseline_samples, negative_samples)]
            for start_total, end_total in batches:
                for index, (name, path) in enumerate(zip(classes, sources, strict=True)):
                    start = start_total // len(classes) + (index < start_total % len(classes))
                    end = end_total // len(classes) + (index < end_total % len(classes))
                    with gzip.open(path, "rt", encoding="utf-8") as source:
                        for _ in range(start):
                            if not source.readline():
                                raise ValueError(f"{name}: not enough negative drawings")
                        for _ in range(end - start):
                            line = source.readline()
                            if not line:
                                raise ValueError(f"{name}: not enough negative drawings")
                            item = json.loads(line)
                            item["id"] = f"{name}:{item['id']}"
                            output.write(json.dumps(item) + "\n")
        partial.replace(negative)

    work_dir.mkdir(parents=True, exist_ok=True)
    classes_file = work_dir / "binary-classes.json"
    classes_file.write_text(json.dumps({"0": NEGATIVE, "1": POSITIVE}), encoding="utf-8")
    cfg["data"]["classes_file"] = str(classes_file.resolve())
    return prepare(
        cfg,
        work_dir,
        samples=samples,
        sample_counts=(
            {NEGATIVE: negative_samples, POSITIVE: samples}
            if negative_samples != samples or baseline_samples is not None
            else None
        ),
        split_anchor_samples=baseline_samples,
        workers=workers,
        fetch=lambda name, _needed, _raw: positive if name == POSITIVE else negative,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=24000)
    parser.add_argument(
        "--negative-samples", type=int, help="ordinary drawings (default: --samples)"
    )
    parser.add_argument("--baseline-samples", type=int, help="preserve the original split size")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    cfg = load_config()
    cfg["data"].update(val_per_class=1000, test_per_class=1000)
    prepare_hidden(
        cfg,
        args.work_dir,
        samples=args.samples,
        negative_samples=args.negative_samples,
        baseline_samples=args.baseline_samples,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
