"""Download Quick Draw strokes and render them into training arrays.

    python model/training/prepare.py --work-dir /path/to/work

Output in ``<work-dir>/data``:
    X_train.npy   (N, S, S) uint8, already shuffled across classes (read in chunks while training)
    X_val.npy, X_test.npy   ordered by class, small enough to hold in memory
    y_train.npy, y_val.npy, y_test.npy
    meta.json     classes, image size, counts, the settings used and the data fingerprint
    test_ids.json.gz   Quick Draw key_id of every test drawing (the test set is auditable)

Raw strokes are cached in ``<work-dir>/raw`` so changing render settings does not download
anything again. Every step is deterministic for a given seed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image

from render import render_drawing

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.toml")
SIMPLIFIED_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"
CHUNK = 1000


def load_config(path=DEFAULT_CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_classes(cfg: dict) -> list[str]:
    path = Path(cfg["data"]["classes_file"])
    path = path if path.is_absolute() else ROOT_DIR / path
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return [mapping[str(i)] for i in range(len(mapping))]


def fetch_class(name: str, needed: int, raw_dir: Path) -> Path:
    """Stream the first ``needed`` recognized drawings of a class into the raw cache."""
    target = raw_dir / f"{name}.jsonl.gz"
    if target.exists():
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    kept = 0
    with requests.get(SIMPLIFIED_URL.format(quote(name)), stream=True, timeout=60) as response:
        response.raise_for_status()
        with gzip.open(partial, "wt", encoding="utf-8") as out:
            for line in response.iter_lines():
                if not line:
                    continue
                record = json.loads(line)
                if record.get("recognized"):
                    entry = {"id": record.get("key_id"), "d": record["drawing"]}
                    out.write(json.dumps(entry, separators=(",", ":")) + "\n")
                    kept += 1
                    if kept >= needed:
                        break
    partial.rename(target)
    return target


def read_records(path: Path) -> tuple[list, list]:
    """``(ids, drawings)`` of a cached class file; ids are Quick Draw's ``key_id``."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return [e["id"] for e in entries], [e["d"] for e in entries]


def data_fingerprint(cfg: dict, classes: list, samples: int, test_ids: list) -> str:
    """A short hash of everything that defines the evaluation data.

    Two models can be compared fairly only if their fingerprints are equal: same classes, same
    image size and rendering, same split settings and exactly the same test drawings.
    """
    data = cfg["data"]
    payload = {
        "classes": classes,
        "samples_per_class": samples,
        "val_per_class": data["val_per_class"],
        "test_per_class": data["test_per_class"],
        "image_size": data["image_size"],
        "seed": data["seed"],
        "render": cfg["render"],
        "test_ids": test_ids,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def split_indices(count: int, val: int, test: int, rng: np.random.Generator):
    """Deterministic (train, val, test) index split of ``count`` items."""
    if count <= val + test:
        raise ValueError(f"Only {count} drawings, need more than {val + test} for val + test")
    order = rng.permutation(count)
    return order[val + test :], order[test : val + test], order[:test]


def _render_chunk(args):
    drawings, widths, image_size, render = args
    out = np.empty((len(drawings), image_size, image_size), dtype=np.uint8)
    for i, (drawing, width) in enumerate(zip(drawings, widths, strict=True)):
        out[i] = render_drawing(
            drawing,
            image_size,
            canvas=render["canvas"],
            margin=render["margin"],
            line_width=width,
            min_extent=render["min_extent"],
        )
    return out


def _render(pool, drawings, widths, image_size, render):
    jobs = [
        (drawings[i : i + CHUNK], widths[i : i + CHUNK], image_size, render)
        for i in range(0, len(drawings), CHUNK)
    ]
    parts = pool.map(_render_chunk, jobs) if pool else [_render_chunk(j) for j in jobs]
    if not parts:
        return np.empty((0, image_size, image_size), dtype=np.uint8)
    return np.concatenate(parts)


def prepare(
    cfg: dict, work_dir: Path, *, fetch=fetch_class, class_limit=None, samples=None, workers=None
):
    data, render = cfg["data"], cfg["render"]
    size, seed = data["image_size"], data["seed"]
    n_val, n_test = data["val_per_class"], data["test_per_class"]
    samples = samples or data["samples_per_class"]
    classes = load_classes(cfg)[:class_limit]
    raw_dir, out_dir = work_dir / "raw", work_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(classes)} classes ({samples} drawings each) ...")
    with ThreadPoolExecutor(8) as executor:
        paths = list(executor.map(lambda c: fetch(c, samples, raw_dir), classes))

    # Pass 1: split sizes, so the output files can be allocated up front.
    splits, train_counts = [], []
    for idx, path in enumerate(paths):
        count = len(read_records(path)[0])
        rng = np.random.default_rng(seed + idx)
        splits.append(split_indices(count, n_val, n_test, rng))
        train_counts.append(len(splits[-1][0]))
    n_train = sum(train_counts)
    offsets = np.concatenate([[0], np.cumsum(train_counts)[:-1]])
    row_of = np.random.default_rng(seed).permutation(n_train)  # global index -> shuffled row

    x_train = np.lib.format.open_memmap(
        out_dir / "X_train.npy", mode="w+", dtype=np.uint8, shape=(n_train, size, size)
    )
    y_train = np.empty(n_train, dtype=np.int16)
    x_val, y_val, x_test, y_test = [], [], [], []
    test_ids = []

    workers = workers or os.cpu_count() or 1
    pool = Pool(workers) if workers > 1 else None
    jitter = render["line_width_jitter"]
    started = time.time()
    try:
        for idx, (name, path) in enumerate(zip(classes, paths, strict=True)):
            ids, drawings = read_records(path)
            train_idx, val_idx, test_idx = splits[idx]
            width_rng = np.random.default_rng(seed + 10_000 + idx)
            fixed = float(render["line_width"])

            train_imgs = _render(
                pool,
                [drawings[i] for i in train_idx],
                width_rng.uniform(fixed - jitter, fixed + jitter, len(train_idx)).tolist(),
                size,
                render,
            )
            test_ids.append([ids[i] for i in test_idx])
            rows = row_of[offsets[idx] : offsets[idx] + len(train_idx)]
            x_train[rows] = train_imgs
            y_train[rows] = idx

            for imgs_list, labels_list, picked in (
                (x_val, y_val, val_idx),
                (x_test, y_test, test_idx),
            ):
                imgs_list.append(
                    _render(
                        pool, [drawings[i] for i in picked], [fixed] * len(picked), size, render
                    )
                )
                labels_list.append(np.full(len(picked), idx, dtype=np.int16))
            print(
                f"[{idx + 1}/{len(classes)}] {name}: {len(train_idx)} train "
                f"({time.time() - started:.0f}s)"
            )
    finally:
        if pool:
            pool.close()
            pool.join()

    x_train.flush()
    np.save(out_dir / "y_train.npy", y_train)
    for split, xs, ys in (("val", x_val, y_val), ("test", x_test, y_test)):
        np.save(out_dir / f"X_{split}.npy", np.concatenate(xs))
        np.save(out_dir / f"y_{split}.npy", np.concatenate(ys))

    fingerprint = data_fingerprint(cfg, classes, samples, test_ids)
    with gzip.open(out_dir / "test_ids.json.gz", "wt", encoding="utf-8") as f:
        json.dump(test_ids, f)
    meta = {
        "classes": classes,
        "image_size": size,
        "fingerprint": fingerprint,
        "counts": {"train": n_train, "val": len(classes) * n_val, "test": len(classes) * n_test},
        "config": cfg,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_sample_grid(out_dir, x_train, y_train, classes)
    print(f"Done: {meta['counts']} in {out_dir} (data fingerprint {fingerprint})")
    return out_dir


def save_sample_grid(out_dir: Path, x_train, y_train, classes, per_class=8, max_classes=12):
    """A quick visual check of what the model will be trained on."""
    size = x_train.shape[1]
    shown = min(len(classes), max_classes)
    grid = Image.new("L", (per_class * size, shown * size), 255)
    for c in range(shown):
        rows = np.flatnonzero(y_train == c)[:per_class]
        for j, row in enumerate(rows):
            grid.paste(Image.fromarray(255 - np.asarray(x_train[row])), (j * size, c * size))
    grid.save(out_dir / "samples.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--classes", type=int, help="only the first N classes (quick trial)")
    parser.add_argument("--samples", type=int, help="override samples_per_class")
    parser.add_argument("--workers", type=int, help="render processes (default: all CPUs)")
    args = parser.parse_args()
    prepare(
        load_config(args.config),
        args.work_dir,
        class_limit=args.classes,
        samples=args.samples,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
