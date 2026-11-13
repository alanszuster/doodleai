import copy
import gzip
import json
import math

import numpy as np
import pytest

from prepare import data_fingerprint, load_config, prepare, split_indices
from render import render_drawing

SIZE = 28


def circle(radius=100, n=24):
    xs = [128 + radius * math.cos(2 * math.pi * i / n) for i in range(n + 1)]
    ys = [128 + radius * math.sin(2 * math.pi * i / n) for i in range(n + 1)]
    return [[xs, ys]]


# --- renderer -------------------------------------------------------------------------


def test_empty_drawing_is_blank():
    assert render_drawing([], SIZE).sum() == 0


def test_render_has_the_expected_shape_and_range():
    image = render_drawing(circle(), 64)
    assert image.shape == (64, 64)
    assert image.dtype == np.uint8
    assert image.max() == 255  # ink is 255
    assert image[0, 0] == 0  # background is 0


def test_drawing_is_centred_and_fills_the_frame_with_the_site_margin():
    image = render_drawing([[[10, 240], [50, 50]]], 280, canvas=280, margin=25, line_width=22)
    ys, xs = np.nonzero(image > 128)
    # 25 px margin + half the stroke width of cap on every side
    assert xs.min() == pytest.approx(25 - 11, abs=2)
    assert xs.max() == pytest.approx(280 - 25 + 11, abs=2)
    assert (ys.min() + ys.max()) / 2 == pytest.approx(140, abs=2)


def test_render_ignores_position_and_scale_of_the_input():
    base = render_drawing(circle(radius=60), SIZE)
    moved = render_drawing([[[x + 30 for x in circle(60)[0][0]], circle(60)[0][1]]], SIZE)
    scaled = render_drawing(
        [[[x * 0.5 for x in circle(60)[0][0]], [y * 0.5 for y in circle(60)[0][1]]]], SIZE
    )
    # pixel rounding on the edges aside, the pictures are the same
    assert np.abs(base.astype(int) - moved).mean() < 1.5
    assert np.abs(base.astype(int) - scaled).mean() < 1.5


def test_a_single_tap_is_drawn_as_a_dot():
    assert render_drawing([[[100], [100]]], SIZE).sum() > 0


def test_thicker_strokes_have_more_ink():
    thin = render_drawing(circle(), SIZE, line_width=14)
    thick = render_drawing(circle(), SIZE, line_width=30)
    assert thick.sum() > thin.sum()


# --- splitting ------------------------------------------------------------------------


def test_split_is_deterministic_and_disjoint():
    first = split_indices(100, 10, 20, np.random.default_rng(1))
    again = split_indices(100, 10, 20, np.random.default_rng(1))
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    train, val, test = first
    assert (len(train), len(val), len(test)) == (70, 10, 20)
    assert len(set(train) | set(val) | set(test)) == 100


def test_split_needs_enough_data():
    with pytest.raises(ValueError):
        split_indices(30, 10, 20, np.random.default_rng(1))


# --- end to end on synthetic data -----------------------------------------------------

CLASSES = ["ring", "bar", "cross"]


def fake_drawing(label, rng):
    if label == 0:
        return circle(radius=int(rng.integers(60, 120)))
    if label == 1:
        y = float(rng.integers(100, 160))
        return [[[20.0, 230.0], [y, y + float(rng.integers(-5, 5))]]]
    return [[[30.0, 220.0], [30.0, 220.0]], [[220.0, 30.0], [30.0, 220.0]]]


@pytest.fixture
def tiny_cfg(tmp_path):
    classes_file = tmp_path / "classes.json"
    classes_file.write_text(json.dumps(dict(enumerate(CLASSES))))
    cfg = copy.deepcopy(load_config())
    cfg["data"].update(
        classes_file=str(classes_file),
        image_size=SIZE,
        samples_per_class=60,
        val_per_class=8,
        test_per_class=12,
    )
    cfg["train"].update(
        batch_size=16,
        epochs=1,
        widths=[8, 16],
        blocks=[1, 1],
        warmup_epochs=1,
        mixed_precision=False,
        shuffle_chunk=32,
        shuffle_buffer=64,
    )
    return cfg


def fake_fetch(name, needed, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name}.jsonl.gz"
    rng = np.random.default_rng(CLASSES.index(name))
    with gzip.open(path, "wt") as f:
        for i in range(needed):
            entry = {"id": f"{name}-{i}", "d": fake_drawing(CLASSES.index(name), rng)}
            f.write(json.dumps(entry) + "\n")
    return path


@pytest.fixture
def prepared(tiny_cfg, tmp_path):
    work = tmp_path / "work"
    prepare(tiny_cfg, work, fetch=fake_fetch, workers=1)
    return work


def test_prepare_writes_consistent_arrays(prepared):
    data = prepared / "data"
    meta = json.loads((data / "meta.json").read_text())
    assert meta["classes"] == CLASSES
    assert meta["counts"] == {"train": 3 * 40, "val": 3 * 8, "test": 3 * 12}

    x_train, y_train = np.load(data / "X_train.npy"), np.load(data / "y_train.npy")
    assert x_train.shape == (120, SIZE, SIZE) and x_train.dtype == np.uint8
    assert sorted(np.bincount(y_train)) == [40, 40, 40]
    assert list(y_train) != sorted(y_train)  # shuffled across classes
    assert x_train.reshape(120, -1).max(axis=1).min() > 200  # every picture has ink

    y_test = np.load(data / "y_test.npy")
    assert list(y_test) == sorted(y_test)  # evaluation splits stay ordered by class
    assert (data / "samples.png").exists()


def test_prepare_records_which_drawings_are_in_the_test_set(prepared):
    data = prepared / "data"
    meta = json.loads((data / "meta.json").read_text())
    assert len(meta["fingerprint"]) == 16
    with gzip.open(data / "test_ids.json.gz", "rt") as f:
        test_ids = json.load(f)
    assert [len(ids) for ids in test_ids] == [12, 12, 12]
    assert all(i.startswith(name) for name, ids in zip(CLASSES, test_ids, strict=True) for i in ids)


def test_fingerprint_changes_with_anything_that_defines_the_test_set(tiny_cfg):
    base = data_fingerprint(tiny_cfg, CLASSES, 60, [["a"], ["b"], ["c"]])
    assert base == data_fingerprint(tiny_cfg, CLASSES, 60, [["a"], ["b"], ["c"]])
    assert base != data_fingerprint(tiny_cfg, CLASSES, 60, [["a"], ["b"], ["x"]])  # other drawings
    assert base != data_fingerprint(tiny_cfg, CLASSES[:2], 60, [["a"], ["b"]])  # other classes
    assert base != data_fingerprint(tiny_cfg, CLASSES, 90, [["a"], ["b"], ["c"]])  # other size
    other = copy.deepcopy(tiny_cfg)
    other["render"]["line_width"] += 1
    assert base != data_fingerprint(other, CLASSES, 60, [["a"], ["b"], ["c"]])  # other rendering


def test_prepare_is_deterministic(tiny_cfg, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    prepare(tiny_cfg, a, fetch=fake_fetch, workers=1)
    prepare(tiny_cfg, b, fetch=fake_fetch, workers=1)
    for name in ("X_train", "y_train", "X_val", "X_test"):
        assert np.array_equal(
            np.load(a / "data" / f"{name}.npy"), np.load(b / "data" / f"{name}.npy")
        )


def test_history_plot_survives_a_diverged_run(tmp_path):
    pytest.importorskip("matplotlib")
    import evaluate

    history = tmp_path / "history.csv"
    history.write_text(
        "epoch,accuracy,loss,top3,val_accuracy,val_loss,val_top3\n"
        "0,0.7,1.8,0.86,0.85,1.3,0.96\n"
        "1,0.76,nan,0.84,0.008,nan,0.0\n"
    )
    out = tmp_path / "plot.png"
    evaluate.save_history_plot(history, out, "run")
    assert out.stat().st_size > 1000


def test_training_that_diverges_fails_instead_of_wasting_the_run(tiny_cfg, prepared, monkeypatch):
    import train

    cfg = copy.deepcopy(tiny_cfg)
    cfg["train"]["learning_rate"] = float("nan")  # guarantees a NaN loss on the first step
    with pytest.raises(RuntimeError, match="diverged"):
        train.train(cfg, prepared, name="nan", epochs=2)


def test_train_evaluate_and_resume(tiny_cfg, prepared, monkeypatch):
    import evaluate
    import train

    monkeypatch.setenv("DOODLEAI_RUN_ID", "20261111-000000-test")

    run = train.train(tiny_cfg, prepared, name="t", epochs=1)
    assert (run / "best.keras").exists() and (run / "last.keras").exists()
    assert len((run / "history.csv").read_text().strip().splitlines()) == 2  # header + 1 epoch

    train.train(tiny_cfg, prepared, name="t", epochs=2, resume=True)
    assert len((run / "history.csv").read_text().strip().splitlines()) == 3

    metrics = evaluate.evaluate(run / "best.keras", prepared / "data", run / "report")
    assert metrics["temperature"] > 0 and "ece_calibrated" in metrics  # fitted on the val split
    config = json.loads((run / "config.json").read_text())
    fingerprint = json.loads((prepared / "data" / "meta.json").read_text())["fingerprint"]
    if (run / "report" / "training_history.png").exists():  # needs matplotlib (training group)
        assert (run / "report" / "training_history.png").stat().st_size > 1000
    assert config["run_id"] == metrics["run_id"] == "20261111-000000-test"
    assert config["data_fingerprint"] == metrics["data_fingerprint"] == fingerprint
    assert metrics["image_size"] == SIZE
    assert metrics["samples"] == 36
    assert 0 <= metrics["top1"] <= metrics["top3"] <= 1
    assert set(metrics["per_class_accuracy"]) == set(CLASSES)
    assert (run / "report" / "metrics.json").exists()


def test_log_loss_is_plain_cross_entropy_without_label_smoothing():
    import evaluate

    certain = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert evaluate.log_loss(certain, np.array([0, 1])) == pytest.approx(0.0, abs=1e-9)
    half = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert evaluate.log_loss(half, np.array([0, 1])) == pytest.approx(math.log(2))
    wrong = np.array([[0.0, 1.0]])
    assert evaluate.log_loss(wrong, np.array([0])) > 20  # confidently wrong is punished hard
