import json

import pytest

import promote
import registry

RULES = {"margin": 0.005, "max_class_drop": 0.03, "max_ece_rise": 0.03}
CLASSES = ["cat", "dog", "bee"]


def make_record(run_id="r1", top1=0.80, ece=0.05, per_class=None, fingerprint="fp1", **extra):
    return {
        "run_id": run_id,
        "created": "2026-11-11T10:00:00",
        "kaggle": {},
        "git_commit": "abcdef123456",
        "data_fingerprint": fingerprint,
        "image_size": 64,
        "metrics": {"samples": 300, "top1": top1, "top3": 0.95, "macro_f1": 0.79, "ece": ece},
        "per_class_accuracy": per_class or {"cat": 0.8, "dog": 0.8, "bee": 0.8},
        "train_config": {"epochs": 20},
        **extra,
    }


def write_run_dir(path, record, model_bytes=b"model"):
    """A downloaded run directory as evaluate.py/train.py leave it."""
    (path / "report").mkdir(parents=True)
    metrics = {**record["metrics"], "per_class_accuracy": record["per_class_accuracy"]}
    metrics.update(
        run_id=record["run_id"],
        data_fingerprint=record["data_fingerprint"],
        image_size=record["image_size"],
    )
    (path / "report" / "metrics.json").write_text(json.dumps(metrics))
    config = {
        "run_id": record["run_id"],
        "git_commit": record["git_commit"],
        "config": {"train": record["train_config"]},
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "best.keras").write_bytes(model_bytes)
    return path


# --- records ---------------------------------------------------------------------------


def test_build_record_reads_a_run_directory(tmp_path):
    expected = make_record()
    run_dir = write_run_dir(tmp_path / "run", expected)
    record = registry.build_record(run_dir, {"notebook": "me/x", "version": 3})
    assert record["run_id"] == "r1" and record["data_fingerprint"] == "fp1"
    assert record["metrics"] == expected["metrics"]
    assert record["kaggle"] == {"notebook": "me/x", "version": 3}


def test_build_record_reports_epochs_and_flags_a_diverged_run(tmp_path):
    healthy = write_run_dir(tmp_path / "ok", make_record())
    (healthy / "history.csv").write_text("epoch,loss\n0,1.5\n1,1.2\n")
    assert registry.build_record(healthy)["epochs_run"] == 2
    assert registry.build_record(healthy)["diverged"] is False

    broken = write_run_dir(tmp_path / "nan", make_record())
    (broken / "history.csv").write_text("epoch,loss\n0,1.5\n1,nan\n2,nan\n")
    record = registry.build_record(broken)
    assert record["epochs_run"] == 3 and record["diverged"] is True

    no_history = write_run_dir(tmp_path / "none", make_record())
    assert registry.build_record(no_history)["diverged"] is False


def test_a_diverged_run_is_never_promoted():
    ok, reasons = registry.check_promotion(make_record(diverged=True), None, RULES)
    assert not ok and "diverged" in reasons[0]


def test_build_record_refuses_a_run_without_an_id(tmp_path):
    run_dir = write_run_dir(tmp_path / "run", make_record())
    config = json.loads((run_dir / "config.json").read_text())
    del config["run_id"]
    (run_dir / "config.json").write_text(json.dumps(config))
    metrics = json.loads((run_dir / "report" / "metrics.json").read_text())
    del metrics["run_id"]
    (run_dir / "report" / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="run_id"):
        registry.build_record(run_dir)


def test_records_round_trip_and_rank(tmp_path):
    for record in (make_record("run-a", top1=0.7), make_record("run-b", top1=0.9)):
        registry.save_record(record, tmp_path)
    loaded = registry.load_records(tmp_path)
    assert [r["run_id"] for r in loaded] == ["run-a", "run-b"]
    table = registry.comparison_table(loaded)
    assert table.index("run-b") < table.index("run-a")  # best first
    assert "not directly comparable" not in table


def test_comparison_flags_runs_with_a_different_test_set():
    records = [make_record("a", top1=0.9), make_record("b", top1=0.8, fingerprint="other")]
    table = registry.comparison_table(records)
    assert "not directly comparable" in table
    assert registry.comparison_table([]) == "No experiment records yet."


def test_card_carries_the_hash_of_the_model_file(tmp_path):
    model = tmp_path / "best_model.keras"
    model.write_bytes(b"abc")
    card = registry.make_card(make_record(), model)
    assert card["model_file"] == "best_model.keras"
    assert card["model_sha256"] == registry.sha256_file(model)
    model.write_bytes(b"abd")
    assert registry.sha256_file(model) != card["model_sha256"]


# --- the gate --------------------------------------------------------------------------


def test_first_model_is_accepted():
    ok, _ = registry.check_promotion(make_record(), None, RULES)
    assert ok


def test_models_on_different_test_sets_are_not_compared():
    ok, reasons = registry.check_promotion(make_record(fingerprint="a"), make_record(), RULES)
    assert not ok and "not comparable" in reasons[0]


def test_a_clearly_better_model_passes():
    ok, reasons = registry.check_promotion(make_record(top1=0.83), make_record(), RULES)
    assert ok and "+3.00" in reasons[0]


@pytest.mark.parametrize("top1", [0.80, 0.802, 0.70])
def test_a_model_that_is_not_clearly_better_is_rejected(top1):
    ok, reasons = registry.check_promotion(make_record(top1=top1), make_record(), RULES)
    assert not ok and "below the required" in reasons[0]


def test_a_model_that_wins_on_average_but_ruins_one_class_is_rejected():
    worse = {"cat": 0.99, "dog": 0.99, "bee": 0.60}
    candidate = make_record(top1=0.86, per_class=worse)
    ok, reasons = registry.check_promotion(candidate, make_record(), RULES)
    assert not ok and any("bee" in r for r in reasons)


def test_worse_calibration_is_rejected():
    ok, reasons = registry.check_promotion(make_record(top1=0.9, ece=0.2), make_record(), RULES)
    assert not ok and any("calibration" in r for r in reasons)


# --- promote ----------------------------------------------------------------------------


@pytest.fixture
def artifacts(tmp_path):
    folder = tmp_path / "artifacts"
    folder.mkdir()
    (folder / "classes.json").write_text(json.dumps(dict(enumerate(CLASSES))))
    return folder


def fake_export(keras_path, onnx_path):
    """Stands in for the TensorFlow export: 'converts' by tagging the bytes."""
    onnx_path.write_bytes(b"onnx:" + keras_path.read_bytes())


def try_promote(tmp_path, artifacts, record, model_bytes=b"model"):
    run_dir = write_run_dir(tmp_path / record["run_id"], record, model_bytes)
    return promote.promote(
        run_dir,
        artifacts_dir=artifacts,
        experiments_dir=tmp_path / "experiments",
        export=fake_export,
    )


def test_first_promotion_installs_the_model_card_and_record(tmp_path, artifacts):
    ok, _ = try_promote(tmp_path, artifacts, make_record("first"), b"model-1")
    assert ok
    assert (artifacts / registry.MODEL_NAME).read_bytes() == b"model-1"
    assert (artifacts / registry.ONNX_NAME).read_bytes() == b"onnx:model-1"
    card = json.loads((artifacts / registry.CARD_NAME).read_text())
    assert card["run_id"] == "first"
    assert card["model_sha256"] == registry.sha256_file(artifacts / registry.MODEL_NAME)
    assert card["onnx_sha256"] == registry.sha256_file(artifacts / registry.ONNX_NAME)
    assert (tmp_path / "experiments" / "first.json").exists()


def test_promotion_publishes_the_training_history_plot(tmp_path, artifacts):
    run_dir = write_run_dir(tmp_path / "r", make_record("first"))
    (run_dir / "report" / "training_history.png").write_bytes(b"png")
    reports = tmp_path / "reports"
    ok, _ = promote.promote(
        run_dir,
        artifacts_dir=artifacts,
        experiments_dir=tmp_path / "e",
        reports_dir=reports,
        export=fake_export,
    )
    assert ok and (reports / "training_history.png").read_bytes() == b"png"


def test_a_failing_export_leaves_production_untouched(tmp_path, artifacts):
    try_promote(tmp_path, artifacts, make_record("first", top1=0.8), b"model-1")

    def broken(keras_path, onnx_path):
        raise RuntimeError("export failed")

    run_dir = write_run_dir(tmp_path / "second", make_record("second", top1=0.9), b"model-2")
    with pytest.raises(RuntimeError, match="export failed"):
        promote.promote(
            run_dir,
            artifacts_dir=artifacts,
            experiments_dir=tmp_path / "e",
            export=broken,
        )
    assert (artifacts / registry.MODEL_NAME).read_bytes() == b"model-1"
    assert (artifacts / registry.ONNX_NAME).read_bytes() == b"onnx:model-1"
    assert json.loads((artifacts / registry.CARD_NAME).read_text())["run_id"] == "first"


def test_a_worse_model_leaves_production_untouched(tmp_path, artifacts):
    try_promote(tmp_path, artifacts, make_record("first", top1=0.8), b"model-1")
    ok, reasons = try_promote(tmp_path, artifacts, make_record("second", top1=0.79), b"model-2")
    assert not ok and reasons
    assert (artifacts / registry.MODEL_NAME).read_bytes() == b"model-1"
    assert json.loads((artifacts / registry.CARD_NAME).read_text())["run_id"] == "first"


def test_a_better_model_replaces_production(tmp_path, artifacts):
    try_promote(tmp_path, artifacts, make_record("first", top1=0.8), b"model-1")
    ok, _ = try_promote(tmp_path, artifacts, make_record("second", top1=0.85), b"model-2")
    assert ok
    assert (artifacts / registry.MODEL_NAME).read_bytes() == b"model-2"
    assert json.loads((artifacts / registry.CARD_NAME).read_text())["run_id"] == "second"


def test_dry_run_only_checks_the_gate(tmp_path, artifacts):
    run_dir = write_run_dir(tmp_path / "r", make_record("first"))
    ok, reasons = promote.promote(
        run_dir, artifacts_dir=artifacts, experiments_dir=tmp_path / "e", dry_run=True
    )
    assert ok and "dry run" in reasons[-1]
    assert not (artifacts / registry.MODEL_NAME).exists()


def test_a_model_with_other_classes_is_refused(tmp_path, artifacts):
    other = make_record("x", per_class={"cat": 0.8, "dog": 0.8, "emu": 0.8})
    ok, reasons = try_promote(tmp_path, artifacts, other)
    assert not ok and "classes" in reasons[0]
