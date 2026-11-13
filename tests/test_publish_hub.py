import json

import pytest

import publish_hub
import registry


def make_card():
    return {
        "run_id": "20261113-105549-full",
        "image_size": 64,
        "metrics": {
            "samples": 120000,
            "top1": 0.9,
            "top3": 0.97,
            "macro_f1": 0.89,
            "ece": 0.04,
        },
        "per_class_accuracy": {
            "cat": 0.95,
            "dog": 0.5,
            "bee": 0.9,
            "owl": 0.6,
            "fox": 0.99,
            "ant": 0.8,
        },
    }


def test_the_card_is_filled_with_the_numbers_of_the_model():
    text = publish_hub.render_card(make_card())
    assert "{{" not in text
    assert "20261113-105549-full" in text
    assert "90.0%" in text and "97.0%" in text and "0.890" in text and "0.040" in text
    assert "6 categories" in text and "120 000 drawings" in text
    assert "dog (50%)" in text  # the hardest class is named
    assert text.startswith("---\n") and "library_name: onnx" in text  # Hub metadata


def test_a_template_with_an_unknown_placeholder_is_refused():
    with pytest.raises(ValueError, match="placeholder"):
        publish_hub.render_card(make_card(), template="{{NOT_A_FIELD}}")


@pytest.fixture
def artifacts(tmp_path):
    folder = tmp_path / "artifacts"
    folder.mkdir()
    (folder / registry.MODEL_NAME).write_bytes(b"keras")
    (folder / registry.ONNX_NAME).write_bytes(b"onnx")
    (folder / "classes.json").write_text("{}")
    (folder / registry.CARD_NAME).write_text(json.dumps(make_card()))
    return folder


def test_staging_collects_the_models_the_card_and_the_plot(tmp_path, artifacts):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "training_history.png").write_bytes(b"png")
    names = publish_hub.stage(artifacts, reports, tmp_path / "out")
    assert set(names) == {
        "best_model.keras",
        "best_model.onnx",
        "model_card.json",
        "classes.json",
        "training_history.png",
        "README.md",
    }
    assert (tmp_path / "out" / "best_model.onnx").read_bytes() == b"onnx"


def test_the_plot_is_optional(tmp_path, artifacts):
    names = publish_hub.stage(artifacts, tmp_path / "no-reports", tmp_path / "out")
    assert "training_history.png" not in names and "README.md" in names


def test_a_dry_run_uploads_nothing_and_publishing_needs_a_promoted_model(
    tmp_path, artifacts, capsys
):
    run_id = publish_hub.publish("me/model", artifacts, tmp_path / "r", dry_run=True)
    assert run_id == "20261113-105549-full"
    assert "dry run" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="promote a model first"):
        publish_hub.publish("me/model", tmp_path / "empty", tmp_path / "r", dry_run=True)
