"""Guards the model in production: the card must describe the file, and a pull request may only
replace the model in ``main`` with a better one (see registry.check_promotion)."""

import json
import os
from pathlib import Path

import pytest

import registry

CARD = registry.ARTIFACTS_DIR / registry.CARD_NAME


@pytest.fixture
def card():
    if not CARD.exists():
        pytest.skip("no model_card.json yet: the model predates the promotion workflow")
    return json.loads(CARD.read_text(encoding="utf-8"))


def test_card_describes_the_model_file(card):
    model = registry.ARTIFACTS_DIR / card["model_file"]
    assert model.exists()
    assert registry.sha256_file(model) == card["model_sha256"]


def test_card_describes_the_served_onnx_file(card):
    if "onnx_file" not in card:
        pytest.skip("the card predates the ONNX export")
    served = registry.ARTIFACTS_DIR / card["onnx_file"]
    assert served.exists()
    assert registry.sha256_file(served) == card["onnx_sha256"]


def test_card_classes_match_the_served_classes(card):
    mapping = json.loads((registry.ARTIFACTS_DIR / "classes.json").read_text(encoding="utf-8"))
    assert list(card["per_class_accuracy"]) == [mapping[str(i)] for i in range(len(mapping))]


def test_card_has_what_the_gate_needs(card):
    assert card["data_fingerprint"] and card["run_id"] and card["git_commit"]
    assert 0 < card["metrics"]["top1"] <= 1


def test_a_changed_model_is_better_than_the_one_on_main(card):
    """CI puts the card of the base branch in BASE_MODEL_CARD; locally this is skipped."""
    base_path = os.environ.get("BASE_MODEL_CARD")
    if not base_path or not Path(base_path).exists() or Path(base_path).stat().st_size == 0:
        pytest.skip("no model card on the base branch (or not running for a pull request)")
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    if base["run_id"] == card["run_id"]:
        return  # the model is unchanged
    ok, reasons = registry.check_promotion(card, base, registry.promotion_rules())
    assert ok, "; ".join(reasons)
