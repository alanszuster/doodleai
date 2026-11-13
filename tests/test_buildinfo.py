import json

import buildinfo


def test_model_info_reads_the_card(tmp_path, monkeypatch):
    card = tmp_path / "model_card.json"
    card.write_text(
        json.dumps({"run_id": "r1", "metrics": {"top1": 0.9}, "model_sha256": "abc", "extra": 1})
    )
    monkeypatch.setattr(buildinfo, "CARD_PATH", str(card))
    buildinfo.model_info.cache_clear()
    assert buildinfo.model_info() == {"run_id": "r1", "top1": 0.9, "sha256": "abc"}
    buildinfo.model_info.cache_clear()


def test_model_info_is_none_without_a_usable_card(tmp_path, monkeypatch):
    for content in (None, "not json"):
        card = tmp_path / "model_card.json"
        card.unlink(missing_ok=True)
        if content:
            card.write_text(content)
        monkeypatch.setattr(buildinfo, "CARD_PATH", str(card))
        buildinfo.model_info.cache_clear()
        assert buildinfo.model_info() is None
    buildinfo.model_info.cache_clear()
