import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import kaggle_hidden
import kaggle_run


def test_push_uses_separate_notebook_and_state(monkeypatch, tmp_path):
    source = (kaggle_run.NOTEBOOK.parent / "hidden.ipynb").read_text()
    state_path = tmp_path / "hidden-state.json"
    monkeypatch.setattr(kaggle_hidden, "STATE", state_path)
    monkeypatch.setattr(kaggle_run, "owner", lambda: "someone")
    monkeypatch.setattr(kaggle_run, "main_notebook_text", lambda name: source)

    def fake_kaggle(*args):
        assert args[:2] == ("kernels", "push")
        folder = Path(args[args.index("-p") + 1])
        metadata = json.loads((folder / "kernel-metadata.json").read_text())
        notebook = json.loads((folder / "train.ipynb").read_text())
        assert metadata["id"] == "someone/doodleai-hidden"
        assert metadata["enable_gpu"] == "true"
        assert any(
            line.startswith("RUN_ID = ") and "hidden" in line
            for cell in notebook["cells"]
            for line in cell["source"]
        )
        return SimpleNamespace(stdout="Kernel version 1 successfully pushed.", stderr="")

    monkeypatch.setattr(kaggle_run, "kaggle", fake_kaggle)
    kaggle_hidden.push()
    state = json.loads(state_path.read_text())
    assert state["id"] == "someone/doodleai-hidden"
    assert state["version"] == 1
    assert state["run_id"].endswith("-hidden")


@pytest.mark.parametrize("script", ["prepare_hidden.py", "train.py", "evaluate.py"])
def test_notebook_stops_on_pipeline_error(monkeypatch, tmp_path, script):
    notebook = json.loads((kaggle_run.NOTEBOOK.parent / "hidden.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if f"model/training/{script}" in "".join(cell["source"])
    )

    def failed(command, *, check):
        assert check is True
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        exec(
            source,
            {
                "os": os,
                "subprocess": subprocess,
                "sys": sys,
                "WORK": tmp_path,
                "RUNS": tmp_path,
                "DATA": tmp_path,
            },
        )


def test_collect_checks_run_identity_and_required_files(monkeypatch, tmp_path):
    state_path = tmp_path / "hidden-state.json"
    kaggle_run.save_state(
        {"id": "someone/doodleai-hidden", "version": 1, "run_id": "expected"}, state_path
    )
    monkeypatch.setattr(kaggle_hidden, "STATE", state_path)
    monkeypatch.setattr(kaggle_run, "current_status", lambda _: "complete")
    run_id = "wrong"

    def fake_download(*args):
        folder = Path(args[args.index("-p") + 1])
        files = (
            "runs/hidden/best.keras",
            "runs/hidden/config.json",
            "runs/hidden/report/metrics.json",
            "hidden-data/X_val.npy",
            "hidden-data/y_val.npy",
            "hidden-data/X_test.npy",
            "hidden-data/y_test.npy",
            "hidden-data/meta.json",
        )
        for name in files:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"run_id": run_id}) if name.endswith("metrics.json") else "")

    monkeypatch.setattr(kaggle_run, "kaggle", fake_download)
    with pytest.raises(SystemExit, match="run_id"):
        kaggle_hidden.collect(tmp_path / "out")
    assert not (tmp_path / "out").exists()
    run_id = "expected"
    kaggle_hidden.collect(tmp_path / "out")
    assert (tmp_path / "out/someone/doodleai-hidden/1/runs/hidden/best.keras").is_file()


def test_collect_does_not_download_unfinished_run(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_hidden, "target", lambda: ({"run_id": "r"}, "someone/hidden/1"))
    monkeypatch.setattr(kaggle_run, "current_status", lambda _: "running")
    monkeypatch.setattr(kaggle_run, "kaggle", lambda *args: pytest.fail("must not download"))
    with pytest.raises(SystemExit, match="running"):
        kaggle_hidden.collect(tmp_path)
