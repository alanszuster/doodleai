import json

import pytest

import kaggle_run


def notebook_text():
    return kaggle_run.NOTEBOOK.read_text(encoding="utf-8")


def source_lines(text):
    notebook = json.loads(text)
    return [line for c in notebook["cells"] if c["cell_type"] == "code" for line in c["source"]]


@pytest.mark.parametrize("quick", [True, False])
def test_render_notebook_sets_mode_and_ref(quick):
    lines = source_lines(kaggle_run.render_notebook(notebook_text(), quick=quick, ref="abc123"))
    assert any(line.startswith(f"QUICK = {quick}") for line in lines)
    assert any(line.startswith('REF = "abc123"') for line in lines)


def test_render_notebook_keeps_the_rest_of_the_notebook_intact():
    original = json.loads(notebook_text())
    rendered = json.loads(kaggle_run.render_notebook(notebook_text(), quick=False, ref="x"))
    assert len(rendered["cells"]) == len(original["cells"])
    assert rendered["metadata"] == original["metadata"]


@pytest.mark.parametrize("marker", ["QUICK = ", "REF = "])
def test_render_notebook_refuses_a_notebook_that_lost_its_settings(marker):
    broken = notebook_text().replace(marker, "# renamed = ", 1)
    with pytest.raises(SystemExit):
        kaggle_run.render_notebook(broken, quick=True, ref="x")


def test_bundle_has_a_notebook_and_gpu_metadata(tmp_path):
    kaggle_run.build_bundle(tmp_path, "someone/doodleai-train", quick=True, ref="main")
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert (tmp_path / meta["code_file"]).exists()
    assert meta["id"] == "someone/doodleai-train"
    assert meta["id"].split("/")[1] == meta["title"]  # Kaggle ties the slug to the title
    assert meta["enable_gpu"] == "true" and meta["enable_internet"] == "true"
    assert meta["is_private"] == "true"
    assert meta["machine_shape"] == kaggle_run.ACCELERATOR


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('someone/x has status "KernelWorkerStatus.COMPLETE"', "complete"),
        ('someone/x has status "complete"', "complete"),
        ('someone/x has status "running"', "running"),
        ('someone/x has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', "cancelacknowledged"),
        ("something unexpected", "unknown"),
    ],
)
def test_parse_status(text, expected):
    assert kaggle_run.parse_status(text) == expected


def test_is_finished():
    assert all(kaggle_run.is_finished(s) for s in ("complete", "error", "cancelacknowledged"))
    assert not any(kaggle_run.is_finished(s) for s in ("queued", "running", "unknown"))


def test_kernel_id_needs_a_username(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    with pytest.raises(SystemExit):
        kaggle_run.kernel_id()
    monkeypatch.setenv("KAGGLE_USERNAME", " someone ")
    assert kaggle_run.kernel_id() == "someone/doodleai-train"


def test_kaggle_command_falls_back_to_uvx(monkeypatch):
    monkeypatch.setattr(kaggle_run.shutil, "which", lambda _name: None)
    command = kaggle_run.kaggle_command()
    assert command[0] == "uvx" and f"kaggle=={kaggle_run.KAGGLE_VERSION}" in command
    monkeypatch.setattr(kaggle_run.shutil, "which", lambda _name: "/usr/bin/kaggle")
    assert kaggle_run.kaggle_command() == ["/usr/bin/kaggle"]


def test_summarize_metrics():
    metrics = {
        "samples": 100,
        "top1": 0.8,
        "top3": 0.95,
        "macro_f1": 0.79,
        "ece": 0.05,
        "per_class_accuracy": {"cat": 0.5, "dog": 0.9, "bee": 1.0},
        "most_confused": [{"true": "cat", "predicted": "dog", "count": 7}],
    }
    summary = kaggle_run.summarize_metrics(metrics)
    assert "80.00%" in summary and "95.00%" in summary
    assert summary.index("cat (50%)") < summary.index("dog (90%)")  # hardest first
    assert "cat mistaken for dog: 7" in summary


def test_collect_does_not_download_while_the_run_is_going(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setattr(kaggle_run, "current_status", lambda _id: "running")
    monkeypatch.setattr(kaggle_run, "kaggle", lambda *a, **k: pytest.fail("must not download"))

    kaggle_run.cmd_collect(type("Args", (), {"dir": str(tmp_path / "out")}))

    assert "still 'running'" in capsys.readouterr().out


def test_collect_downloads_once_the_run_has_finished(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setattr(kaggle_run, "current_status", lambda _id: "complete")
    downloaded = []
    monkeypatch.setattr(kaggle_run, "cmd_output", lambda args: downloaded.append(args.dir))

    kaggle_run.cmd_collect(type("Args", (), {"dir": "out"}))

    assert downloaded == ["out"]


class FakeClock:
    """Replaces time.time and time.sleep so waiting loops run instantly."""

    def __init__(self, monkeypatch):
        self.now = 1000.0
        monkeypatch.setattr(kaggle_run.time, "time", lambda: self.now)
        monkeypatch.setattr(kaggle_run.time, "sleep", self.sleep)

    def sleep(self, seconds):
        self.now += seconds


def scripted_statuses(monkeypatch, statuses):
    queue = list(statuses)
    monkeypatch.setattr(kaggle_run, "poll_status", lambda _id: queue.pop(0) if queue else "running")
    return queue


def test_wait_for_returns_when_the_run_finishes(monkeypatch):
    FakeClock(monkeypatch)
    scripted_statuses(monkeypatch, ["queued", "running", "running", "complete"])
    assert kaggle_run.wait_for("someone/x", timeout=3600) == "complete"


def test_wait_for_gives_up_after_the_timeout(monkeypatch):
    FakeClock(monkeypatch)
    scripted_statuses(monkeypatch, [])  # always "running"
    assert kaggle_run.wait_for("someone/x", timeout=300) == "running"


def test_wait_for_ignores_the_previous_runs_status_right_after_a_push(monkeypatch):
    FakeClock(monkeypatch)
    queue = scripted_statuses(
        monkeypatch,
        ["complete", "complete", "queued", "running", "complete"],
    )
    assert kaggle_run.wait_for("someone/x", timeout=3600, stale="complete") == "complete"
    assert queue == []  # the first "complete"s belonged to the old run and did not end the wait


def test_wait_for_accepts_a_stale_looking_status_after_the_grace_period(monkeypatch):
    FakeClock(monkeypatch)
    scripted_statuses(monkeypatch, ["error"] * 10)  # a run that really failed instantly
    assert kaggle_run.wait_for("someone/x", timeout=3600, stale="error") == "error"


def test_poll_status_survives_a_few_failures(monkeypatch):
    FakeClock(monkeypatch)
    outcomes = [SystemExit("network down"), SystemExit("network down"), "running"]

    def flaky(_id):
        outcome = outcomes.pop(0)
        if isinstance(outcome, SystemExit):
            raise outcome
        return outcome

    monkeypatch.setattr(kaggle_run, "current_status", flaky)
    assert kaggle_run.poll_status("someone/x") == "running"


def test_poll_status_gives_up_after_too_many_failures(monkeypatch):
    FakeClock(monkeypatch)

    def always_down(_id):
        raise SystemExit("network down")

    monkeypatch.setattr(kaggle_run, "current_status", always_down)
    with pytest.raises(SystemExit):
        kaggle_run.poll_status("someone/x")


def test_run_pushes_waits_and_downloads(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    FakeClock(monkeypatch)
    # first status is read before the push (the previous run); a real run takes minutes
    scripted_statuses(monkeypatch, ["complete", "running", "running", "running", "complete"])
    steps = []
    monkeypatch.setattr(kaggle_run, "cmd_push", lambda args: steps.append("push"))
    monkeypatch.setattr(kaggle_run, "cmd_output", lambda args: steps.append("output"))

    args = type("Args", (), {"mode": "quick", "ref": "main", "dir": "out", "timeout": 3600})
    kaggle_run.cmd_run(args)

    assert steps == ["push", "output"]


def test_run_does_not_download_a_failed_run(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    FakeClock(monkeypatch)
    scripted_statuses(monkeypatch, ["complete", "running", "error"])
    monkeypatch.setattr(kaggle_run, "cmd_push", lambda args: None)
    monkeypatch.setattr(kaggle_run, "cmd_output", lambda args: pytest.fail("must not download"))

    args = type("Args", (), {"mode": "quick", "ref": "main", "dir": "out", "timeout": 3600})
    with pytest.raises(SystemExit, match="error"):
        kaggle_run.cmd_run(args)
