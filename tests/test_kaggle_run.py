import json
from types import SimpleNamespace

import pytest

import kaggle_run
import registry


def notebook_text():
    return kaggle_run.NOTEBOOK.read_text(encoding="utf-8")


def source_lines(text):
    notebook = json.loads(text)
    return [line for c in notebook["cells"] if c["cell_type"] == "code" for line in c["source"]]


# --- notebook and metadata ------------------------------------------------------------


@pytest.mark.parametrize("quick", [True, False])
def test_render_notebook_fills_in_mode_ref_and_run_id(quick):
    text = kaggle_run.render_notebook(notebook_text(), quick=quick, ref="abc123", run_id="r-1")
    lines = source_lines(text)
    assert any(line.startswith(f"QUICK = {quick}") for line in lines)
    assert any(line.startswith('REF = "abc123"') for line in lines)
    assert any(line.startswith('RUN_ID = "r-1"') for line in lines)


def test_render_notebook_keeps_the_rest_of_the_notebook_intact():
    original = json.loads(notebook_text())
    text = kaggle_run.render_notebook(notebook_text(), quick=False, ref="x", run_id="r")
    rendered = json.loads(text)
    assert len(rendered["cells"]) == len(original["cells"])
    assert rendered["metadata"] == original["metadata"]


@pytest.mark.parametrize("marker", ["QUICK = ", "REF = ", "RUN_ID = "])
def test_render_notebook_refuses_a_notebook_that_lost_a_setting(marker):
    broken = notebook_text().replace(marker, "# renamed = ", 1)
    with pytest.raises(SystemExit):
        kaggle_run.render_notebook(broken, quick=True, ref="x", run_id="r")


def test_notebook_passes_the_run_id_to_the_training_scripts():
    lines = source_lines(notebook_text())
    assert any(line.startswith('os.environ["DOODLEAI_RUN_ID"] = RUN_ID') for line in lines)


# --- names and remembered run -----------------------------------------------------------


# --- status handling --------------------------------------------------------------------


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
    monkeypatch.setattr(kaggle_run, "poll_status", lambda _k: queue.pop(0) if queue else "running")
    return queue


def test_wait_for_returns_when_the_run_finishes(monkeypatch):
    FakeClock(monkeypatch)
    queue = scripted_statuses(monkeypatch, ["queued", "running", "running", "complete"])
    assert kaggle_run.wait_for("someone/x", timeout=3600) == "complete"
    assert queue == []


def test_wait_for_gives_up_after_the_timeout(monkeypatch):
    FakeClock(monkeypatch)
    scripted_statuses(monkeypatch, [])  # always "running"
    assert kaggle_run.wait_for("someone/x", timeout=300) == "running"


def test_poll_status_survives_a_few_failures(monkeypatch):
    FakeClock(monkeypatch)
    outcomes = [SystemExit("network down"), SystemExit("network down"), "running"]

    def flaky(_kernel):
        outcome = outcomes.pop(0)
        if isinstance(outcome, SystemExit):
            raise outcome
        return outcome

    monkeypatch.setattr(kaggle_run, "current_status", flaky)
    assert kaggle_run.poll_status("someone/x") == "running"


def test_poll_status_gives_up_after_too_many_failures(monkeypatch):
    FakeClock(monkeypatch)

    def always_down(_kernel):
        raise SystemExit("network down")

    monkeypatch.setattr(kaggle_run, "current_status", always_down)
    with pytest.raises(SystemExit):
        kaggle_run.poll_status("someone/x")


# --- commands -----------------------------------------------------------------------------


@pytest.fixture
def pushing(monkeypatch, tmp_path):
    """cmd_push with the Kaggle CLI replaced by a recorder."""
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setattr(kaggle_run, "STATE_FILE", tmp_path / "state.json")
    calls = []

    def fake_kaggle(*args, **_kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="Kernel version 5 successfully pushed.", stderr="")

    monkeypatch.setattr(kaggle_run, "kaggle", fake_kaggle)
    return calls


def test_push_saves_a_version_and_remembers_it(pushing):
    args = SimpleNamespace(mode="full", ref="main", resume_from=None)
    kernel = kaggle_run.cmd_push(args)

    assert kernel == "someone/doodleai-train/5"
    push = pushing[0]
    assert push[:2] == ("kernels", "push")
    assert "--accelerator" in push and kaggle_run.ACCELERATOR in push
    state = kaggle_run.load_state()
    assert state["version"] == 5 and state["mode"] == "full" and state["ref"] == "main"
    assert state["run_id"].endswith("-full")


def test_run_pushes_waits_and_downloads(monkeypatch):
    FakeClock(monkeypatch)
    scripted_statuses(monkeypatch, ["queued", "running", "complete"])
    steps = []
    monkeypatch.setattr(kaggle_run, "cmd_push", lambda args: steps.append("push") or "me/nb/2")
    monkeypatch.setattr(kaggle_run, "cmd_output", lambda args: steps.append(args.kernel))

    kaggle_run.cmd_run(SimpleNamespace(mode="quick", dir="out", timeout=3600))

    assert steps == ["push", "me/nb/2"]  # downloads the version it just pushed


def test_run_does_not_download_a_failed_run(monkeypatch):
    FakeClock(monkeypatch)
    scripted_statuses(monkeypatch, ["running", "error"])
    monkeypatch.setattr(kaggle_run, "cmd_push", lambda args: "me/nb/2")
    monkeypatch.setattr(kaggle_run, "cmd_output", lambda args: pytest.fail("must not download"))

    with pytest.raises(SystemExit, match="error"):
        kaggle_run.cmd_run(SimpleNamespace(mode="quick", dir="out", timeout=3600))


def test_collect_does_not_download_while_the_run_is_going(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(kaggle_run, "current_status", lambda _k: "running")
    monkeypatch.setattr(kaggle_run, "kaggle", lambda *a, **k: pytest.fail("must not download"))

    kaggle_run.cmd_collect(SimpleNamespace(dir=str(tmp_path / "out"), kernel="someone/x"))

    assert "still 'running'" in capsys.readouterr().out


def test_collect_downloads_once_the_run_has_finished(monkeypatch):
    monkeypatch.setattr(kaggle_run, "current_status", lambda _k: "complete")
    downloaded = []
    monkeypatch.setattr(kaggle_run, "cmd_output", lambda args: downloaded.append(args.dir))

    kaggle_run.cmd_collect(SimpleNamespace(dir="out", kernel="someone/x"))

    assert downloaded == ["out"]


def test_output_saves_a_record_of_the_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(registry, "EXPERIMENTS_DIR", experiments)

    destination = tmp_path / "out"

    def fake_download(*args, **_kwargs):
        run_dir = destination / "runs" / "main"
        (run_dir / "report").mkdir(parents=True)
        metrics = {
            "samples": 10,
            "top1": 0.9,
            "top3": 0.99,
            "macro_f1": 0.9,
            "ece": 0.05,
            "per_class_accuracy": {"cat": 0.9},
            "most_confused": [],
            "run_id": "r-1",
            "data_fingerprint": "fp",
            "image_size": 64,
        }
        (run_dir / "report" / "metrics.json").write_text(json.dumps(metrics))
        config = {"run_id": "r-1", "git_commit": "abc", "config": {"train": {"epochs": 1}}}
        (run_dir / "config.json").write_text(json.dumps(config))
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(kaggle_run, "kaggle", fake_download)
    kaggle_run.cmd_output(SimpleNamespace(dir=str(destination), kernel="someone/nb/3"))

    saved = json.loads((experiments / "r-1.json").read_text())
    assert saved["kaggle"] == {"notebook": "someone/nb", "version": 3}
    assert saved["metrics"]["top1"] == 0.9
    assert "Run record saved" in capsys.readouterr().out
