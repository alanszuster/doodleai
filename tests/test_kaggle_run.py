import json
from datetime import datetime
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
    assert 'os.environ["DOODLEAI_RUN_ID"] = RUN_ID' in notebook_text().replace("\\n", "\n")


def test_bundle_has_a_notebook_and_gpu_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    kaggle_run.build_bundle(tmp_path, quick=True, ref="main", run_id="r-1")
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert (tmp_path / meta["code_file"]).exists()
    assert meta["id"] == "someone/doodleai-train"  # one notebook; every push is a version
    assert meta["id"].split("/")[1] == meta["title"]  # Kaggle ties the slug to the title
    assert meta["enable_gpu"] == "true" and meta["enable_internet"] == "true"
    assert meta["is_private"] == "true"
    assert meta["machine_shape"] == kaggle_run.ACCELERATOR
    assert meta["kernel_sources"] == []


def test_resume_from_attaches_the_earlier_notebook(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    kaggle_run.build_bundle(tmp_path, quick=False, ref="main", run_id="r", resume_from="old")
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == ["someone/old"]


# --- names and remembered run -----------------------------------------------------------


def test_run_ids_are_unique_sortable_and_readable():
    first = kaggle_run.new_run_id("quick", datetime(2026, 11, 11, 10, 23, 5))
    second = kaggle_run.new_run_id("quick", datetime(2026, 11, 11, 10, 23, 6))
    assert first == "20261111-102305-quick"
    assert first < second
    assert kaggle_run.new_run_id("full", datetime(2026, 11, 11, 10, 23, 5)).endswith("-full")


def test_owner_needs_a_username(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    with pytest.raises(SystemExit):
        kaggle_run.owner()
    monkeypatch.setenv("KAGGLE_USERNAME", " someone ")
    assert kaggle_run.owner() == "someone"


def test_with_owner_adds_the_owner_only_when_missing(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    assert kaggle_run.with_owner("x") == "someone/x"
    assert kaggle_run.with_owner("x/12") == "someone/x/12"
    assert kaggle_run.with_owner("other/x") == "other/x"
    assert kaggle_run.with_owner("other/x/12") == "other/x/12"


def test_parse_version():
    assert kaggle_run.parse_version("Kernel version 7 successfully pushed.  Please check") == 7
    assert kaggle_run.parse_version("something else") is None


def test_state_round_trip(tmp_path):
    path = tmp_path / "state.json"
    assert kaggle_run.load_state(path) is None
    kaggle_run.save_state({"id": "someone/x", "version": 2}, path)
    assert kaggle_run.load_state(path) == {"id": "someone/x", "version": 2}


def test_resolve_kernel_prefers_an_explicit_target_then_the_latest_version(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setattr(kaggle_run, "STATE_FILE", tmp_path / "state.json")

    with pytest.raises(SystemExit):  # nothing pushed yet
        kaggle_run.resolve_kernel(SimpleNamespace(kernel=None))

    kaggle_run.save_state({"id": "someone/doodleai-train", "version": 4})
    assert kaggle_run.resolve_kernel(SimpleNamespace(kernel=None)) == "someone/doodleai-train/4"
    assert kaggle_run.resolve_kernel(SimpleNamespace(kernel="older/3")) == "older/3"

    kaggle_run.save_state({"id": "someone/doodleai-train", "version": None})
    assert kaggle_run.resolve_kernel(SimpleNamespace(kernel=None)) == "someone/doodleai-train"


def test_kaggle_info_splits_notebook_and_version():
    assert kaggle_run.kaggle_info("me/x/5") == {"notebook": "me/x", "version": 5}
    assert kaggle_run.kaggle_info("me/x") == {"notebook": "me/x", "version": None}


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
def test_parse_status(text, expected):
    assert kaggle_run.parse_status(text) == expected


def test_is_finished():
    assert all(kaggle_run.is_finished(s) for s in ("complete", "error", "cancelacknowledged"))
    assert not any(kaggle_run.is_finished(s) for s in ("queued", "running", "unknown"))


def test_kaggle_command_falls_back_to_uvx(monkeypatch):
    monkeypatch.setattr(kaggle_run.shutil, "which", lambda _name: None)
    command = kaggle_run.kaggle_command()
    assert command[0] == "uvx" and f"kaggle=={kaggle_run.KAGGLE_VERSION}" in command
    monkeypatch.setattr(kaggle_run.shutil, "which", lambda _name: "/usr/bin/kaggle")
    assert kaggle_run.kaggle_command() == ["/usr/bin/kaggle"]


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
