"""Run the training notebook on Kaggle from your own machine.

    export KAGGLE_API_TOKEN=...      # kaggle.com/settings/api -> Generate New Token
    export KAGGLE_USERNAME=your-kaggle-username

    python model/training/kaggle_run.py run --mode quick      # push, wait, download: minutes
    python model/training/kaggle_run.py run --mode full       # the same for the real run: hours

or step by step:

    python model/training/kaggle_run.py push --mode full
    python model/training/kaggle_run.py status
    python model/training/kaggle_run.py wait                  # blocks until it finishes
    python model/training/kaggle_run.py collect --dir out     # downloads if finished, else says so
    python model/training/kaggle_run.py compare               # all recorded runs, ranked

There is one Kaggle notebook, ``<username>/doodleai-train``. Every ``push`` saves a new *version* of
it and starts that version; a version is one training run. A run is identified by its ``run_id``
(date, time and mode, e.g. ``20261111-102305-quick``), which is written into its results. The
latest run is remembered in ``.kaggle-run.json`` and is what ``status``, ``wait``, ``collect`` and
``output`` use unless you pass ``--kernel``. ``output`` also saves a small record of the run in
``model/experiments/``; commit it to keep the run in the repository's history (see registry.py).

The notebook in the repository (``train.ipynb``) is the single source of truth: ``push`` uploads a
copy with ``QUICK``, ``REF`` and ``RUN_ID`` filled in, with a GPU and internet. Every step can be
repeated safely; the run itself keeps going on Kaggle if the terminal closes. Standard library
only; the ``kaggle`` command is taken from PATH or, failing that, run via ``uvx``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import registry

NOTEBOOK = Path(__file__).with_name("train.ipynb")
STATE_FILE = Path(__file__).resolve().parents[2] / ".kaggle-run.json"
SLUG = "doodleai-train"
ACCELERATOR = "NvidiaTeslaT4"  # Kaggle's "GPU T4 x2"
KAGGLE_VERSION = "2.2.5"
TIMEOUT_SECONDS = {"quick": 30 * 60, "full": 12 * 3600}
POLL_SECONDS = 60
MAX_POLL_FAILURES = 10  # consecutive failed status checks before giving up
NOTEBOOK_SETTINGS = ("QUICK", "REF", "RUN_ID")


def kaggle_command() -> list[str]:
    """The kaggle CLI: from PATH, else through uvx (Python 3.12; the CLI misses a ``six`` dep)."""
    found = shutil.which("kaggle")
    if found:
        return [found]
    return ["uvx", "--python", "3.12", "--with", "six", f"kaggle=={KAGGLE_VERSION}"]


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [*kaggle_command(), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result


def owner() -> str:
    name = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not name:
        sys.exit("Set KAGGLE_USERNAME to your Kaggle username.")
    return name


def new_run_id(mode: str, now: datetime) -> str:
    """``20261111-102305-quick``: unique per run, sortable, readable."""
    return f"{now:%Y%m%d-%H%M%S}-{mode}"


def with_owner(kernel: str) -> str:
    """``slug`` or ``slug/12`` -> ``<you>/slug[/12]``; a full ``owner/slug`` is kept."""
    parts = kernel.split("/")
    if len(parts) == 1 or (len(parts) == 2 and parts[1].isdigit()):
        return f"{owner()}/{kernel}"
    return kernel


def parse_version(text: str) -> int | None:
    """The version number from ``Kernel version 3 successfully pushed.``"""
    match = re.search(r"[Kk]ernel version (\d+) successfully pushed", text)
    return int(match.group(1)) if match else None


def save_state(state: dict, path: Path | None = None) -> None:
    (path or STATE_FILE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def load_state(path: Path | None = None) -> dict | None:
    path = path or STATE_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_kernel(args) -> str:
    """The run to act on: ``--kernel`` if given, else the latest run from ``push``/``run``."""
    explicit = getattr(args, "kernel", None)
    if explicit:
        return with_owner(explicit)
    state = load_state()
    if not state:
        sys.exit("No run yet: use `push` or `run` first, or pass --kernel <slug>.")
    version = state.get("version")
    return f"{state['id']}/{version}" if version else state["id"]


def render_notebook(text: str, *, quick: bool, ref: str, run_id: str) -> str:
    """Return the notebook JSON with QUICK, REF and RUN_ID set. Fails if one is missing."""
    values = {"QUICK": str(quick), "REF": json.dumps(ref), "RUN_ID": json.dumps(run_id)}
    patterns = {
        "QUICK": r"^QUICK = (True|False)",
        "REF": r'^REF = ".*?"',
        "RUN_ID": r'^RUN_ID = ".*?"',
    }
    notebook = json.loads(text)
    replaced = dict.fromkeys(NOTEBOOK_SETTINGS, 0)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            for name in NOTEBOOK_SETTINGS:
                if re.match(patterns[name], line):
                    cell["source"][i] = re.sub(patterns[name], f"{name} = {values[name]}", line)
                    replaced[name] += 1
    if any(count != 1 for count in replaced.values()):
        sys.exit(f"train.ipynb must have exactly one line each of {NOTEBOOK_SETTINGS}: {replaced}")
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def kernel_metadata(resume_from: str | None = None) -> dict:
    return {
        "id": f"{owner()}/{SLUG}",
        "title": SLUG,
        "code_file": "train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": ACCELERATOR,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [with_owner(resume_from)] if resume_from else [],
        "model_sources": [],
    }


def build_bundle(
    folder: Path, *, quick: bool, ref: str, run_id: str, resume_from: str | None = None
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    text = NOTEBOOK.read_text(encoding="utf-8")
    rendered = render_notebook(text, quick=quick, ref=ref, run_id=run_id)
    (folder / "train.ipynb").write_text(rendered, encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(resume_from), indent=2), encoding="utf-8"
    )


def parse_status(text: str) -> str:
    """Normalise ``... has status "KernelWorkerStatus.COMPLETE"`` to ``complete``."""
    match = re.search(r'status "([^"]+)"', text)
    if not match:
        return "unknown"
    return match.group(1).split(".")[-1].replace("_", "").lower()


def is_finished(status: str) -> bool:
    return status in {"complete", "error"} or status.startswith("cancel")


def current_status(kernel: str) -> str:
    return parse_status(kaggle("kernels", "status", kernel).stdout)


def poll_status(kernel: str) -> str:
    """The run status, tolerating short network or API hiccups during a multi-hour wait."""
    failures = 0
    while True:
        try:
            return current_status(kernel)
        except SystemExit as exc:
            failures += 1
            if failures >= MAX_POLL_FAILURES:
                raise
            print(f"Status check failed ({failures}/{MAX_POLL_FAILURES}), retrying: {exc}")
            time.sleep(POLL_SECONDS)


def wait_for(kernel: str, timeout: int) -> str:
    """Poll until the run finishes or ``timeout`` seconds pass; return the last status."""
    started = time.time()
    last = None
    while True:
        status = poll_status(kernel)
        if status != last:
            print(f"{time.strftime('%H:%M:%S')} {kernel}: {status}", flush=True)
            last = status
        if is_finished(status) or time.time() - started > timeout:
            return status
        time.sleep(POLL_SECONDS)


def finish_message(status: str, timeout: int) -> str:
    if is_finished(status):
        return f"Run ended with status '{status}'."
    return f"Still '{status}' after {timeout}s; run `wait` again to keep waiting."


def summarize_metrics(metrics: dict) -> str:
    """A short markdown summary of an evaluate.py ``metrics.json``."""
    rows = [
        "| metric | value |",
        "|---|---|",
        f"| top-1 | {metrics['top1']:.2%} |",
        f"| top-3 | {metrics['top3']:.2%} |",
        f"| macro F1 | {metrics['macro_f1']:.3f} |",
        f"| calibration error (ECE) | {metrics['ece']:.3f} |",
    ]
    if "log_loss" in metrics:
        rows.append(f"| log loss (no label smoothing) | {metrics['log_loss']:.3f} |")
    rows.append(f"| test samples | {metrics['samples']} |")
    hardest = sorted(metrics["per_class_accuracy"].items(), key=lambda kv: kv[1])[:5]
    lines = ["## Training result", "", *rows, "", "Hardest classes: "]
    lines[-1] += ", ".join(f"{name} ({acc:.0%})" for name, acc in hardest)
    for pair in metrics["most_confused"][:3]:
        lines.append(f"- {pair['true']} mistaken for {pair['predicted']}: {pair['count']}")
    return "\n".join(lines) + "\n"


def cmd_push(args) -> str:
    run_id = new_run_id(args.mode, datetime.now())
    with tempfile.TemporaryDirectory() as tmp:
        build_bundle(
            Path(tmp),
            quick=args.mode == "quick",
            ref=args.ref,
            run_id=run_id,
            resume_from=args.resume_from,
        )
        result = kaggle(
            "kernels",
            "push",
            "-p",
            tmp,
            "--accelerator",
            ACCELERATOR,
            "--timeout",
            str(TIMEOUT_SECONDS[args.mode]),
        )
    version = parse_version(result.stdout + result.stderr)
    state = {
        "id": f"{owner()}/{SLUG}",
        "version": version,
        "run_id": run_id,
        "mode": args.mode,
        "ref": args.ref,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    save_state(state)
    print(result.stdout.strip())
    print(f"Started run {run_id} (version {version}) of {args.ref}")
    print(f"https://www.kaggle.com/code/{state['id']}")
    return resolve_kernel(argparse.Namespace(kernel=None))


def cmd_status(args) -> None:
    print(current_status(resolve_kernel(args)))


def cmd_wait(args) -> None:
    status = wait_for(resolve_kernel(args), args.timeout)
    if status != "complete":
        sys.exit(finish_message(status, args.timeout))


def kaggle_info(kernel: str) -> dict:
    parts = kernel.split("/")
    version = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
    return {"notebook": "/".join(parts[:2]), "version": version}


def cmd_output(args) -> None:
    kernel = resolve_kernel(args)
    destination = Path(args.dir)
    destination.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel, "-p", str(destination), "-o")
    found = sorted(destination.rglob("metrics.json"))
    if not found:
        sys.exit(f"No metrics.json under {destination}; did the run finish evaluation?")
    metrics_path = found[-1]
    print(summarize_metrics(json.loads(metrics_path.read_text(encoding="utf-8"))))
    try:
        record = registry.build_record(metrics_path.parent.parent, kaggle_info(kernel))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"No experiment record saved: {exc}")
        return
    path = registry.save_record(record)
    print(f"Run record saved to {path}\nCommit it to keep this run in the repository's history.")


def cmd_collect(args) -> None:
    """Single-shot: download the result if the run has finished, otherwise just say so."""
    status = current_status(resolve_kernel(args))
    if not is_finished(status):
        print(f"The run is still '{status}': nothing to collect yet. Try again later.")
        return
    cmd_output(args)


def cmd_run(args) -> None:
    """Push a new version, wait for the run to finish, download the result."""
    kernel = cmd_push(args)
    status = wait_for(kernel, args.timeout)
    if status != "complete":
        sys.exit(finish_message(status, args.timeout))
    args.kernel = kernel
    cmd_output(args)


def cmd_compare(_args) -> None:
    print(registry.comparison_table(registry.load_records()))


def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--ref", default="main", help="git branch, tag or commit to train from")
    parser.add_argument("--resume-from", help="notebook slug whose output to continue from")


def add_target_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kernel", help="notebook[/version] (default: the latest run)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    push = commands.add_parser("push", help="save a new notebook version and start it")
    add_run_options(push)
    push.set_defaults(func=cmd_push)

    run = commands.add_parser("run", help="push, wait for the run to finish, download the result")
    add_run_options(run)
    run.add_argument("--dir", default="kaggle-output")
    run.add_argument("--timeout", type=int, default=13 * 3600, help="seconds to wait in total")
    run.set_defaults(func=cmd_run)

    status = commands.add_parser("status", help="print the run status")
    add_target_option(status)
    status.set_defaults(func=cmd_status)

    wait = commands.add_parser("wait", help="block until the run finishes")
    add_target_option(wait)
    wait.add_argument("--timeout", type=int, default=13 * 3600, help="seconds to wait")
    wait.set_defaults(func=cmd_wait)

    output = commands.add_parser("output", help="download the run's files, save its record")
    add_target_option(output)
    output.add_argument("--dir", default="kaggle-output")
    output.set_defaults(func=cmd_output)

    collect = commands.add_parser("collect", help="like output, but only if the run has finished")
    add_target_option(collect)
    collect.add_argument("--dir", default="kaggle-output")
    collect.set_defaults(func=cmd_collect)

    commands.add_parser("compare", help="rank all recorded runs").set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
