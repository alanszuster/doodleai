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

Every step can be repeated safely, so if the terminal closes during a long run, start again with
``wait`` or ``collect``: the run itself keeps going on Kaggle.
    python model/training/kaggle_run.py output --dir out      # downloads files, prints metrics

The notebook in the repository (``train.ipynb``) is the single source of truth: ``push`` makes a
copy with ``QUICK`` and ``REF`` filled in and uploads it as the Kaggle notebook
``<username>/doodleai-train`` with a GPU and internet access. Standard library only; the
``kaggle`` command is taken from PATH or, failing that, run through ``uvx``.
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
from pathlib import Path

NOTEBOOK = Path(__file__).with_name("train.ipynb")
SLUG = "doodleai-train"  # Kaggle derives the slug from the title, so they must match
ACCELERATOR = "NvidiaTeslaT4"  # Kaggle's "GPU T4 x2"
KAGGLE_VERSION = "2.2.5"
TIMEOUT_SECONDS = {"quick": 30 * 60, "full": 12 * 3600}
POLL_SECONDS = 60
MAX_POLL_FAILURES = 10  # consecutive failed status checks before giving up
STALE_GRACE_SECONDS = 180  # after a push, the old run's status may still be shown


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


def kernel_id() -> str:
    owner = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not owner:
        sys.exit("Set KAGGLE_USERNAME to your Kaggle username.")
    return f"{owner}/{SLUG}"


def render_notebook(text: str, *, quick: bool, ref: str) -> str:
    """Return the notebook JSON with ``QUICK`` and ``REF`` set. Fails if either is missing."""
    notebook = json.loads(text)
    replaced = {"QUICK": 0, "REF": 0}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            if re.match(r"QUICK = (True|False)", line):
                cell["source"][i] = re.sub(r"^QUICK = (True|False)", f"QUICK = {quick}", line)
                replaced["QUICK"] += 1
            elif re.match(r'REF = ".*?"', line):
                cell["source"][i] = re.sub(r'^REF = ".*?"', f"REF = {json.dumps(ref)}", line)
                replaced["REF"] += 1
    if replaced != {"QUICK": 1, "REF": 1}:
        sys.exit(f"train.ipynb no longer has exactly one QUICK and one REF line: {replaced}")
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def kernel_metadata(identifier: str) -> dict:
    return {
        "id": identifier,
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
        "kernel_sources": [],
        "model_sources": [],
    }


def build_bundle(folder: Path, identifier: str, *, quick: bool, ref: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rendered = render_notebook(NOTEBOOK.read_text(encoding="utf-8"), quick=quick, ref=ref)
    (folder / "train.ipynb").write_text(rendered, encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(identifier), indent=2), encoding="utf-8"
    )


def parse_status(text: str) -> str:
    """Normalise ``... has status "KernelWorkerStatus.COMPLETE"`` to ``complete``."""
    match = re.search(r'status "([^"]+)"', text)
    if not match:
        return "unknown"
    return match.group(1).split(".")[-1].replace("_", "").lower()


def is_finished(status: str) -> bool:
    return status in {"complete", "error"} or status.startswith("cancel")


def current_status(identifier: str) -> str:
    return parse_status(kaggle("kernels", "status", identifier).stdout)


def summarize_metrics(metrics: dict) -> str:
    """A short markdown summary of an evaluate.py ``metrics.json``."""
    rows = [
        "| metric | value |",
        "|---|---|",
        f"| top-1 | {metrics['top1']:.2%} |",
        f"| top-3 | {metrics['top3']:.2%} |",
        f"| macro F1 | {metrics['macro_f1']:.3f} |",
        f"| calibration error (ECE) | {metrics['ece']:.3f} |",
        f"| test samples | {metrics['samples']} |",
    ]
    hardest = sorted(metrics["per_class_accuracy"].items(), key=lambda kv: kv[1])[:5]
    lines = ["## Training result", "", *rows, "", "Hardest classes: "]
    lines[-1] += ", ".join(f"{name} ({acc:.0%})" for name, acc in hardest)
    for pair in metrics["most_confused"][:3]:
        lines.append(f"- {pair['true']} mistaken for {pair['predicted']}: {pair['count']}")
    return "\n".join(lines) + "\n"


def cmd_push(args) -> None:
    identifier = kernel_id()
    with tempfile.TemporaryDirectory() as tmp:
        build_bundle(Path(tmp), identifier, quick=args.mode == "quick", ref=args.ref)
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
    print(result.stdout.strip())
    print(f"Started {args.mode} run of {args.ref} -> https://www.kaggle.com/code/{identifier}")


def cmd_status(_args) -> None:
    print(current_status(kernel_id()))


def poll_status(identifier: str) -> str:
    """The run status, tolerating short network or API hiccups during a multi-hour wait."""
    failures = 0
    while True:
        try:
            return current_status(identifier)
        except SystemExit as exc:
            failures += 1
            if failures >= MAX_POLL_FAILURES:
                raise
            print(f"Status check failed ({failures}/{MAX_POLL_FAILURES}), retrying: {exc}")
            time.sleep(POLL_SECONDS)


def wait_for(identifier: str, timeout: int, *, stale: str | None = None) -> str:
    """Poll until the run finishes or ``timeout`` seconds pass; return the last status.

    ``stale`` is the status seen before a push. Right after a push Kaggle may still report the
    previous run, so a finished status equal to it is ignored for a short grace period.
    """
    started = time.time()
    last = None
    while True:
        status = poll_status(identifier)
        if status != last:
            print(f"{time.strftime('%H:%M:%S')} {identifier}: {status}", flush=True)
            last = status
        waiting_out_stale = status == stale and time.time() - started < STALE_GRACE_SECONDS
        if is_finished(status) and not waiting_out_stale:
            return status
        if time.time() - started > timeout:
            return status
        time.sleep(POLL_SECONDS)


def finish_message(status: str, timeout: int) -> str:
    if is_finished(status):
        return f"Run ended with status '{status}'."
    return f"Still '{status}' after {timeout}s; run `wait` again to keep waiting."


def cmd_wait(args) -> None:
    status = wait_for(kernel_id(), args.timeout)
    if status != "complete":
        sys.exit(finish_message(status, args.timeout))


def cmd_run(args) -> None:
    """Push, wait for the run to finish, download the result."""
    identifier = kernel_id()
    before = poll_status(identifier)
    cmd_push(args)
    status = wait_for(identifier, args.timeout, stale=before)
    if status != "complete":
        sys.exit(finish_message(status, args.timeout))
    cmd_output(args)


def cmd_output(args) -> None:
    destination = Path(args.dir)
    destination.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel_id(), "-p", str(destination), "-o")
    found = sorted(destination.rglob("metrics.json"))
    if not found:
        sys.exit(f"No metrics.json under {destination}; did the run finish evaluation?")
    print(summarize_metrics(json.loads(found[-1].read_text(encoding="utf-8"))))


def cmd_collect(args) -> None:
    """Single-shot: download the result if the run has finished, otherwise just say so."""
    status = current_status(kernel_id())
    if not is_finished(status):
        print(f"The run is still '{status}': nothing to collect yet. Try again later.")
        return
    cmd_output(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    push = commands.add_parser("push", help="upload the notebook and start a run")
    push.add_argument("--mode", choices=["quick", "full"], default="quick")
    push.add_argument("--ref", default="main", help="git branch, tag or commit to train from")
    push.set_defaults(func=cmd_push)

    run = commands.add_parser("run", help="push, wait for the run to finish, download the result")
    run.add_argument("--mode", choices=["quick", "full"], default="quick")
    run.add_argument("--ref", default="main", help="git branch, tag or commit to train from")
    run.add_argument("--dir", default="kaggle-output")
    run.add_argument("--timeout", type=int, default=13 * 3600, help="seconds to wait in total")
    run.set_defaults(func=cmd_run)

    commands.add_parser("status", help="print the run status").set_defaults(func=cmd_status)

    wait = commands.add_parser("wait", help="block until the run finishes")
    wait.add_argument("--timeout", type=int, default=13 * 3600, help="seconds to wait")
    wait.set_defaults(func=cmd_wait)

    output = commands.add_parser("output", help="download the run's files and print metrics")
    output.add_argument("--dir", default="kaggle-output")
    output.set_defaults(func=cmd_output)

    collect = commands.add_parser("collect", help="like output, but only if the run has finished")
    collect.add_argument("--dir", default="kaggle-output")
    collect.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
