"""Launch and collect the separate binary detector on Kaggle."""

import argparse
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import kaggle_run

SLUG = "doodleai-hidden"
STATE = kaggle_run.ROOT / ".kaggle-hidden-run.json"


def render_notebook(text: str, run_id: str) -> str:
    notebook = json.loads(text)
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for index, line in enumerate(cell["source"]):
                if line == 'RUN_ID = "local"\n':
                    cell["source"][index] = f"RUN_ID = {json.dumps(run_id)}\n"
                    count += 1
    if count != 1:
        raise ValueError("hidden.ipynb must contain exactly one RUN_ID setting")
    return json.dumps(notebook, indent=1) + "\n"


def push():
    username = kaggle_run.owner()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-hidden")
    text = kaggle_run.main_notebook_text("hidden.ipynb")
    metadata = kaggle_run.kernel_metadata(username)
    metadata["id"], metadata["title"] = f"{username}/{SLUG}", SLUG
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        (bundle / "train.ipynb").write_text(render_notebook(text, run_id), encoding="utf-8")
        (bundle / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        result = kaggle_run.kaggle(
            "kernels",
            "push",
            "-p",
            tmp,
            "--accelerator",
            kaggle_run.ACCELERATOR,
            "--timeout",
            str(kaggle_run.TIMEOUT_SECONDS["full"]),
        )
    version = kaggle_run.parse_version(result.stdout + result.stderr)
    kaggle_run.save_state({"id": metadata["id"], "version": version, "run_id": run_id}, STATE)
    print(result.stdout.strip())
    print(f"Started {run_id}, version {version}: https://www.kaggle.com/code/{metadata['id']}")


def target():
    state = kaggle_run.load_state(STATE)
    if not state or not state.get("version"):
        raise SystemExit("No recorded detector version; check the Kaggle notebook before retrying.")
    return state, f"{state['id']}/{state['version']}"


def collect(directory: Path):
    state, kernel = target()
    status = kaggle_run.current_status(kernel)
    if status != "complete":
        raise SystemExit(f"Detector run is {status}; output is only collected after completion.")
    destination = directory / kernel
    with tempfile.TemporaryDirectory() as tmp:
        kaggle_run.kaggle("kernels", "output", kernel, "-p", tmp, "-o")
        downloaded = Path(tmp)
        required = [
            "runs/hidden/best.keras",
            "runs/hidden/config.json",
            "runs/hidden/report/metrics.json",
            "hidden-data/X_val.npy",
            "hidden-data/y_val.npy",
            "hidden-data/X_test.npy",
            "hidden-data/y_test.npy",
            "hidden-data/meta.json",
        ]
        missing = [name for name in required if not (downloaded / name).is_file()]
        if missing:
            raise SystemExit(f"Detector output is missing files: {', '.join(missing)}")
        metrics = json.loads((downloaded / "runs/hidden/report/metrics.json").read_text())
        if metrics.get("run_id") != state["run_id"]:
            raise SystemExit("Downloaded detector run_id does not match the recorded run.")
        shutil.copytree(downloaded, destination, dirs_exist_ok=True)
    print(f"Downloaded detector output to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("push")
    commands.add_parser("status")
    commands.add_parser("wait")
    output = commands.add_parser("collect")
    output.add_argument("--dir", type=Path, default=Path("kaggle-output"))
    args = parser.parse_args()
    if args.command == "push":
        push()
    elif args.command == "status":
        print(kaggle_run.current_status(target()[1]))
    elif args.command == "wait":
        status = kaggle_run.wait_for(target()[1], 13 * 3600)
        if status != "complete":
            raise SystemExit(kaggle_run.finish_message(status, 13 * 3600))
    else:
        collect(args.dir)


if __name__ == "__main__":
    main()
