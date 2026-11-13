"""Which build is running: the app version and the model it serves.

``_version.py`` holds ``dev`` in git and is overwritten by the deploy workflow with the release
version, so a plain checkout reports ``dev``. The model comes from ``model_card.json`` next to
the model file.
"""

from __future__ import annotations

import functools
import json
import os

from _version import __commit__, __version__  # "dev" until the deploy workflow stamps it
from classifier import ARTIFACTS_DIR

CARD_PATH = os.path.join(ARTIFACTS_DIR, "model_card.json")


def app_info() -> dict:
    return {"version": __version__, "commit": __commit__}


@functools.cache
def model_info() -> dict | None:
    """Identity of the served model, or None when it has no card (e.g. a hand-placed model)."""
    try:
        with open(CARD_PATH, encoding="utf-8") as handle:
            card = json.load(handle)
    except (OSError, ValueError):
        return None
    return {
        "run_id": card.get("run_id"),
        "top1": card.get("metrics", {}).get("top1"),
        "sha256": card.get("model_sha256"),
    }
