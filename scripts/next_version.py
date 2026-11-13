"""Next release version and release notes from Conventional Commits.

    python scripts/next_version.py version   # e.g. 1.4.0 (the last tag if HEAD is already tagged)
    python scripts/next_version.py notes     # Markdown for the GitHub Release

Every push to main is released: `feat:` bumps the minor version, anything marked breaking
(`feat!:` or a `BREAKING CHANGE` footer) the major, everything else the patch. Standard
library only; the deploy workflow runs it on a full clone (tags included).
"""

from __future__ import annotations

import re
import subprocess
import sys

FIRST_VERSION = (1, 0, 0)
TAG_GLOB = "v[0-9]*"
SUBJECT = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:\s*(?P<text>.+)$")
SECTIONS = (("feat", "Features"), ("fix", "Fixes"), ("perf", "Performance"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def last_tag() -> str | None:
    try:
        return git("describe", "--tags", "--abbrev=0", "--match", TAG_GLOB)
    except subprocess.CalledProcessError:
        return None


def parse_version(tag: str) -> tuple[int, int, int]:
    major, minor, patch = (int(part) for part in tag.removeprefix("v").split("."))
    return major, minor, patch


def commits_since(tag: str | None) -> list[tuple[str, str]]:
    """``(subject, body)`` of every commit after ``tag`` (all of them without a tag)."""
    span = f"{tag}..HEAD" if tag else "HEAD"
    raw = git("log", span, "--format=%s%x1f%b%x1e")
    pairs = []
    for entry in raw.split("\x1e"):
        if entry.strip():
            subject, _, body = entry.strip().partition("\x1f")
            pairs.append((subject.strip(), body.strip()))
    return pairs


def is_breaking(subject: str, body: str) -> bool:
    match = SUBJECT.match(subject)
    return bool(match and match["bang"]) or "BREAKING CHANGE" in body


def bump(current: tuple[int, int, int], commits: list[tuple[str, str]]) -> tuple[int, int, int]:
    major, minor, patch = current
    if any(is_breaking(subject, body) for subject, body in commits):
        return major + 1, 0, 0
    if any((m := SUBJECT.match(subject)) and m["type"] == "feat" for subject, _ in commits):
        return major, minor + 1, 0
    return major, minor, patch + 1


def next_version(tag: str | None, commits: list[tuple[str, str]]) -> tuple[int, int, int]:
    if tag is None:
        return FIRST_VERSION
    if not commits:
        return parse_version(tag)  # HEAD is already released (a re-run)
    return bump(parse_version(tag), commits)


def notes(commits: list[tuple[str, str]]) -> str:
    groups: dict[str, list[str]] = {title: [] for _, title in SECTIONS}
    groups["Other changes"] = []
    for subject, body in commits:
        match = SUBJECT.match(subject)
        title = "Other changes"
        if match:
            title = dict(SECTIONS).get(match["type"], "Other changes")
            scope = f"**{match['scope']}:** " if match["scope"] else ""
            line = f"- {scope}{match['text']}"
        else:
            line = f"- {subject}"
        if is_breaking(subject, body):
            line += " **(breaking)**"
        groups[title].append(line)
    parts = [f"## {title}\n" + "\n".join(lines) for title, lines in groups.items() if lines]
    return "\n\n".join(parts) or "No notable changes."


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"version", "notes"}:
        print(__doc__, file=sys.stderr)
        return 2
    tag = last_tag()
    commits = commits_since(tag)
    if argv[1] == "version":
        print(".".join(str(part) for part in next_version(tag, commits)))
    else:
        print(notes(commits))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
