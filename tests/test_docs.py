"""The README's commands have to be real ones.

A README that drifts from the CLI is worse than no README: every example in it
reads as a promise. These tests take the promises literally -- every
`sima-vision ...` line in the docs is fed to the actual parser, and every flag
mentioned in a table has to exist.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from sima_vision.api import _alias_table
from sima_vision.cli import build_parser
from sima_vision.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
DOCS = sorted((REPO / "docs").glob("*.md"))

#: Fragments that stand for something rather than being runnable as written.
PLACEHOLDERS = ("<", ">", "...", "$EDITOR", "EMAIL", "PATH", "NAME", "DIR", "|")


def command_lines(text: str) -> list[str]:
    """Every `sima-vision ...` invocation in fenced bash blocks, joined and cleaned."""
    found = []
    for body in re.findall(r"```bash\n(.*?)```", text, re.S):
        # Re-join shell line continuations before splitting.
        body = body.replace("\\\n", " ")
        for raw in body.splitlines():
            line = raw.split("#", 1)[0].strip()
            if line.startswith("sima-vision "):
                found.append(line)
    return found


def test_the_readme_actually_has_examples():
    assert len(command_lines(README.read_text(encoding="utf-8"))) > 10


@pytest.mark.parametrize(
    "doc", [README, *DOCS], ids=lambda p: p.name
)
def test_every_documented_command_parses(doc):
    parser = build_parser()
    checked = 0
    for line in command_lines(doc.read_text(encoding="utf-8")):
        argv = shlex.split(line)[1:]
        if any(token.startswith(p) or p in token for token in argv for p in PLACEHOLDERS):
            continue                      # a stand-in, not a real invocation
        if not argv:
            continue
        try:
            parser.parse_args(argv)
        except SystemExit as exc:          # argparse exits on a bad flag
            raise AssertionError(f"{doc.name}: `{line}` is not a valid command") from exc
        checked += 1
    assert checked or doc is not README, f"{doc.name}: nothing was checked"


def flags_in(text: str) -> set[str]:
    """Every `--flag` in a code span or a fenced block.

    Only those: a badge URL like `sima--vision-3775A9` contains something that
    looks like a flag but is not one.
    """
    spans = re.findall(r"`([^`\n]+)`", text)
    blocks = re.findall(r"```[a-z]*\n(.*?)```", text, re.S)
    return {
        flag
        for chunk in spans + blocks
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", chunk)
    }


def known_flags() -> set[str]:
    """Every option string the CLI accepts, from every subcommand."""
    flags = {"--help", "--version"}

    def collect(parser):
        for action in parser._actions:
            flags.update(action.option_strings)
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):        # a subparsers action
                for sub in choices.values():
                    collect(sub)

    collect(build_parser())
    return flags


def test_the_readme_mentions_no_flag_that_does_not_exist():
    """A table row promising `--video` when the flag is `--video-path` is a bug."""
    documented = flags_in(README.read_text(encoding="utf-8"))
    unknown = documented - known_flags()
    assert not unknown, f"README documents flags that do not exist: {sorted(unknown)}"


def test_the_readme_python_keywords_exist():
    """Every `keyword=` shown in a Python block must be a real setting."""
    text = README.read_text(encoding="utf-8")
    aliases: set[str] = set()
    for task_cls in TASKS.values():
        table, _ = _alias_table(task_cls())
        aliases |= set(table)
    # Arguments of the API functions themselves, which are not config settings.
    aliases |= {"out", "size", "config", "use_config_file", "task"}

    used = set()
    for body in re.findall(r"```python\n(.*?)```", text, re.S):
        used |= set(re.findall(r"[( ,]([a-z_]+)=", body))
    unknown = used - aliases
    assert not unknown, f"README uses Python keywords that do not exist: {sorted(unknown)}"


def test_every_task_and_command_is_documented():
    text = README.read_text(encoding="utf-8")
    for name in [*TASKS, "preview", "init", "fetch", "doctor"]:
        assert f"sima-vision {name}" in text, f"{name} is not in the README"


@pytest.mark.parametrize("doc", [README, *DOCS], ids=lambda p: p.name)
def test_internal_links_resolve(doc):
    text = doc.read_text(encoding="utf-8")
    broken = []
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
        if target.startswith(("http", "mailto:")):
            continue
        if not (doc.parent / target).resolve().exists():
            broken.append(target)
    assert not broken, f"{doc.name}: broken links {broken}"


@pytest.mark.parametrize("doc", [README, *DOCS], ids=lambda p: p.name)
def test_no_references_to_the_old_layout(doc):
    """The per-app folders and `src/app.py` are gone; nothing may still point at them."""
    text = doc.read_text(encoding="utf-8")
    stale = [
        token for token in
        ("object-detection/", "instance-segmentation/", "fall-detection/",
         "src/app.py", "--validate-config", "requirements.txt")
        # `apps-src/examples/object-detection/...` is a path inside SiMa's SDK.
        if token in text.replace("apps-src/examples/object-detection", "")
    ]
    assert not stale, f"{doc.name}: stale references {stale}"
