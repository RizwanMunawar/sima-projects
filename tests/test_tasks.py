"""The task registry, and the fourth app that does not live in this package.

`sima-vision detect` and a task someone else pip-installs have to arrive by the
same road, or the plugin path is a second-class one nobody tests. So the
built-ins are checked against the registry, and a fake distribution is pushed
through `load_plugins` to prove a stranger's class really does become a
subcommand with the shared flags attached.
"""

from __future__ import annotations

import pytest

from sima_vision.cli import build_parser
from sima_vision.tasks import BUILTIN, TASKS, Task, build_registry, load_plugins


class Entry:
    """The two things `load_plugins` asks of an entry point."""

    def __init__(self, name, target):
        self.name = name
        self._target = target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def with_entries(monkeypatch, *entries):
    """Make the whole world's installed distributions be exactly these."""
    monkeypatch.setattr(
        "importlib.metadata.entry_points", lambda group=None: list(entries)
    )


class CountTask(Task):
    name = "count"
    help = "Count things crossing a line"


def test_every_built_in_is_registered():
    assert [task.name for task in BUILTIN] == list(TASKS)[: len(BUILTIN)]


def test_the_built_ins_come_first(monkeypatch):
    """--help order is the reading order: the three that ship, then the rest."""
    with_entries(monkeypatch, Entry("count", CountTask))
    assert list(build_registry())[: len(BUILTIN)] == [task.name for task in BUILTIN]


def test_a_plugin_becomes_a_task(monkeypatch):
    with_entries(monkeypatch, Entry("count", CountTask))
    assert build_registry()["count"] is CountTask


def test_a_plugin_cannot_replace_a_built_in(monkeypatch, capsys):
    """Otherwise installing something could silently change what `detect` means."""

    class Impostor(Task):
        name = "detect"

    with_entries(monkeypatch, Entry("detect", Impostor))
    assert build_registry()["detect"] is not Impostor
    assert "is taken" in capsys.readouterr().out


def test_a_broken_plugin_does_not_take_the_cli_down(monkeypatch, capsys):
    """One bad third-party package must not stop `sima-vision detect` running."""
    with_entries(monkeypatch, Entry("bad", ImportError("no module named nonsense")))
    assert build_registry() == {task.name: task for task in BUILTIN}
    assert "failed to import" in capsys.readouterr().out


def test_something_that_is_not_a_task_is_refused(monkeypatch, capsys):
    with_entries(monkeypatch, Entry("odd", "a string, not a class"))
    assert "odd" not in build_registry()
    assert "not a Task subclass" in capsys.readouterr().out


def test_unreadable_metadata_is_survivable(monkeypatch, capsys):
    def explode(group=None):
        raise OSError("the metadata directory is a mess")

    monkeypatch.setattr("importlib.metadata.entry_points", explode)
    assert load_plugins(set()) == {}
    assert "entry points" in capsys.readouterr().out


def test_a_plugin_gets_the_shared_flags(monkeypatch):
    """The whole point: a stranger's task inherits everything, having done nothing."""
    with_entries(monkeypatch, Entry("count", CountTask))
    monkeypatch.setattr("sima_vision.cli.TASKS", build_registry())

    args = build_parser().parse_args(
        ["count", "--source", "clip.h264", "--conf", "0.4", "--validate", "--quiet"]
    )
    assert args.command == "count"
    assert getattr(args, "source.uri") == "clip.h264"
    assert getattr(args, "decode.score_threshold") == 0.4
    assert args.validate and args.quiet


def test_a_task_without_a_runtime_says_so():
    """The one method a plugin must write. Forgetting it should be loud."""
    with pytest.raises(NotImplementedError):
        CountTask().runtime(None, None)
