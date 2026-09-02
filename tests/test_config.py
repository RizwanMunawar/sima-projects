"""Config loading, CLI overrides and discovery.

None of this needs pyneat or a board, which is the point: a bad config should
fail on a laptop in under a second.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sima_vision.config import (
    BaseConfig,
    DrawConfig,
    TaskDefaults,
    apply_overrides,
    discover_config,
    load_base_config,
    packaged_config,
    validate_base,
)
from sima_vision.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]

#: The starter configs shipped inside the wheel, and the task that owns each.
#: Loading every one of them is the regression guard for the shared-core
#: refactor, and proves `sima-vision init` cannot hand out a broken file.
SHIPPED = [(name, packaged_config(name)) for name in TASKS]


@pytest.mark.parametrize("name,path", SHIPPED, ids=[n for n, _ in SHIPPED])
def test_shipped_config_loads_and_validates(name, path):
    """Every config.yaml in the repo must load and pass its task's validation."""
    assert path.is_file(), f"{path} is missing"
    cfg = TASKS[name]().load(path, {})
    assert cfg.config_path == path
    assert cfg.model_path, "model.path should be set by the shipped config"


@pytest.mark.parametrize("name,path", SHIPPED, ids=[n for n, _ in SHIPPED])
def test_shipped_config_describes(name, path):
    """--validate must be able to describe every shipped config."""
    task = TASKS[name]()
    cfg = task.load(path, {})
    for line in task.describe(cfg):
        assert isinstance(line, str) and line


def test_defaults_alone_are_a_complete_config():
    """A TaskDefaults with no catalogue entry is the one case still missing a model.

    The three real tasks name themselves, so `model.path` defaults into
    `assets/`; see test_assets. A bare TaskDefaults stands for a task that has
    no default archive, and that one still has to be told.
    """
    cfg = load_base_config({}, None, TaskDefaults())
    assert isinstance(cfg, BaseConfig)
    with pytest.raises(ValueError, match="model.path must be set"):
        validate_base(cfg)


def test_flags_alone_are_enough_to_run():
    cfg = TASKS["detect"]().load(
        None, {"model.path": "m.tar.gz", "source.uri": "c.h264"}, use_file=False
    )
    assert cfg.model_path == "m.tar.gz"
    assert cfg.source_uri == "c.h264"
    assert cfg.config_path is None


# ── overrides ──


def test_apply_overrides_creates_missing_sections():
    raw = {}
    apply_overrides(raw, {"source.uri": "clip.h264", "decode.score_threshold": 0.7})
    assert raw == {"source": {"uri": "clip.h264"}, "decode": {"score_threshold": 0.7}}


def test_apply_overrides_reaches_nested_sections():
    raw = {}
    apply_overrides(raw, {"alerts.smtp.port": 465})
    assert raw["alerts"]["smtp"]["port"] == 465


def test_apply_overrides_ignores_none():
    """An unset flag must defer to the config file rather than blank it."""
    raw = {"source": {"uri": "from-file.h264"}}
    apply_overrides(raw, {"source.uri": None})
    assert raw["source"]["uri"] == "from-file.h264"


def test_apply_overrides_replaces_a_non_mapping():
    raw = {"source": "not-a-mapping"}
    apply_overrides(raw, {"source.uri": "clip.h264"})
    assert raw["source"] == {"uri": "clip.h264"}


def test_flags_beat_the_config_file():
    cfg = TASKS["detect"]().load(
        packaged_config("detect"),
        {"decode.score_threshold": 0.9, "output.save.enable": False},
    )
    assert cfg.score_threshold == 0.9
    assert cfg.save_enable is False


def test_override_goes_through_validation():
    """A CLI flag cannot reach a state a config file could not."""
    with pytest.raises(ValueError, match=r"score_threshold"):
        TASKS["detect"]().load(
            None,
            {"model.path": "m.tar.gz", "source.uri": "c.h264", "decode.score_threshold": 5.0},
            use_file=False,
        )


# ── discovery ──


def test_discovery_finds_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("model:\n  path: a.tar.gz\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert discover_config(None) == tmp_path / "config.yaml"


def test_discovery_returns_none_when_there_is_nothing(tmp_path, monkeypatch):
    """No file is not an error: the defaults are a complete configuration."""
    monkeypatch.chdir(tmp_path)
    assert discover_config(None) is None


def test_explicit_config_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_config(tmp_path / "nope.yaml")


def test_init_writes_a_config_every_task_can_load(tmp_path, monkeypatch):
    """`sima-vision init` must never hand out a file that then fails to load."""
    from sima_vision.setup_commands import run_init

    monkeypatch.chdir(tmp_path)
    for name in TASKS:
        out = tmp_path / f"{name}.yaml"
        assert run_init(name, out, force=False) == 0
        assert TASKS[name]().load(out, {}).config_path == out


# ── scalar readers ──


def test_yaml_bare_on_off_folds_onto_tokens(tmp_path):
    """YAML 1.1 turns `enable: on` into True; the loader must fold it back."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  path: m.tar.gz\nsource:\n  uri: c.h264\n"
        "preprocess:\n  enable: on\n  resize:\n    enable: off\n",
        encoding="utf-8",
    )
    cfg = TASKS["detect"]().load(path, {})
    assert cfg.preprocess.enable == "on"
    assert cfg.preprocess.resize_enable == "off"


def test_bad_colour_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  path: m.tar.gz\nsource:\n  uri: c.h264\n"
        "visualization:\n  text_color: [0, 0, 300]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="between 0 and 255"):
        TASKS["detect"]().load(path, {})


def test_draw_defaults_differ_per_task():
    """The DrawConfig is shared, but its defaults are not."""
    detect = TASKS["detect"]().defaults.draw
    segment = TASKS["segment"]().defaults.draw
    assert detect.box_thickness == 3
    assert segment.box_thickness == 2
    assert detect.centre_dot is True
    assert segment.centre_dot is False
    assert isinstance(detect, DrawConfig)
