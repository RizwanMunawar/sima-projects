"""The Python API.

Its whole promise is that a CLI flag and a Python keyword are the same setting
under the same name, so most of these tests are about that not drifting.
"""

from __future__ import annotations

import pytest

import sima_vision
from sima_vision.api import _alias_table, settings_to_overrides
from sima_vision.cli import build_parser
from sima_vision.tasks import TASKS


def test_the_package_exports_the_three_verbs():
    for name in ("run", "preview", "validate", "load"):
        assert callable(getattr(sima_vision, name))


def test_the_preview_function_does_not_hide_the_scene_module():
    """`from sima_vision import preview` is the function; the module still imports."""
    from sima_vision.scene import build_scene

    assert callable(sima_vision.preview)
    assert callable(build_scene)


# ── the alias table ──


@pytest.mark.parametrize("name", list(TASKS))
def test_every_cli_flag_has_a_python_keyword(name):
    """If a flag exists, Python can set it. That is the whole contract."""
    task = TASKS[name]()
    aliases, _ = _alias_table(task)
    assert build_parser().parse_args([name, "--no-config", "--validate"]).command == name

    # Collect every dotted dest the subcommand can write...
    import argparse

    probe = argparse.ArgumentParser(add_help=False)
    from sima_vision.cli import add_shared_arguments

    add_shared_arguments(probe)
    task.add_arguments(probe.add_argument_group("task"))
    dests = {a.dest for a in probe._actions if "." in a.dest}
    # ...and check each is reachable from Python.
    assert dests <= set(aliases.values()), dests - set(aliases.values())


def test_negative_flags_become_positive_keywords():
    """Nobody should have to write no_save=True."""
    aliases, _ = _alias_table(TASKS["detect"]())
    assert aliases["save"] == "output.save.enable"
    assert aliases["video"] == "output.video.enable"
    assert "no_save" not in aliases


def test_a_flag_that_clears_something_is_inverted():
    """`--send` reads as "do send" but clears alerts.dry_run."""
    aliases, inverted = _alias_table(TASKS["fall"]())
    assert aliases["send"] == "alerts.dry_run"
    assert "send" in inverted
    # ...while a plain negative flag is not inverted twice.
    assert "save" not in inverted


def test_keywords_map_to_config_paths():
    task = TASKS["detect"]()
    assert settings_to_overrides(task, {"conf": 0.4}) == {"decode.score_threshold": 0.4}
    assert settings_to_overrides(task, {"source": "c.h264"}) == {"source.uri": "c.h264"}
    assert settings_to_overrides(task, {"max_det": 7}) == {"decode.max_detections": 7}


def test_dotted_keys_pass_straight_through():
    """Anything the aliases miss is still reachable."""
    task = TASKS["detect"]()
    out = settings_to_overrides(task, {"runtime.output_buffers": 2})
    assert out == {"runtime.output_buffers": 2}


def test_an_unknown_keyword_suggests_the_near_miss():
    with pytest.raises(TypeError, match="conf"):
        settings_to_overrides(TASKS["detect"](), {"confidence": 0.5})


# ── the verbs ──


def test_validate_returns_the_resolved_config():
    cfg = sima_vision.validate(
        "detect", use_config_file=False, model="m.tar.gz", source="c.h264",
        conf=0.55, max_det=12, save=False,
    )
    assert cfg.score_threshold == 0.55
    assert cfg.max_detections == 12
    assert cfg.save_enable is False


def test_validate_raises_on_a_bad_setting():
    with pytest.raises(ValueError, match="score_threshold"):
        sima_vision.validate(
            "detect", use_config_file=False, model="m", source="c", conf=5.0
        )


def test_send_turns_alerts_on_and_dry_run_off():
    cfg = sima_vision.validate(
        "fall", use_config_file=False, model="m", source="c",
        alert_to=["a@x.com"], alert_from="b@x.com", send=True,
    )
    assert cfg.alerts.enable is True
    assert cfg.alerts.dry_run is False


def test_anonymise_keyword():
    cfg = sima_vision.validate(
        "segment", use_config_file=False, model="m", source="c",
        anonymise=True, keep_classes=["person"],
    )
    assert cfg.blur.invert is True
    assert cfg.blur.keep_classes == ("person",)


def test_unknown_task():
    with pytest.raises(ValueError, match="unknown task"):
        sima_vision.validate("nope")


@pytest.mark.parametrize("name", list(TASKS))
def test_preview_writes_a_png(tmp_path, name):
    out = sima_vision.preview(name, out=tmp_path / "p.png", size=(320, 180))
    assert out.is_file() and out.stat().st_size > 0


def test_preview_settings_reach_the_drawing(tmp_path):
    """Two different blur settings must not produce the same image."""
    gaussian = sima_vision.preview(
        "segment", out=tmp_path / "g.png", size=(320, 180), blur_method="gaussian"
    ).read_bytes()
    pixelated = sima_vision.preview(
        "segment", out=tmp_path / "p.png", size=(320, 180), blur_method="pixelate"
    ).read_bytes()
    assert gaussian != pixelated


def test_preview_needs_no_model_or_source(tmp_path):
    """The placeholders are filled in, so this must not raise about model.path."""
    assert sima_vision.preview("detect", out=tmp_path / "p.png", size=(320, 180)).is_file()
