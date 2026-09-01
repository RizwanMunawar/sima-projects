"""The command surface: parsing, dispatch and the compatibility shims."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from sima_vision import __version__
from sima_vision.cli import build_parser, collect_overrides, main
from sima_vision.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]


def parse(argv):
    return build_parser().parse_args(argv)


def test_every_task_has_a_subcommand():
    parser = build_parser()
    for name in TASKS:
        args = parser.parse_args([name, "--no-config", "--validate"])
        assert args.task == name


def test_no_command_prints_help_and_fails():
    assert main([]) == 2


def test_version_is_the_package_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_shared_flags_map_to_config_paths():
    args = parse(["detect", "--source", "c.h264", "--conf", "0.4", "--max-det", "7"])
    overrides = collect_overrides(args)
    assert overrides["source.uri"] == "c.h264"
    assert overrides["decode.score_threshold"] == 0.4
    assert overrides["decode.max_detections"] == 7


def test_unset_flags_are_not_overrides():
    """Only what the user typed may override the file."""
    args = parse(["detect", "--source", "c.h264"])
    assert "decode.score_threshold" not in collect_overrides(args)


def test_negative_switches_override_to_false():
    args = parse(["detect", "--no-save", "--no-video"])
    overrides = collect_overrides(args)
    assert overrides["output.save.enable"] is False
    assert overrides["output.video.enable"] is False


def test_segment_flags():
    args = parse(["segment", "--anonymise", "--keep-classes", "person", "car",
                  "--blur-method", "pixelate", "--mask-threshold", "0.3"])
    overrides = collect_overrides(args)
    assert overrides["blur.invert"] is True
    assert overrides["blur.keep_classes"] == ["person", "car"]
    assert overrides["blur.method"] == "pixelate"
    assert overrides["segmentation.threshold"] == 0.3


def test_fall_flags_reach_the_nested_smtp_section():
    args = parse(["fall", "--alert-to", "a@x.com", "--smtp-port", "465", "--send"])
    overrides = collect_overrides(args)
    assert overrides["alerts.to"] == ["a@x.com"]
    assert overrides["alerts.smtp.port"] == 465
    assert overrides["alerts.dry_run"] is False


def test_alert_recipient_turns_alerts_on():
    cfg = TASKS["fall"]().load(
        None,
        {"model.path": "m.tar.gz", "source.uri": "c.h264", "alerts.to": ["a@x.com"]},
        use_file=False,
    )
    assert cfg.alerts.enable is True
    # ...but still a dry run, so nobody is emailed by accident.
    assert cfg.alerts.dry_run is True


def test_send_is_needed_to_actually_send():
    cfg = TASKS["fall"]().load(
        None,
        {
            "model.path": "m.tar.gz", "source.uri": "c.h264",
            "alerts.to": ["a@x.com"], "alerts.from": "b@x.com",
            "alerts.dry_run": False,
        },
        use_file=False,
    )
    assert cfg.alerts.enable is True
    assert cfg.alerts.dry_run is False


def test_config_and_no_config_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse(["detect", "--config", "a.yaml", "--no-config"])


def test_minimal_strips_the_sinks():
    task = TASKS["segment"]()
    cfg = task.load(
        None, {"model.path": "m.tar.gz", "source.uri": "c.h264"}, use_file=False
    )
    args = parse(["segment", "--minimal"])
    stripped = task.post_process(cfg, args)
    assert stripped.segment.masks == "off"
    assert stripped.blur.enable is False
    assert not (stripped.save_enable or stripped.video_enable or stripped.insight_enable)


def test_validate_exits_zero_without_a_board():
    code = main([
        "detect", "--no-config", "--model", "m.tar.gz", "--source", "c.h264", "--validate",
    ])
    assert code == 0


def test_a_bad_config_exits_one(capsys):
    code = main(["detect", "--no-config", "--validate"])   # no model.path
    assert code == 1
    assert "model.path must be set" in capsys.readouterr().err


# ── compatibility shims ──

SHIMS = [
    ("object-detection", "detect"),
    ("instance-segmentation", "segment"),
    ("fall-detection", "fall"),
]


@pytest.mark.parametrize("directory,task", SHIMS, ids=[d for d, _ in SHIMS])
def test_shim_targets_the_right_task(directory, task):
    path = REPO / directory / "src" / "app.py"
    spec = importlib.util.spec_from_file_location(f"shim_{task}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TASK == task


@pytest.mark.parametrize("directory,task", SHIMS, ids=[d for d, _ in SHIMS])
def test_shim_accepts_the_old_flag(directory, task):
    """The READMEs document `--validate-config`, so it has to keep working."""
    result = subprocess.run(
        [sys.executable, "src/app.py", "--config", "config.yaml", "--validate-config"],
        cwd=REPO / directory, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "config OK" in result.stdout
