"""The command surface: parsing, dispatch and the compatibility shims."""

from __future__ import annotations

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
        assert args.command == name


def test_the_subcommand_name_survives_a_task_flag():
    """`preview --task` and the init/fetch positional are both called `task`.

    They used to share argparse's dest with the subcommand itself, so which one
    won came down to parse order.
    """
    parser = build_parser()
    assert parser.parse_args(["preview", "--task", "segment"]).command == "preview"
    assert parser.parse_args(["preview", "--task", "segment"]).task == "segment"
    assert parser.parse_args(["preview"]).task == "detect"
    assert parser.parse_args(["init", "fall"]).command == "init"
    assert parser.parse_args(["init", "fall"]).task == "fall"
    assert parser.parse_args(["fetch"]).task == "detect"
    assert parser.parse_args(["doctor"]).command == "doctor"


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
    code = main(["detect", "--no-config", "--conf", "5", "--validate"])
    assert code == 1
    assert "decode.score_threshold" in capsys.readouterr().err


def test_no_flags_at_all_still_validates(capsys):
    """Neither --model nor --source is required: both default into assets/."""
    assert main(["detect", "--no-config", "--validate"]) == 0
    out = capsys.readouterr().out
    assert "assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz" in out
    assert "assets/videos/people-walking-outside-mall.h264" in out


# ── init and fetch ──


@pytest.mark.parametrize("name", list(TASKS))
def test_init_writes_a_documented_config(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    assert main(["init", name]) == 0
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert written.count("#") > 50, "the starter config should be commented"
    # And the command that reads it back must find it with no --config at all.
    assert main([name, "--validate"]) == 0


def test_init_refuses_to_clobber(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "detect"]) == 0
    assert main(["init", "detect"]) == 1
    assert "already exists" in capsys.readouterr().err


def test_init_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("stale", encoding="utf-8")
    assert main(["init", "segment", "--force"]) == 0
    assert "stale" not in (tmp_path / "config.yaml").read_text(encoding="utf-8")


def test_init_can_write_elsewhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "cfgs" / "fall.yaml"
    assert main(["init", "fall", "-o", str(out)]) == 0
    assert out.is_file()


def test_fetch_prints_a_runnable_model_command():
    """The model is behind a login, so the command has to be exact."""
    from sima_vision.assets import CATALOGUE, model_command

    for name in TASKS:
        assert name in CATALOGUE
        command = model_command(name)
        assert "sima-cli download" in command
        assert CATALOGUE[name].model_file in command
        # It must land where a run then looks for it.
        assert "assets/models" in command


def test_every_command_is_reachable():
    parser = build_parser()
    for name in [*TASKS, "init", "fetch", "preview", "doctor"]:
        assert name in parser.format_help()


def test_fetch_reports_a_download_failure(tmp_path, monkeypatch, capsys):
    """A half-written file must not be left behind looking like a good one."""
    import urllib.error

    from sima_vision import assets, setup_commands

    def boom(url, timeout=0):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(assets.urllib.request, "urlopen", boom)
    monkeypatch.chdir(tmp_path)
    assert setup_commands.run_fetch("detect", tmp_path / "assets") == 1
    assert "FAIL" in capsys.readouterr().err
    assert not list((tmp_path / "assets").rglob("*.part"))
    assert not list((tmp_path / "assets").rglob("*.h264"))


def test_fetch_does_not_redownload(tmp_path, monkeypatch, capsys):
    from sima_vision import assets, setup_commands

    videos = tmp_path / "assets" / "videos"
    videos.mkdir(parents=True)
    for name in assets.SAMPLE_VIDEOS:
        (videos / name).write_bytes(b"already here")

    def boom(url, timeout=0):
        raise AssertionError("should not have been fetched")

    monkeypatch.setattr(assets.urllib.request, "urlopen", boom)
    monkeypatch.chdir(tmp_path)
    assert setup_commands.run_fetch("segment", tmp_path / "assets") == 0
    assert "have" in capsys.readouterr().out
