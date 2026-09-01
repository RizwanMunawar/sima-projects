"""`sima-vision preview` and `sima-vision doctor`.

Between them these are what a new user can run before owning a DevKit, so they
have to work with no board, no model and no pyneat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sima_vision.cli import main, parse_size
from sima_vision.scene import boxes_from, build_scene, read_first_frame, render
from sima_vision.sinks import load_labels
from sima_vision.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]
TASK_NAMES = list(TASKS)


# ── the synthetic scene ──


def test_scene_has_subjects_inside_the_frame():
    frame, subjects = build_scene(640, 360)
    assert frame.shape == (360, 640, 3)
    assert frame.dtype == np.uint8
    assert len(subjects) >= 3
    for x1, y1, x2, y2, class_id, score in subjects:
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 360
        assert 0.0 < score <= 1.0
        assert class_id >= 0


def test_scene_is_textured_so_a_blur_is_visible():
    """A flat frame would make every blur setting look identical."""
    frame, _ = build_scene(640, 360)
    assert frame.std() > 20


@pytest.mark.parametrize("size", [(320, 180), (1920, 1080)])
def test_scene_scales(size):
    frame, subjects = build_scene(*size)
    assert frame.shape[:2] == (size[1], size[0])
    assert subjects


def test_boxes_from_subjects():
    _, subjects = build_scene(640, 360)
    boxes = boxes_from(subjects)
    assert len(boxes) == len(subjects)
    assert set(boxes[0]) == {"x1", "y1", "x2", "y2", "score", "class_id"}


# ── rendering, through each task's real render() ──


@pytest.mark.parametrize("name", TASK_NAMES)
def test_preview_renders_every_task(name):
    task = TASKS[name]()
    cfg = task.load(
        None, {"model.path": "<preview>", "source.uri": "<preview>"}, use_file=False
    )
    labels = load_labels(cfg.labels_path)
    frame, subjects = build_scene(640, 360)
    out = render(task, cfg, frame, subjects, labels)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame), "nothing was drawn"


def test_preview_leaves_the_source_frame_alone():
    task = TASKS["detect"]()
    cfg = task.load(
        None, {"model.path": "<p>", "source.uri": "<p>"}, use_file=False
    )
    frame, subjects = build_scene(640, 360)
    before = frame.copy()
    render(task, cfg, frame, subjects, load_labels(cfg.labels_path))
    np.testing.assert_array_equal(frame, before)


def test_segment_preview_blurs_the_background():
    """The whole point of previewing segment is seeing the blur."""
    task = TASKS["segment"]()
    cfg = task.load(
        None,
        {"model.path": "<p>", "source.uri": "<p>", "blur.enable": True,
         "blur.kernel": 41, "visualization.mask_alpha": 0.0, "visualization.mask_outline": False},
        use_file=False,
    )
    frame, subjects = build_scene(640, 360)
    out = render(task, cfg, frame, subjects, load_labels(cfg.labels_path))
    # A corner is background everywhere, so it must have been treated.
    assert not np.array_equal(out[:40, :40], frame[:40, :40])


def test_fall_preview_shows_more_than_one_state():
    task = TASKS["fall"]()
    cfg = task.load(
        None, {"model.path": "<p>", "source.uri": "<p>"}, use_file=False
    )
    frame, subjects = build_scene(640, 360)
    pipeline = task.make_pipeline(cfg, load_labels(cfg.labels_path))
    tracks = task.sample_results(cfg, pipeline, frame, boxes_from(subjects))
    assert len({t.state for t in tracks}) > 1, "preview should show several states"


def test_read_first_frame_returns_none_for_a_missing_file(tmp_path):
    assert read_first_frame(str(tmp_path / "nope.png")) is None


def test_read_first_frame_reads_an_image(tmp_path):
    import cv2

    path = tmp_path / "shot.png"
    cv2.imwrite(str(path), np.full((32, 48, 3), 120, np.uint8))
    frame = read_first_frame(str(path))
    assert frame is not None and frame.shape == (32, 48, 3)


# ── the commands ──


def test_parse_size():
    assert parse_size("1280x720") == (1280, 720)
    assert parse_size("640X480") == (640, 480)
    for bad in ("banana", "1280", "10x10", "x"):
        with pytest.raises(ValueError):
            parse_size(bad)


@pytest.mark.parametrize("name", TASK_NAMES)
def test_preview_command_writes_a_png(tmp_path, name):
    out = tmp_path / "preview.png"
    code = main(["preview", "--task", name, "--no-config", "-o", str(out),
                 "--size", "320x180"])
    assert code == 0
    assert out.is_file() and out.stat().st_size > 0


def test_preview_command_uses_a_shipped_config(tmp_path):
    from sima_vision.config import packaged_config

    out = tmp_path / "seg.png"
    code = main(["preview", "--task", "segment", "-o", str(out),
                 "-c", str(packaged_config("segment")), "--size", "320x180"])
    assert code == 0
    assert out.is_file()


def test_preview_creates_the_output_directory(tmp_path):
    out = tmp_path / "nested" / "dir" / "p.png"
    assert main(["preview", "--no-config", "-o", str(out), "--size", "320x180"]) == 0
    assert out.is_file()


def test_preview_falls_back_when_the_source_cannot_be_read(tmp_path, capsys):
    """Raw .h264 is exactly what OpenCV cannot open, and this repo ships those."""
    fake = tmp_path / "clip.h264"
    fake.write_bytes(b"\x00\x00\x00\x01\x67" + b"\x00" * 64)
    out = tmp_path / "p.png"
    code = main(["preview", "--no-config", "--source", str(fake), "-o", str(out),
                 "--size", "320x180"])
    assert code == 0
    assert out.is_file()
    assert "synthetic scene" in capsys.readouterr().err


def test_preview_says_no_model_was_run(capsys, tmp_path):
    main(["preview", "--no-config", "-o", str(tmp_path / "p.png"), "--size", "320x180"])
    assert "NO MODEL WAS RUN" in capsys.readouterr().out


def test_doctor_exits_zero_and_names_what_is_missing(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "sima-vision" in out
    for module in ("yaml", "numpy", "cv2", "pyneat"):
        assert module in out
    assert "What you can do right now" in out
