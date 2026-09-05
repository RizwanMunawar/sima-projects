"""Turning a trained .pt into something the board can run.

The export half is checked here without torch: what matters and what has been
wrong before is the *contract* -- which outputs, named what, shaped how -- and
that is fixed by the board's box decoder, not by anything ultralytics chooses.
Those names and shapes were read out of a working pack's own mpk.json, so they
are pinned here rather than left to be rediscovered.

The compile half needs the SiMa Model SDK, which is the Palette container on
x86. What is testable is that this refuses clearly instead of failing at the
last step, and that the recipe it hands over is the one the pack shipped.
"""

from __future__ import annotations

import io
import tarfile

import pytest

from sima_vision import export


def test_the_output_contract_is_the_one_the_board_decodes():
    """`Configured for subtensors: 6` in a run's log is this list.

    Boxes first, then classes, which is the order a working pack's final
    PassThrough lists them in. Get the order wrong and every detection is
    noise rather than an error.
    """
    assert export.RAW_OUTPUTS == (
        "bbox_0", "bbox_1", "bbox_2",
        "class_logit_0", "class_logit_1", "class_logit_2",
    )


def test_the_expected_shapes_match_the_published_pack():
    """Checked against the byte sizes in yolo26n's own mpk.json.

    bbox_0 is 102400 bytes, which is 25600 float32 -- 80x80x4. class_logit_0
    is 2048000, which is 512000 floats -- 80x80x80. Four box channels rather
    than 64 is YOLO26 having reg_max=1: no DFL bins to unpack.
    """
    shapes = export.expected_shapes(640, 80)
    assert shapes["bbox_0"] == (1, 4, 80, 80)
    assert shapes["bbox_1"] == (1, 4, 40, 40)
    assert shapes["bbox_2"] == (1, 4, 20, 20)
    assert shapes["class_logit_0"] == (1, 80, 80, 80)
    assert shapes["class_logit_2"] == (1, 80, 20, 20)

    for name, size in (("bbox_0", 102400), ("class_logit_0", 2048000)):
        floats = 1
        for dim in shapes[name]:
            floats *= dim
        assert floats * 4 == size, f"{name} disagrees with the published pack"


def test_the_shapes_follow_the_input_size():
    shapes = export.expected_shapes(512, 3)
    assert shapes["bbox_0"] == (1, 4, 64, 64)
    assert shapes["class_logit_2"] == (1, 3, 16, 16)


# ── refusing heads that cannot produce it ──


class FakeHead:
    def __init__(self, nl=3, nc=80, reg_max=1):
        self.nl, self.nc, self.reg_max = nl, nc, reg_max
        self.cv2 = [None] * nl
        self.cv3 = [None] * nl


class FakeNet:
    def __init__(self, head):
        self.model = [None, head]


def test_a_yolov8_style_head_is_named_rather_than_exported():
    """reg_max 16 means 64 DFL channels where the board reads 4 coordinates.

    It would export, and the shapes would be wrong in a way that only shows up
    as garbage boxes on the board.
    """
    with pytest.raises(RuntimeError, match="reg_max=16"):
        export.check_head(FakeNet(FakeHead(reg_max=16)))


def test_a_head_with_the_wrong_number_of_levels_is_refused():
    with pytest.raises(RuntimeError, match="2 levels"):
        export.check_head(FakeNet(FakeHead(nl=2)))


def test_something_that_is_not_a_detection_model_is_refused():
    class Segment:
        pass

    with pytest.raises(RuntimeError, match="not a YOLO detection model"):
        export.check_head(FakeNet(Segment()))


def test_a_good_head_reports_its_levels_and_classes():
    assert export.check_head(FakeNet(FakeHead(nc=7))) == (3, 7)


# ── the recipe travels inside the pack ──


def pack_with(names: dict[str, bytes], path):
    with tarfile.open(path, "w:gz") as tar:
        for name, body in names.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return path


def test_the_compile_recipe_is_read_out_of_a_published_pack(tmp_path):
    """Not paraphrased here.

    The settings that matter -- bfloat16, MSE calibration, the MLA tessellation
    layouts -- are the ones SiMa shipped, and a copy written from memory would
    drift from them silently.
    """
    body = b"#!/usr/bin/env python3\n# compile it\n"
    pack = pack_with(
        {
            "archived_compile_script.compile_yolo26_modelsdk.py": body,
            "model_stage1_mla.elf": b"elf",
        },
        tmp_path / "pack.tar.gz",
    )
    assert export.compile_recipe(pack) == body.decode()


def test_a_pack_without_a_recipe_says_so(tmp_path):
    pack = pack_with({"model_stage1_mla.elf": b"elf"}, tmp_path / "bare.tar.gz")
    with pytest.raises(RuntimeError, match="no archived_compile_script"):
        export.compile_recipe(pack)


def test_the_next_steps_name_the_recipe_that_was_written(tmp_path):
    """A message that stops at "use the Model SDK" is not worth printing."""
    onnx = tmp_path / "best-raw.onnx"
    recipe = tmp_path / "compile_modelsdk.py"
    text = export.next_steps(onnx, recipe)

    assert "Palette" in text
    assert "compile_modelsdk.py --model best-raw.onnx" in text
    assert "sima-vision push" in text and "--model" in text


def test_the_next_steps_still_help_with_no_recipe_to_hand(tmp_path):
    text = export.next_steps(tmp_path / "best-raw.onnx", None)
    assert "Compile it with the Model SDK" in text
    assert "--model best-raw.onnx" not in text
