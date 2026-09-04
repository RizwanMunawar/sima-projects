"""H.264 Annex-B inspection.

These are the functions that make "the clip ended" distinguishable from "the
source stalled", so they are worth pinning down. All of them read bytes and
need no board.
"""

from __future__ import annotations

import pytest

from sima_vision.media import (
    BitReader,
    count_h264_pictures,
    count_pictures_in,
    fps_from_rate,
    is_elementary_h264,
    parse_sps,
    probe_h264_sps,
    unescape_rbsp,
)
from sima_vision.tasks import TASKS

# Real SPS NAL payloads, lifted byte for byte out of the two DevKit sample
# clips. Both are 1920x1080 High profile with VUI timing, and the 24 fps one
# carries an emulation-prevention byte (`00 00 03 00 20`), so parsing it end to
# end exercises unescape_rbsp against something a decoder actually accepts
# rather than against a fixture built by the same code that reads it.
SPS_1080P_24 = bytes.fromhex(
    "640028acd940780227e59a808080a0000003002000000601e30632c000"
)
SPS_1080P_30 = bytes.fromhex(
    "640028acd940780227e59a808080a000007d20001d4c01e30632c000"
)
REAL_SPS = [(SPS_1080P_24, 24), (SPS_1080P_30, 30)]


def test_bitreader_unsigned():
    r = BitReader(b"\xa0")           # 1010 0000
    assert r.u(1) == 1
    assert r.u(1) == 0
    assert r.u(2) == 0b10


def test_bitreader_exp_golomb():
    # 1 -> 0, 010 -> 1, 011 -> 2
    assert BitReader(b"\x80").ue() == 0
    assert BitReader(b"\x40").ue() == 1
    assert BitReader(b"\x60").ue() == 2


def test_bitreader_signed_exp_golomb():
    assert BitReader(b"\x80").se() == 0
    assert BitReader(b"\x40").se() == 1
    assert BitReader(b"\x60").se() == -1


def test_bitreader_rejects_a_truncated_sps():
    r = BitReader(b"\x00")
    try:
        r.u(64)
    except ValueError as exc:
        assert "truncated" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError")


def test_unescape_strips_emulation_prevention():
    assert unescape_rbsp(bytes.fromhex("000003010203")) == bytes.fromhex("0000010203")
    # Only after two zeros, and only for 0x03.
    assert unescape_rbsp(bytes.fromhex("000103")) == bytes.fromhex("000103")


def test_unescape_leaves_clean_data_alone():
    payload = bytes(range(1, 32))
    assert unescape_rbsp(payload) == payload


@pytest.mark.parametrize("payload,expected_fps", REAL_SPS, ids=["24fps", "30fps"])
def test_parse_sps_reads_geometry_and_rate(payload, expected_fps):
    """The bytes on disk are authoritative; this is what reads them."""
    assert parse_sps(unescape_rbsp(payload)) == (1920, 1080, expected_fps)


def test_fps_from_rate():
    assert fps_from_rate("25/1") == 25
    assert fps_from_rate("30000/1001") == 30
    assert fps_from_rate("0/0") == 0
    assert fps_from_rate("") == 0
    assert fps_from_rate("nonsense") == 0


def test_is_elementary_h264():
    assert is_elementary_h264("clip.h264")
    assert is_elementary_h264("CLIP.264")
    assert not is_elementary_h264("clip.mp4")
    assert not is_elementary_h264("clip")


def _annexb(*nals: bytes) -> bytes:
    return b"".join(b"\x00\x00\x00\x01" + n for n in nals)


def _slice_nal(first_mb_zero: bool) -> bytes:
    """One IDR slice NAL. first_mb_in_slice = 0 starts a new picture."""
    # nal_unit_type 5 (IDR), then first_mb_in_slice as Exp-Golomb: 0 -> 0b1,
    # 1 -> 0b010. Padding after that is never read for the count.
    header = b"\x65"
    return header + (b"\x80" + b"\x00" * 30 if first_mb_zero else b"\x40" + b"\x00" * 30)


def test_count_pictures_counts_first_mb_zero_only():
    stream = _annexb(_slice_nal(True), _slice_nal(False), _slice_nal(True))
    count, _ = count_pictures_in(stream, final=True)
    assert count == 2


def test_count_pictures_ignores_non_slice_nals():
    stream = _annexb(b"\x67" + b"\x00" * 20, _slice_nal(True))   # SPS then a slice
    count, _ = count_pictures_in(stream, final=True)
    assert count == 1


def test_count_pictures_defers_a_split_start_code():
    """A buffer ending mid-start-code must hand those bytes to the next chunk."""
    stream = _annexb(_slice_nal(True)) + b"\x00\x00"
    count, consumed = count_pictures_in(stream, final=False)
    assert count == 1
    assert consumed <= len(stream) - 2


def test_count_h264_pictures_on_a_file(tmp_path):
    path = tmp_path / "clip.h264"
    path.write_bytes(_annexb(*[_slice_nal(True) for _ in range(7)]))
    assert count_h264_pictures(str(path)) == 7


def test_count_h264_pictures_on_a_missing_file(tmp_path):
    assert count_h264_pictures(str(tmp_path / "nope.h264")) == 0


def test_probe_sps_on_a_file(tmp_path):
    path = tmp_path / "clip.h264"
    path.write_bytes(_annexb(b"\x67" + SPS_1080P_24, _slice_nal(True)))
    assert probe_h264_sps(str(path)) == (1920, 1080, 24)


def test_probe_sps_returns_zeros_without_an_sps(tmp_path):
    path = tmp_path / "clip.h264"
    path.write_bytes(_annexb(_slice_nal(True)))
    assert probe_h264_sps(str(path)) == (0, 0, 0)


# -- refusing a Neat build that cannot do the job --


def test_an_old_neat_build_is_refused_while_probing(tmp_path, monkeypatch):
    """A DevKit paired with an older SDK has no `SimaDecodeOptions`.

    It surfaced as `AttributeError: module 'pyneat' has no attribute
    'SimaDecodeOptions'` from inside graph construction -- after a model load
    that takes the better part of a minute, and naming a symbol rather than the
    problem. It has to be caught while probing the source, which is the last
    cheap moment before that load.
    """
    from sima_vision import media, runtime

    clip = tmp_path / "clip.h264"
    clip.write_bytes(bytes([0, 0, 0, 1, 0x67]))
    cfg = TASKS["detect"]().load(
        None, {"source.uri": str(clip), "model.path": "m.tar.gz"}, use_file=False
    )

    old = type(runtime)("pyneat")
    old.__version__ = "0.2.2"
    monkeypatch.setattr(runtime, "pyneat", old)

    with pytest.raises(RuntimeError) as caught:
        media.check_source_support(cfg)
    message = str(caught.value)
    assert "0.2.2" in message, "say which build is installed"
    assert "pyneat.SimaDecodeOptions" in message, "and exactly what it is missing"
    assert "sima-cli sdk setup" in message, "and what to do about it"


def test_a_capable_build_passes(tmp_path, monkeypatch):
    from sima_vision import media, runtime

    clip = tmp_path / "clip.h264"
    clip.write_bytes(bytes([0, 0, 0, 1, 0x67]))
    cfg = TASKS["detect"]().load(
        None, {"source.uri": str(clip), "model.path": "m.tar.gz"}, use_file=False
    )

    current = type(runtime)("pyneat")
    current.SimaDecodeOptions = current.SimaDecodeType = object
    monkeypatch.setattr(runtime, "pyneat", current)
    media.check_source_support(cfg)          # must not raise


def test_the_check_is_skipped_when_nothing_is_bound(tmp_path, monkeypatch):
    """--validate never binds pyneat, and must not be made to."""
    from sima_vision import media, runtime

    clip = tmp_path / "clip.h264"
    clip.write_bytes(bytes([0, 0, 0, 1, 0x67]))
    cfg = TASKS["detect"]().load(
        None, {"source.uri": str(clip), "model.path": "m.tar.gz"}, use_file=False
    )
    monkeypatch.setattr(runtime, "pyneat", None)
    media.check_source_support(cfg)


def test_every_source_kind_has_requirements_listed():
    """A source path with no entry would silently skip the check entirely."""
    from sima_vision.media import SOURCE_REQUIREMENTS

    assert set(SOURCE_REQUIREMENTS) == {"h264", "container", "rtsp", "usb"}
