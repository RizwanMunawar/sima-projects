"""Deferred third-party imports, and the config-token to pyneat-enum tables.

``pyneat`` is an aarch64 wheel that only exists on the DevKit, and ``cv2`` comes
from the board's system packages rather than from pip. Importing either at
module scope would make ``sima-vision --help`` and ``sima-vision detect
--validate`` fail on a laptop, which is exactly where you want to check a config
before driving the board with it. So both are left as None here and bound by
:func:`sima_vision.bootstrap.ensure_runtime` at the point of first real use.

This module deliberately imports nothing of its own. It is the bottom of the
stack: :mod:`sima_vision.bootstrap` finds the modules and writes them here, and
everything else reads them back through this module (``from . import runtime``
then ``runtime.cv2``) rather than importing the names directly, because a
``from .runtime import cv2`` binds None once and for all.

The enum tables below are plain strings for the same reason: a config can be
validated, and a wrong ``family=`` rejected, without pyneat being anywhere near.
"""

from __future__ import annotations

import time

#: Bound by :func:`sima_vision.bootstrap.ensure_runtime`.
cv2 = None
np = None
pyneat = None

#: ``cv2.FONT_HERSHEY_SIMPLEX``, filled in once cv2 is available.
FONT = 0


def ready() -> bool:
    """Whether the deferred modules have been bound yet."""
    return pyneat is not None and cv2 is not None and np is not None


def time_ms() -> float:
    return time.perf_counter() * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Config token -> pyneat enum mapping
#
# Kept as string tokens so the tables can be validated before pyneat is
# imported. model.family -> BoxDecodeType attribute name. `yolo11` intentionally
# maps to YoloV8: BoxDecodeType has no YOLO11 member, and Ultralytics YOLO11
# exports the same decoupled DFL detect head as YOLOv8.
# ─────────────────────────────────────────────────────────────────────────────

FAMILY_DECODE_TOKENS: dict[str, str] = {
    "yolo": "Yolo",
    "yolov5": "YoloV5",
    "yolov5-seg": "YoloV5Seg",
    "yolov6": "YoloV6",
    "yolov7": "YoloV7",
    "yolov7-seg": "YoloV7Seg",
    "yolov8": "YoloV8",
    "yolov8-seg": "YoloV8Seg",
    "yolov8-pose": "YoloV8Pose",
    "yolov9": "YoloV9",
    "yolov9-seg": "YoloV9Seg",
    "yolov10": "YoloV10",
    "yolov10-seg": "YoloV10Seg",
    "yolo11": "YoloV8",
    "yolo11-seg": "YoloV8Seg",
    "yolo11-pose": "YoloV8Pose",
    "yolo26": "YoloV26",
    "yolo26-seg": "YoloV26Seg",
    "yolo26-pose": "YoloV26Pose",
    "yolox": "YoloX",
}

#: Families whose head emits mask data as well as boxes.
SEG_FAMILIES = frozenset(name for name in FAMILY_DECODE_TOKENS if name.endswith("-seg"))

DECODE_TYPE_OPTIONS: dict[str, str] = {
    "auto": "Auto",
    "packed_per_head": "PackedPerHead",
    "interleaved_by_head": "InterleavedByHead",
    "grouped_by_role": "GroupedByRole",
    "split3_interleaved": "Split3Interleaved",
    "split3_grouped": "Split3Grouped",
    "interleaved_by_head_probability": "InterleavedByHeadProbability",
    "interleaved_by_head_logit": "InterleavedByHeadLogit",
    "grouped_by_role_probability": "GroupedByRoleProbability",
    "grouped_by_role_logit": "GroupedByRoleLogit",
}

AUTO_FLAGS: dict[str, str] = {"auto": "Auto", "on": "On", "off": "Off"}
INPUT_KINDS: dict[str, str] = {"auto": "Auto", "image": "Image", "tensor": "Tensor"}
RESIZE_MODES: dict[str, str] = {"stretch": "Stretch", "letterbox": "Letterbox", "crop": "Crop"}
COLOR_FORMATS: dict[str, str] = {
    "AUTO": "Auto",
    "RGB": "RGB",
    "BGR": "BGR",
    "GRAY8": "GRAY8",
    "NV12": "NV12",
    "I420": "I420",
}
# NormalizePreset.None is bound under the Python keyword `None`, so it must be
# reached with getattr(). `pyneat.NormalizePreset.None` is a SyntaxError.
NORMALIZE_PRESETS: dict[str, str] = {
    "none": "None",
    "imagenet": "ImageNet",
    "coco_yolo": "COCO_YOLO",
}
SCALING_TYPES = {
    "BILINEAR",
    "NEAREST_NEIGHBOUR",
    "NEAREST_NEIGHBOR",
    "BICUBIC",
    "INTERAREA",
    "INTER_AREA",
    "NO_SCALING",
}
RUN_PRESETS: dict[str, str] = {
    "auto": "Reliable",   # placeholder, resolve_flow_control() picks the real one
    "realtime": "Realtime",
    "balanced": "Balanced",
    "reliable": "Reliable",
}
OVERFLOW_POLICIES: dict[str, str] = {
    "auto": "Block",      # placeholder, resolve_flow_control() picks the real one
    "block": "Block",
    "keep_latest": "KeepLatest",
    "drop_incoming": "DropIncoming",
}


def enum_value(enum_cls, token: str, table: dict[str, str], what: str):
    name = table.get(token)
    if name is None:
        raise ValueError(f"unsupported {what}: {token!r}")
    try:
        return getattr(enum_cls, name)
    except AttributeError as exc:  # pragma: no cover - guards SDK drift
        raise RuntimeError(
            f"{what} {token!r} maps to {enum_cls.__name__}.{name}, which this "
            f"Neat Library build does not expose"
        ) from exc
