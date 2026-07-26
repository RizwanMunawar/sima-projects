"""YOLO object detector for SiMa Modalix — video file / RTSP / camera source,
annotated frames to disk + Neat Insight (H.264 RTP/UDP video + JSON metadata).

Built against the Neat Library public Python API (pyneat) as packaged in
2.1.2_Palette_SDK. Runs on the Modalix DevKit, not in the x86 SDK container.

Pipeline shape (Graph, because there are multiple stages and named endpoints):

    source ──> branch ──> frame ───────────────┐
                    │                          ├─> combine("detector_output")
                    └──> model ──> detections ─┘

    detector_output ──> BBOX parse ──> overlay ──> disk
                                   ├─> MetadataSender (UDP JSON, Insight)
                                   └─> video Graph: Input ──> VideoSender (RTP/UDP, Insight)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

# Imported lazily by load_runtime_dependencies() so --validate-config works off-board.
cv2 = None
np = None
pyneat = None


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PreprocessConfig:
    """Mirrors pyneat.ModelOptions.preprocess (PreprocessOptions)."""

    kind: str = "image"
    enable: str = "on"
    input_format: str = "NV12"
    output_format: str = "auto"
    input_max_width: int = 0
    input_max_height: int = 0

    resize_enable: str = "on"
    resize_width: int = 0
    resize_height: int = 0
    resize_mode: str = "letterbox"
    pad_value: int = 114
    scaling_type: str = "BILINEAR"

    normalize_enable: str = "on"
    normalize_preset: str = "coco_yolo"
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stddev: tuple[float, float, float] = (1.0, 1.0, 1.0)

    quantize_enable: str = "auto"
    quantize_zero_point: int = 0
    quantize_scale: float = 0.0

    tessellate_enable: str = "auto"
    tessellate_slice_shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    family: str
    decode_type_option: str
    num_classes: int

    source_type: str
    source_uri: str
    source_fps: int
    source_width: int
    source_height: int
    rtsp_codec: str
    rtsp_tcp: bool
    rtsp_latency_ms: int
    usb_camera_name: str
    usb_format: str

    preprocess: PreprocessConfig

    score_threshold: float
    nms_iou: float
    max_detections: int

    frames: int
    pull_timeout_ms: int
    queue_depth: int
    run_preset: str
    overflow_policy: str
    profile: bool
    profile_interval: int

    save_enable: bool
    save_dir: str
    save_every: int
    save_overlay: bool
    save_format: str

    video_enable: bool
    video_path: str
    video_codec: str
    video_fps: int
    video_hud: bool

    insight_enable: bool
    insight_host: str
    insight_channel: int
    video_port_base: int
    metadata_port_base: int
    bitrate_kbps: int


def _section(raw: dict, key: str) -> dict:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config section `{key}` must be a mapping")
    return value


def _str(raw: dict, key: str, default: str = "") -> str:
    value = raw.get(key, default)
    return default if value is None else str(value)


def _int(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config key `{key}` must be an integer")
    return int(value)


def _float(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config key `{key}` must be numeric")
    return float(value)


def _bool(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"config key `{key}` must be true or false")
    return value


def _flag(raw: dict, key: str, default: str) -> str:
    """Read a tri-state auto/on/off knob.

    YAML 1.1 resolves bare `on`/`off`/`yes`/`no` to booleans, so `enable: on`
    reaches us as True. Fold those back onto the token vocabulary.
    """
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return "on" if value else "off"
    token = str(value).lower()
    if token in {"yes", "true"}:
        return "on"
    if token in {"no", "false"}:
        return "off"
    if token not in AUTO_FLAGS:
        raise ValueError(f"config key `{key}` must be auto, on or off (got {value!r})")
    return token


def _triple(raw: dict, key: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"config key `{key}` must be a list of 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def load_preprocess_config(raw: dict) -> PreprocessConfig:
    resize = _section(raw, "resize")
    normalize = _section(raw, "normalize")
    quantize = _section(raw, "quantize")
    tessellate = _section(raw, "tessellate")
    slice_shape = tessellate.get("slice_shape") or []

    return PreprocessConfig(
        kind=_str(raw, "kind", "image").lower(),
        enable=_flag(raw, "enable", "on"),
        input_format=_str(raw, "input_format", "NV12").upper(),
        output_format=_str(raw, "output_format", "auto").upper(),
        input_max_width=_int(raw, "input_max_width", 0),
        input_max_height=_int(raw, "input_max_height", 0),
        resize_enable=_flag(resize, "enable", "on"),
        resize_width=_int(resize, "width", 0),
        resize_height=_int(resize, "height", 0),
        resize_mode=_str(resize, "mode", "letterbox").lower(),
        pad_value=_int(resize, "pad_value", 114),
        scaling_type=_str(resize, "scaling_type", "BILINEAR").upper(),
        normalize_enable=_flag(normalize, "enable", "on"),
        normalize_preset=_str(normalize, "preset", "coco_yolo").lower(),
        mean=_triple(normalize, "mean", (0.0, 0.0, 0.0)),
        stddev=_triple(normalize, "stddev", (1.0, 1.0, 1.0)),
        quantize_enable=_flag(quantize, "enable", "auto"),
        quantize_zero_point=_int(quantize, "zero_point", 0),
        quantize_scale=_float(quantize, "scale", 0.0),
        tessellate_enable=_flag(tessellate, "enable", "auto"),
        tessellate_slice_shape=tuple(int(v) for v in slice_shape),
    )


def load_app_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = _section(raw, "model")
    source = _section(raw, "source")
    rtsp = _section(source, "rtsp")
    usb = _section(source, "usb")
    decode = _section(raw, "decode")
    runtime = _section(raw, "runtime")
    output = _section(raw, "output")
    save = _section(output, "save")
    video = _section(output, "video")
    insight = _section(output, "insight")

    default_labels = Path(__file__).resolve().parent / "coco_labels.txt"

    cfg = AppConfig(
        model_path=_str(model, "path"),
        labels_path=Path(_str(model, "labels", str(default_labels))),
        family=_str(model, "family", "yolo26").lower(),
        decode_type_option=_str(model, "decode_type_option", "auto").lower(),
        num_classes=_int(model, "num_classes", 0),
        source_type=_str(source, "type", "video").lower(),
        source_uri=_str(source, "uri"),
        source_fps=_int(source, "fps", 0),
        source_width=_int(source, "width", 0),
        source_height=_int(source, "height", 0),
        rtsp_codec=_str(rtsp, "codec", "h264").lower(),
        rtsp_tcp=_bool(rtsp, "tcp", True),
        rtsp_latency_ms=_int(rtsp, "latency_ms", 100),
        usb_camera_name=_str(usb, "camera_name"),
        usb_format=_str(usb, "format", "NV12").upper(),
        preprocess=load_preprocess_config(_section(raw, "preprocess")),
        score_threshold=_float(decode, "score_threshold", 0.30),
        nms_iou=_float(decode, "nms_iou", 0.60),
        max_detections=_int(decode, "max_detections", 50),
        frames=_int(runtime, "frames", 0),
        pull_timeout_ms=_int(runtime, "pull_timeout_ms", 20000),
        queue_depth=_int(runtime, "queue_depth", 3),
        run_preset=_str(runtime, "preset", "realtime").lower(),
        overflow_policy=_str(runtime, "overflow_policy", "keep_latest").lower(),
        profile=_bool(runtime, "profile", False),
        profile_interval=_int(runtime, "profile_interval", 100),
        save_enable=_bool(save, "enable", True),
        save_dir=_str(save, "dir", "sandbox/object-detection"),
        save_every=_int(save, "every", 10),
        save_overlay=_bool(save, "overlay", True),
        save_format=_str(save, "format", "jpg").lower().lstrip("."),
        video_enable=_bool(video, "enable", True),
        video_path=_str(video, "path", "sandbox/detections.mp4"),
        video_codec=_str(video, "codec", "mp4v"),
        video_fps=_int(video, "fps", 0),
        video_hud=_bool(video, "hud", True),
        insight_enable=_bool(insight, "enable", True),
        insight_host=_str(insight, "host", "127.0.0.1"),
        insight_channel=_int(insight, "channel", 0),
        video_port_base=_int(insight, "video_port_base", 9000),
        metadata_port_base=_int(insight, "metadata_port_base", 9100),
        bitrate_kbps=_int(insight, "bitrate_kbps", 2000),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    if cfg.family not in FAMILY_DECODE_TOKENS:
        raise ValueError(
            f"model.family `{cfg.family}` is not supported. "
            f"Choose one of: {', '.join(sorted(FAMILY_DECODE_TOKENS))}"
        )
    if cfg.source_type not in {"video", "rtsp", "usb"}:
        raise ValueError("source.type must be video, rtsp or usb")
    if cfg.source_type in {"video", "rtsp"} and not cfg.source_uri:
        raise ValueError(f"source.uri must be set for source.type={cfg.source_type}")
    if cfg.rtsp_codec not in {"h264", "mjpeg"}:
        raise ValueError("source.rtsp.codec must be h264 or mjpeg")
    if cfg.preprocess.kind not in {"image", "tensor", "auto"}:
        raise ValueError("preprocess.kind must be image, tensor or auto")
    if cfg.preprocess.resize_mode not in RESIZE_MODES:
        raise ValueError(f"preprocess.resize.mode must be one of: {', '.join(RESIZE_MODES)}")
    if cfg.preprocess.scaling_type not in SCALING_TYPES:
        raise ValueError(
            f"preprocess.resize.scaling_type must be one of: {', '.join(sorted(SCALING_TYPES))}"
        )
    if cfg.preprocess.input_format not in COLOR_FORMATS:
        raise ValueError(f"preprocess.input_format must be one of: {', '.join(COLOR_FORMATS)}")
    if cfg.preprocess.output_format not in COLOR_FORMATS:
        raise ValueError(f"preprocess.output_format must be one of: {', '.join(COLOR_FORMATS)}")
    if cfg.preprocess.normalize_preset not in NORMALIZE_PRESETS:
        raise ValueError(
            f"preprocess.normalize.preset must be one of: {', '.join(NORMALIZE_PRESETS)}"
        )
    if not 0.0 <= cfg.score_threshold <= 1.0:
        raise ValueError("decode.score_threshold must be in [0.0, 1.0]")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("decode.nms_iou must be in [0.0, 1.0]")
    if cfg.max_detections < 0:
        raise ValueError("decode.max_detections must be >= 0")
    if cfg.frames < 0:
        raise ValueError("runtime.frames must be >= 0")
    if cfg.pull_timeout_ms <= 0:
        raise ValueError("runtime.pull_timeout_ms must be > 0")
    if cfg.profile_interval <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save.every must be >= 0")
    if cfg.save_format not in {"jpg", "jpeg", "png"}:
        raise ValueError("output.save.format must be jpg or png")
    if cfg.video_enable and not cfg.video_path:
        raise ValueError("output.video.path must be set when video output is enabled")
    if len(cfg.video_codec) != 4:
        raise ValueError(
            f"output.video.codec must be a 4-character FourCC such as mp4v or MJPG, "
            f"got {cfg.video_codec!r}"
        )
    if cfg.video_fps < 0:
        raise ValueError("output.video.fps must be >= 0")
    if cfg.insight_enable and not cfg.insight_host:
        raise ValueError("output.insight.host must be set when insight is enabled")
    if not (cfg.save_enable or cfg.insight_enable or cfg.video_enable):
        raise ValueError(
            "enable at least one of output.save, output.video or output.insight"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Config token -> pyneat enum mapping
#
# Kept as string tokens so the tables can be validated before pyneat is
# imported (the wheel is aarch64-only and does not load in the SDK container).
# ─────────────────────────────────────────────────────────────────────────────

# model.family -> BoxDecodeType attribute name.
#
# `yolo11` intentionally maps to YoloV8: BoxDecodeType has no YOLO11 member, and
# Ultralytics YOLO11 exports the same decoupled DFL detect head as YOLOv8.
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
# reached with getattr() — `pyneat.NormalizePreset.None` is a SyntaxError.
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
    "realtime": "Realtime",
    "balanced": "Balanced",
    "reliable": "Reliable",
}
OVERFLOW_POLICIES: dict[str, str] = {
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


# ─────────────────────────────────────────────────────────────────────────────
# Runtime deps
# ─────────────────────────────────────────────────────────────────────────────


def load_runtime_dependencies() -> None:
    global cv2, np, pyneat
    if pyneat is not None:
        return
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)
    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module

    cv2, np, pyneat = cv2_module, np_module, pyneat_module


def time_ms() -> float:
    return time.perf_counter() * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing — the intent layer handed to the route planner
# ─────────────────────────────────────────────────────────────────────────────


def apply_preprocess_options(opt, pre: PreprocessConfig, frame_w: int, frame_h: int) -> None:
    """Translate the `preprocess:` config block onto pyneat.ModelOptions.preprocess.

    `Model` resolves this against the archive's MPK contract and builds the
    matching Preproc / Quant / Tess / QuantTess graph family. Anything left on
    `auto` is decided by the planner.
    """
    p = opt.preprocess

    p.kind = enum_value(pyneat.InputKind, pre.kind, INPUT_KINDS, "preprocess.kind")
    p.enable = enum_value(pyneat.AutoFlag, pre.enable, AUTO_FLAGS, "preprocess.enable")

    # Buffer capacity for the Preproc node. Defaults to 1920x1080 when left at 0;
    # pin it to the real stream geometry so buffers are sized correctly.
    p.input_max_width = pre.input_max_width or frame_w
    p.input_max_height = pre.input_max_height or frame_h

    # ── Resize / letterbox ──
    p.resize.enable = enum_value(
        pyneat.AutoFlag, pre.resize_enable, AUTO_FLAGS, "preprocess.resize.enable"
    )
    # 0 means "infer the target from the model input contract".
    p.resize.width = pre.resize_width
    p.resize.height = pre.resize_height
    p.resize.mode = enum_value(
        pyneat.ResizeMode, pre.resize_mode, RESIZE_MODES, "preprocess.resize.mode"
    )
    p.resize.pad_value = pre.pad_value
    p.resize.scaling_type = pre.scaling_type

    # ── Colour conversion ──
    # input_format must match what the source hands to Preproc: NV12 for the
    # hardware H.264 decoder and libcamera, BGR for cv2.imread.
    p.color_convert.input_format = enum_value(
        pyneat.PreprocessColorFormat,
        pre.input_format,
        COLOR_FORMATS,
        "preprocess.input_format",
    )
    p.color_convert.output_format = enum_value(
        pyneat.PreprocessColorFormat,
        pre.output_format,
        COLOR_FORMATS,
        "preprocess.output_format",
    )
    if pre.input_format != "AUTO" or pre.output_format != "AUTO":
        p.color_convert.enable = pyneat.AutoFlag.On

    # ── Normalisation ──
    p.normalize.enable = enum_value(
        pyneat.AutoFlag, pre.normalize_enable, AUTO_FLAGS, "preprocess.normalize.enable"
    )
    p.preset = enum_value(
        pyneat.NormalizePreset,
        pre.normalize_preset,
        NORMALIZE_PRESETS,
        "preprocess.normalize.preset",
    )
    if pre.normalize_preset == "none":
        # Explicit stats are only read when no preset supplies them.
        p.normalize.mean = list(pre.mean)
        p.normalize.stddev = list(pre.stddev)

    # ── Quantise / tessellate ──
    # Normally left on auto: the model pack's dtype contract decides whether
    # these run in the Preproc graph or inside the MLA.
    p.quantize.enable = enum_value(
        pyneat.AutoFlag, pre.quantize_enable, AUTO_FLAGS, "preprocess.quantize.enable"
    )
    if pre.quantize_enable == "on":
        p.quantize.zero_point = pre.quantize_zero_point
        p.quantize.scale = pre.quantize_scale

    p.tessellate.enable = enum_value(
        pyneat.AutoFlag, pre.tessellate_enable, AUTO_FLAGS, "preprocess.tessellate.enable"
    )
    if pre.tessellate_slice_shape:
        p.tessellate.set_slice_shape(list(pre.tessellate_slice_shape))


def make_model(cfg: AppConfig, frame_w: int, frame_h: int):
    opt = pyneat.ModelOptions()
    apply_preprocess_options(opt, cfg.preprocess, frame_w, frame_h)

    opt.decode_type = enum_value(
        pyneat.BoxDecodeType, cfg.family, FAMILY_DECODE_TOKENS, "model.family"
    )
    opt.decode_type_option = enum_value(
        pyneat.BoxDecodeTypeOption,
        cfg.decode_type_option,
        DECODE_TYPE_OPTIONS,
        "model.decode_type_option",
    )
    # 0 on any of these preserves the value packaged in the model archive.
    opt.score_threshold = cfg.score_threshold
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    if cfg.num_classes > 0:
        opt.num_classes = cfg.num_classes

    if not Path(cfg.model_path).exists():
        raise RuntimeError(f"model archive not found: {cfg.model_path}")
    return pyneat.Model(cfg.model_path, opt)


def describe_preprocess(cfg: AppConfig, frame_w: int, frame_h: int) -> str:
    pre = cfg.preprocess
    target = (
        f"{pre.resize_width}x{pre.resize_height}"
        if pre.resize_width and pre.resize_height
        else "<from model contract>"
    )
    norm = (
        pre.normalize_preset
        if pre.normalize_preset != "none"
        else f"mean={list(pre.mean)} stddev={list(pre.stddev)}"
    )
    return (
        f"preprocess: kind={pre.kind} enable={pre.enable} "
        f"in={pre.input_format} out={pre.output_format} "
        f"capacity={pre.input_max_width or frame_w}x{pre.input_max_height or frame_h} | "
        f"resize={pre.resize_mode} target={target} pad={pre.pad_value} "
        f"scaler={pre.scaling_type} | normalize={norm} | "
        f"quantize={pre.quantize_enable} tessellate={pre.tessellate_enable}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Source geometry + source graph
# ─────────────────────────────────────────────────────────────────────────────


def fps_from_rate(value: str) -> int:
    if not value or value in {"0/0", "0/1"}:
        return 0
    try:
        fps = float(Fraction(value)) if "/" in value else float(value)
    except (ValueError, ZeroDivisionError):
        return 0
    return int(round(fps)) if fps > 0 else 0


def probe_ffprobe(uri: str) -> tuple[int, int, int]:
    cmd = [
        "ffprobe", "-v", "error", "-rw_timeout", "5000000",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
        "-of", "default=nw=1", uri,
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0, 0, 0
    if result.returncode != 0:
        return 0, 0, 0
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    fps = fps_from_rate(values.get("avg_frame_rate", "")) or fps_from_rate(
        values.get("r_frame_rate", "")
    )

    def as_int(v: str | None) -> int:
        try:
            return int(v or 0)
        except ValueError:
            return 0

    return as_int(values.get("width")), as_int(values.get("height")), fps


def probe_opencv(uri: str) -> tuple[int, int, int]:
    """Best-effort probe. Returns zeros rather than raising.

    Raw H.264 elementary streams frequently cannot be opened by OpenCV, and that is
    not a fatal condition: the caller falls back to the configured geometry and
    produces a clearer message than "failed to open source".
    """
    try:
        cap = cv2.VideoCapture(uri)
    except Exception:
        return 0, 0, 0
    if not cap.isOpened():
        cap.release()
        return 0, 0, 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    return width, height, fps


def resolve_source_geometry(cfg: AppConfig) -> tuple[int, int, int]:
    """Return (width, height, fps). Config values win; anything left at 0 is probed."""
    width, height, fps = cfg.source_width, cfg.source_height, cfg.source_fps

    if cfg.source_type == "usb":
        # libcamera is queried at build time, not probeable here — fall back to
        # the CameraInputOptions defaults.
        return width or 1920, height or 1080, fps or 30

    if width <= 0 or height <= 0 or fps <= 0:
        probed_w, probed_h, probed_fps = probe_ffprobe(cfg.source_uri)
        width = width if width > 0 else probed_w
        height = height if height > 0 else probed_h
        fps = fps if fps > 0 else probed_fps

    if width <= 0 or height <= 0 or fps <= 0:
        cv_w, cv_h, cv_fps = probe_opencv(cfg.source_uri)
        width = width if width > 0 else cv_w
        height = height if height > 0 else cv_h
        fps = fps if fps > 0 else cv_fps

    if width <= 0 or height <= 0 or fps <= 0:
        hint = ""
        if cfg.source_type == "video" and is_elementary_h264(cfg.source_uri):
            hint = (
                "\nRaw H.264 elementary streams carry no container metadata, so "
                "geometry usually\ncannot be probed. Set them explicitly in config.yaml:\n"
                "  source:\n    width: 1920\n    height: 1080\n    fps: 25"
            )
        missing = []
        if width <= 0 or height <= 0:
            missing.append("source.width and source.height")
        if fps <= 0:
            missing.append("source.fps")
        raise RuntimeError(f"could not resolve {', '.join(missing)}{hint}")
    return width, height, fps


def set_output_caps(caps, fps: int, width: int, height: int) -> None:
    if width <= 0 or height <= 0 or fps <= 0:
        return
    caps.enable = True
    caps.format = pyneat.Format.NV12
    caps.width = width
    caps.height = height
    caps.fps = fps
    caps.memory = pyneat.CapsMemory.Any


ELEMENTARY_H264_SUFFIXES = {".h264", ".264", ".bin", ".avc"}


def is_elementary_h264(path: str) -> bool:
    return Path(path).suffix.lower() in ELEMENTARY_H264_SUFFIXES


def make_elementary_h264_source(cfg: AppConfig, width: int, height: int, fps: int):
    """VideoInputGroup rebuilt by hand, minus the demuxer.

    Works around a Neat 0.3.0 bug. `VideoTrackSelect` emits its fragment as
    `qtdemux name=<base> <base>.video_0`, which is internally consistent, but the
    graph then appends an instance suffix to element *names* only. The declaration
    becomes `name=n1_demux_8` while the pad reference stays `n1_demux.video_0`, so
    gst_parse_launch fails with:

        No src-element named "n1_demux" - omitting link

    `element_names()` reports just the one name, so the renamer never learns to
    rewrite the pad reference, and any non-empty suffix breaks it. Reordering graph
    construction does not help.

    Dropping the container removes the demuxer, and with it the bug. Convert once:

        ffmpeg -i input.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 output.h264
    """
    graph = pyneat.Graph("file_source")
    graph.add(pyneat.nodes.file_input(cfg.source_uri))
    graph.add(pyneat.nodes.h264_parse(config_interval=1))
    graph.add(pyneat.nodes.queue())

    dec = pyneat.SimaDecodeOptions()
    dec.type = pyneat.SimaDecodeType.H264
    dec.sima_allocator_type = 2
    dec.out_format = pyneat.Format.NV12
    dec.raw_output = False
    graph.add(pyneat.nodes.sima_decode(dec))

    if width > 0 and height > 0 and fps > 0:
        # nodes.caps_raw takes the format as a plain string, unlike the *Options
        # `format` properties which accept the pyneat.Format enum.
        graph.add(
            pyneat.nodes.caps_raw("NV12", width, height, fps, pyneat.CapsMemory.Any)
        )
    return graph


def make_source_graph(cfg: AppConfig, width: int, height: int, fps: int):
    """File / RTSP / camera head of the Graph. All three produce NV12 frames."""
    if cfg.source_type == "video":
        if is_elementary_h264(cfg.source_uri):
            print("source: raw H.264 elementary stream, demuxer bypassed")
            return make_elementary_h264_source(cfg, width, height, fps)

        print(
            "[warn] container input uses groups.video_input, which hits a demuxer\n"
            "       naming bug in Neat 0.3.0. If the pipeline fails to start with\n"
            "       'No src-element named \"nN_demux\"', convert to a raw stream:\n"
            f"         ffmpeg -i {cfg.source_uri} -c:v copy -bsf:v h264_mp4toannexb \\\n"
            f"           -f h264 {Path(cfg.source_uri).with_suffix('.h264')}\n"
            "       then point source.uri at the .h264 file.",
            file=sys.stderr,
        )
        opt = pyneat.VideoInputGroupOptions()
        opt.path = cfg.source_uri
        opt.insert_queue = True
        opt.sync_mode = False
        opt.out_format = pyneat.Format.NV12
        set_output_caps(opt.output_caps, fps, width, height)
        return pyneat.groups.video_input(opt)

    if cfg.source_type == "rtsp":
        opt = pyneat.RtspDecodedInputOptions()
        opt.url = cfg.source_uri
        opt.latency_ms = cfg.rtsp_latency_ms
        opt.tcp = cfg.rtsp_tcp
        opt.insert_queue = True
        opt.decoder_name = "decoder"
        opt.decoder_raw_output = True
        opt.source_fps = fps
        opt.codec = (
            pyneat.RtspCodec.H264 if cfg.rtsp_codec == "h264" else pyneat.RtspCodec.MJPEG
        )
        if cfg.rtsp_codec == "h264":
            opt.payload_type = 96
            opt.auto_caps_from_stream = True
            opt.fallback_h264_width = width
            opt.fallback_h264_height = height
        else:
            opt.mjpeg_payload_type = 26
            opt.dec_width = width
            opt.dec_height = height
        set_output_caps(opt.output_caps, fps, width, height)
        return pyneat.groups.rtsp_decoded_input(opt)

    # usb / on-board camera — libcamera-backed. Confirm the device is visible
    # with `cam -l` on the DevKit and put its name in source.usb.camera_name.
    opt = pyneat.CameraInputOptions()
    opt.camera_name = cfg.usb_camera_name or None
    opt.width = width
    opt.height = height
    opt.framerate_num = fps
    opt.framerate_den = 1
    opt.format = cfg.usb_format
    opt.insert_queue = True
    opt.leaky_queue = True
    return pyneat.nodes.camera_input(opt)


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Pipeline:
    model: object
    graph: object
    run: object
    video_graph: object = None
    video_run: object = None
    metadata_sender: object = None
    labels: list[str] = field(default_factory=list)
    frame_w: int = 0
    frame_h: int = 0
    fps: int = 0
    video_port: int = 0
    writer: object = None
    writer_path: str = ""
    writer_frames: int = 0

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.release()
                size = Path(self.writer_path).stat().st_size / 1e6
                print(
                    f"video: wrote {self.writer_frames} frames to {self.writer_path} "
                    f"({size:.1f} MB)",
                    flush=True,
                )
            except Exception as exc:
                print(f"[warn] closing video writer failed: {exc}", file=sys.stderr)
            self.writer = None
        for handle in (self.video_run, self.run):
            if handle is None:
                continue
            try:
                handle.close()
            except Exception as exc:  # pragma: no cover - teardown must not mask errors
                print(f"[warn] close failed: {exc}", file=sys.stderr)


def open_video_writer(cfg: AppConfig, width: int, height: int, fps: int):
    """Annotated MP4 written on the DevKit. Returns (writer, path) or (None, '')."""
    if not cfg.video_enable:
        return None, ""

    path = Path(cfg.video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_fps = cfg.video_fps or fps or 25

    fourcc = cv2.VideoWriter_fourcc(*cfg.video_codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(out_fps), (width, height))
    if writer.isOpened():
        return writer, str(path)

    # mp4v is not always built into the DevKit's OpenCV. MJPG in an AVI works
    # essentially everywhere, at the cost of a much larger file.
    writer.release()
    fallback = path.with_suffix(".avi")
    print(
        f"[warn] codec {cfg.video_codec!r} unavailable for {path.name}, "
        f"falling back to MJPG/{fallback.name}",
        file=sys.stderr,
    )
    writer = cv2.VideoWriter(
        str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), float(out_fps), (width, height)
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"could not open a video writer for {path} with codec {cfg.video_codec} "
            f"or MJPG fallback. Set output.video.enable: false to skip video output."
        )
    return writer, str(fallback)


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise RuntimeError(f"labels file is empty: {path}")
    return labels


def make_run_options(cfg: AppConfig):
    run_options = pyneat.RunOptions()
    run_options.preset = enum_value(
        pyneat.RunPreset, cfg.run_preset, RUN_PRESETS, "runtime.preset"
    )
    run_options.queue_depth = cfg.queue_depth
    run_options.overflow_policy = enum_value(
        pyneat.OverflowPolicy, cfg.overflow_policy, OVERFLOW_POLICIES, "runtime.overflow_policy"
    )
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return run_options


def build_detector_graph(cfg: AppConfig, model, width: int, height: int, fps: int):
    """source -> branch -> {frame, model->detections} -> combine("detector_output")."""
    source = make_source_graph(cfg, width, height, fps)
    branch = pyneat.graphs.branch("source", ["frame", "model"])

    frame_graph = pyneat.Graph("frame")
    frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)

    detections_graph = pyneat.Graph("detections")
    detections_graph.add(
        pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4))
    )

    joined = pyneat.graphs.combine(
        ["frame", "detections"], "detector_output", pyneat.CombinePolicy.ByFrame
    )

    graph = pyneat.Graph("yolo_detector")
    graph.connect(source, branch)
    graph.connect(branch, frame_graph)
    graph.connect(branch, model_graph)
    graph.connect(model_graph, detections_graph)
    graph.connect(frame_graph, joined)
    graph.connect(detections_graph, joined)
    return graph


def build_video_graph(cfg: AppConfig, width: int, height: int, fps: int):
    """Separate Graph that H.264-encodes pushed RGB frames and sends RTP/UDP to Insight."""
    input_options = pyneat.InputOptions()
    input_options.payload_type = pyneat.PayloadType.Image
    input_options.format = pyneat.Format.RGB
    input_options.width = width
    input_options.height = height
    input_options.depth = 3
    input_options.fps_n = fps
    input_options.fps_d = 1
    input_options.memory_policy = pyneat.InputMemoryPolicy.Ev74

    sender_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, fps)
    sender_options.host = cfg.insight_host
    sender_options.channel = cfg.insight_channel
    sender_options.video_port_base = cfg.video_port_base
    sender_options.encoder.bitrate_kbps = cfg.bitrate_kbps

    graph = pyneat.Graph("video")
    graph.add(pyneat.nodes.input(input_options))
    graph.add(pyneat.groups.video_sender(sender_options))

    seed = pyneat.Tensor.from_numpy(
        np.zeros((height, width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )
    return graph, graph.build([seed]), sender_options.video_port


def build_pipeline(cfg: AppConfig) -> Pipeline:
    step = lambda msg: print(msg, flush=True)

    width, height, fps = resolve_source_geometry(cfg)
    step(f"source: type={cfg.source_type} uri={cfg.source_uri or '<default camera>'} "
         f"stream={width}x{height}@{fps}")
    step(describe_preprocess(cfg, width, height))

    step("loading model (first load unpacks the archive, this can take a minute)...")
    model = make_model(cfg, width, height)
    labels = load_labels(cfg.labels_path)
    step(
        f"model: {cfg.model_path} family={cfg.family} "
        f"decode_type={FAMILY_DECODE_TOKENS[cfg.family]} labels={len(labels)}"
    )

    step("building graph...")
    graph = build_detector_graph(cfg, model, width, height, fps)
    if cfg.profile:
        step(f"Backend:\n{graph.describe_backend()}")
    run = graph.build(make_run_options(cfg))
    step("graph built")

    pipeline = Pipeline(
        model=model, graph=graph, run=run, labels=labels,
        frame_w=width, frame_h=height, fps=fps,
    )

    if cfg.insight_enable:
        step("starting Insight senders...")
        pipeline.video_graph, pipeline.video_run, pipeline.video_port = build_video_graph(
            cfg, width, height, fps
        )
        metadata_options = pyneat.MetadataSenderOptions()
        metadata_options.host = cfg.insight_host
        metadata_options.channel = cfg.insight_channel
        metadata_options.metadata_port_base = cfg.metadata_port_base
        pipeline.metadata_sender = pyneat.MetadataSender(metadata_options)
        step(
            f"insight: host={cfg.insight_host} video={pipeline.video_port} "
            f"metadata={pipeline.metadata_sender.metadata_port()} "
            f"channel={cfg.insight_channel}"
        )
        step(f"  view at https://localhost:9900 and select channel {cfg.insight_channel}")
    if cfg.save_enable:
        step(f"save: dir={cfg.save_dir} every={cfg.save_every} overlay={cfg.save_overlay}")
    if cfg.video_enable:
        pipeline.writer, pipeline.writer_path = open_video_writer(cfg, width, height, fps)
        step(
            f"video: {pipeline.writer_path} codec={cfg.video_codec} "
            f"fps={cfg.video_fps or fps} hud={cfg.video_hud}"
        )
    step("running. press Ctrl-C to stop.")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# BBOX payload
#
# BoxDecode emits one UInt8 tensor tagged BBOX per frame:
#   [uint32 N][RawBox 24B] * N ... trailing padding
#   RawBox = <iiiifi  -> x, y, w, h, score, class_id  (source-image pixels)
# ─────────────────────────────────────────────────────────────────────────────

BBOX_RECORD = struct.Struct("<iiiifi")
BBOX_RECORD_SIZE = BBOX_RECORD.size  # 24


def tensor_bbox_payload(sample, tensor=None) -> bytes:
    tensor = tensor if tensor is not None else getattr(sample, "tensor", None)
    if tensor is None:
        raise RuntimeError("detection sample carries no tensor")
    fmt = getattr(sample, "payload_tag", "") or getattr(sample, "format", "")
    semantic = getattr(tensor, "semantic", None)
    tess = getattr(semantic, "tess", None)
    if not fmt and tess is not None:
        fmt = getattr(tess, "format", "")
    fmt = str(fmt).upper()
    if fmt and fmt != "BBOX":
        raise RuntimeError(
            f"expected a BBOX tensor but got {fmt}. If this is `raw_output_heads`, "
            "the route did not include BoxDecode — check model.family / the model archive."
        )
    payload = tensor.copy_payload_bytes()
    if not payload:
        raise RuntimeError("empty BBOX payload")
    return payload


def extract_bbox_payload(sample) -> bytes:
    if sample.kind == pyneat.SampleKind.Bundle:
        for candidate in sample.fields:
            try:
                return extract_bbox_payload(candidate)
            except RuntimeError:
                continue
        raise RuntimeError("bundle has no BBOX field")
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return tensor_bbox_payload(sample, sample.tensors[0])
    if sample.kind != pyneat.SampleKind.Tensor:
        raise RuntimeError(f"unexpected sample kind {sample.kind}")
    return tensor_bbox_payload(sample)


def parse_boxes(payload: bytes, img_w: int, img_h: int, expected_topk: int) -> list[dict]:
    if len(payload) < 4:
        raise RuntimeError("BBOX buffer too small to hold a count header")
    count = struct.unpack_from("<I", payload, 0)[0]
    capacity = (len(payload) - 4) // BBOX_RECORD_SIZE
    if count > capacity:
        raise RuntimeError(f"BBOX header count {count} exceeds payload capacity {capacity}")
    if expected_topk > 0 and count > expected_topk:
        raise RuntimeError(f"BBOX header count {count} exceeds top_k {expected_topk}")

    boxes = []
    offset = 4
    for _ in range(count):
        x, y, w, h, score, class_id = BBOX_RECORD.unpack_from(payload, offset)
        offset += BBOX_RECORD_SIZE
        boxes.append(
            {
                "x1": max(0.0, min(float(x), float(img_w))),
                "y1": max(0.0, min(float(y), float(img_h))),
                "x2": max(0.0, min(float(x + w), float(img_w))),
                "y2": max(0.0, min(float(y + h), float(img_h))),
                "score": float(score),
                "class_id": int(class_id),
            }
        )
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Frames, overlay, sinks
# ─────────────────────────────────────────────────────────────────────────────


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def first_tensor(sample):
    if sample is None:
        return None
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return sample.tensors[0]
    for candidate in sample.fields:
        tensor = first_tensor(candidate)
        if tensor is not None:
            return tensor
    return None


def find_field(sample, label: str):
    if getattr(sample, "stream_label", "") == label:
        return sample
    for candidate in getattr(sample, "fields", []):
        found = find_field(candidate, label)
        if found is not None:
            return found
    return None


def joined_field(sample, label: str, bundle_index: int):
    field_sample = find_field(sample, label)
    if field_sample is not None:
        return field_sample
    fields = list(getattr(sample, "fields", []))
    if getattr(sample, "kind", None) == pyneat.SampleKind.Bundle and len(fields) > bundle_index:
        return fields[bundle_index]
    raise RuntimeError(f"joined output is missing the `{label}` field")


def frame_to_bgr(tensor):
    """Decoded frames arrive as NV12 from the hardware decoder / libcamera."""
    if tensor.is_nv12() or tensor.is_i420():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"YUV payload too small: {payload.size} < {expected}")
        planar = payload[:expected].reshape((height * 3 // 2, width))
        code = cv2.COLOR_YUV2BGR_NV12 if tensor.is_nv12() else cv2.COLOR_YUV2BGR_I420
        return np.ascontiguousarray(cv2.cvtColor(planar, code))

    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


# A 20-colour palette with even hue spacing and consistent saturation, so
# neighbouring classes stay distinguishable and nothing vanishes against a
# bright or dark frame. BGR, because that is what OpenCV expects.
BOX_COLORS = [
    (56, 56, 255), (49, 210, 207), (10, 249, 72), (227, 195, 0), (255, 112, 132),
    (144, 31, 255), (29, 178, 255), (49, 121, 255), (0, 194, 255), (98, 205, 0),
    (185, 243, 52), (255, 156, 87), (255, 88, 178), (184, 61, 245), (86, 96, 255),
    (0, 151, 255), (0, 229, 178), (146, 255, 51), (255, 194, 26), (255, 92, 92),
]

FONT = cv2.FONT_HERSHEY_DUPLEX if cv2 is not None else 0


def class_color(class_id: int) -> tuple[int, int, int]:
    return BOX_COLORS[class_id % len(BOX_COLORS)]


def draw_corner_box(frame, x1, y1, x2, y2, color, thickness: int) -> None:
    """Box with weighted corner brackets. Reads clearly over busy footage."""
    corner = max(12, int(min(x2 - x1, y2 - y1) * 0.18))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, thickness - 1), cv2.LINE_AA)
    heavy = thickness + 1
    for cx, cy, dx, dy in (
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
    ):
        cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, heavy, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, heavy, cv2.LINE_AA)


def draw_label(frame, x1, y1, text, color, scale: float, thickness: int, top_margin: int = 0):
    """Filled caption above the box, flipped inside when it would clip the top.

    `top_margin` reserves the HUD strip so captions never disappear behind it.
    """
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    pad_x, pad_y = int(th * 0.5), int(th * 0.4)
    box_h = th + base + pad_y * 2

    top = y1 - box_h
    below = top < top_margin
    if below:
        top = max(y1, top_margin)
    top = max(top_margin, min(top, frame.shape[0] - box_h))
    left = max(0, min(x1, frame.shape[1] - (tw + pad_x * 2)))

    cv2.rectangle(
        frame, (left, top), (left + tw + pad_x * 2, top + box_h), color, -1, cv2.LINE_AA
    )
    # Luminance decides black or white text, so light and dark colours both read.
    b, g, r = color
    ink = (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)
    cv2.putText(
        frame, text, (left + pad_x, top + th + pad_y), FONT, scale, ink, thickness, cv2.LINE_AA
    )
    return below


def draw_hud(frame, boxes: list[dict], labels: list[str], fps: float) -> int:
    """Translucent summary strip: frame rate, object count, top classes.

    Returns its height so box captions can avoid drawing underneath it.
    """
    h, w = frame.shape[:2]
    scale = max(0.5, min(w, h) / 1400.0)
    pad = int(14 * scale * 2)
    bar_h = int(46 * scale * 2)

    overlay = frame[0:bar_h, 0:w].copy()
    cv2.rectangle(frame, (0, 0), (w, bar_h), (24, 24, 24), -1)
    cv2.addWeighted(frame[0:bar_h, 0:w], 0.55, overlay, 0.45, 0, frame[0:bar_h, 0:w])
    cv2.line(frame, (0, bar_h), (w, bar_h), (0, 229, 178), max(1, int(2 * scale * 2)))

    counts: dict[str, int] = {}
    for box in boxes:
        cid = int(box["class_id"])
        name = labels[cid] if 0 <= cid < len(labels) else str(cid)
        counts[name] = counts.get(name, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
    summary = "  ".join(f"{n} x{c}" for n, c in top) if top else "no detections"

    text = f"{fps:5.1f} FPS   |   {len(boxes)} objects   |   {summary}"
    cv2.putText(
        frame, text, (pad, int(bar_h * 0.66)), FONT,
        0.6 * scale * 2, (240, 240, 240), max(1, int(scale * 2)), cv2.LINE_AA,
    )
    return bar_h


def draw_boxes(frame, boxes: list[dict], labels: list[str], top_margin: int = 0) -> None:
    h, w = frame.shape[:2]
    # Scale strokes and text to the frame, so 4K does not get hairlines and
    # 480p does not get slabs.
    scale = max(0.45, min(w, h) / 1400.0)
    thickness = max(2, int(round(2.4 * scale)))
    font_scale = 0.62 * scale

    # Paint larger boxes first, so small foreground objects stay legible.
    ordered = sorted(
        boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True
    )
    for box in ordered:
        x1 = max(0, int(round(box["x1"])))
        y1 = max(0, int(round(box["y1"])))
        x2 = min(w - 1, int(round(box["x2"])))
        y2 = min(h - 1, int(round(box["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(box["class_id"])
        color = class_color(class_id)
        name = labels[class_id] if 0 <= class_id < len(labels) else str(class_id)
        draw_corner_box(frame, x1, y1, x2, y2, color, thickness)
        draw_label(
            frame, x1, y1, f"{name}  {box['score'] * 100:.0f}%", color, font_scale, 1,
            top_margin,
        )


def metadata_objects(boxes: list[dict], labels: list[str], w: int, h: int) -> list[dict]:
    objects = []
    for index, box in enumerate(boxes, start=1):
        x = max(0, int(box["x1"]))
        y = max(0, int(box["y1"]))
        bw = max(0, int(box["x2"] - box["x1"]))
        bh = max(0, int(box["y2"] - box["y1"]))
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        class_id = int(box["class_id"])
        objects.append(
            {
                "id": f"obj_{index}",
                "label": labels[class_id] if 0 <= class_id < len(labels) else "unknown",
                "confidence": float(box["score"]),
                "bbox": [float(x), float(y), float(max(0, bw)), float(max(0, bh))],
            }
        )
    return objects


def send_metadata(pipeline: Pipeline, sample, boxes: list[dict]) -> None:
    if pipeline.metadata_sender is None:
        return
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    pipeline.metadata_sender.send_metadata(
        "object-detection",
        json.dumps(
            {"objects": metadata_objects(boxes, pipeline.labels, pipeline.frame_w, pipeline.frame_h)},
            separators=(",", ":"),
        ),
        timestamp_ms,
        frame_id,
    )


def push_video(pipeline: Pipeline, sample, frame_bgr) -> None:
    if pipeline.video_run is None:
        return
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = pyneat.Tensor.from_numpy(
        np.ascontiguousarray(rgb),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )
    video_sample = pyneat.make_tensor_sample("", tensor)
    video_sample.pts_ns = sample.pts_ns
    video_sample.dts_ns = sample.dts_ns
    video_sample.duration_ns = sample.duration_ns
    video_sample.frame_id = sample.frame_id
    video_sample.stream_id = sample.stream_id
    if not pipeline.video_run.push([video_sample]):
        raise RuntimeError("Insight video push failed")


def wants_jpeg(cfg: AppConfig, index: int) -> bool:
    return cfg.save_enable and cfg.save_every > 0 and index % cfg.save_every == 0


def render_annotated(cfg: AppConfig, pipeline: Pipeline, frame, boxes: list[dict], fps: float):
    """Draw once per frame and share the result between the video and JPEG sinks."""
    annotated = frame.copy()
    # HUD first, so box captions can reserve its height and stay visible.
    top_margin = draw_hud(annotated, boxes, pipeline.labels, fps) if cfg.video_hud else 0
    draw_boxes(annotated, boxes, pipeline.labels, top_margin)
    return annotated


def save_frame(cfg: AppConfig, index: int, frame) -> None:
    out_path = Path(cfg.save_dir) / f"frame_{index:06d}.{cfg.save_format}"
    if not cv2.imwrite(str(out_path), frame):
        print(f"[warn] failed to write {out_path}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Run loop
# ─────────────────────────────────────────────────────────────────────────────


class Stopper:
    """Clean teardown on SIGINT/SIGTERM/SIGHUP.

    `dk` / `devkit-run` invoke over SSH without a pty, so a terminal Ctrl-C is
    not forwarded and the app can be orphaned on the DevKit still holding the
    MLA. Launch interactive runs with `ssh -tt` and let these handlers close the
    Run cleanly.
    """

    def __init__(self) -> None:
        self.stop = False
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, self._handle)
            except (AttributeError, ValueError, OSError):
                pass

    def _handle(self, signum, _frame) -> None:
        if not self.stop:
            print(f"\n[signal {signum}] stopping, closing Run…", flush=True)
        self.stop = True


class ProfileWindow:
    def __init__(self, enabled: bool, interval: int) -> None:
        self.enabled = enabled
        self.interval = interval
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.boxes = 0
        self.start_ms = 0.0
        self.pull_ms = 0.0
        self.decode_ms = 0.0
        self.sink_ms = 0.0

    def add(self, pull_ms: float, decode_ms: float, sink_ms: float, box_count: int) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.boxes += box_count
        self.pull_ms += pull_ms
        self.decode_ms += decode_ms
        self.sink_ms += sink_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = time_ms() - self.start_ms
        fps = self.frames * 1000.0 / elapsed if elapsed > 0 else 0.0
        print(
            f"[profile] frames={self.frames} fps={fps:.1f} "
            f"pull={self.pull_ms / self.frames:.1f}ms "
            f"decode={self.decode_ms / self.frames:.1f}ms "
            f"sinks={self.sink_ms / self.frames:.1f}ms "
            f"boxes={self.boxes / self.frames:.1f}",
            flush=True,
        )
        self.reset()


HEARTBEAT_EVERY = 50


def run_pipeline(pipeline: Pipeline, cfg: AppConfig, stopper: Stopper) -> int:
    profile = ProfileWindow(cfg.profile, cfg.profile_interval)
    processed = 0
    timeouts = 0
    heartbeat_start = time_ms()
    heartbeat_boxes = 0
    live_fps = float(pipeline.fps or 25)   # HUD value, refreshed each heartbeat

    while not stopper.stop and (cfg.frames <= 0 or processed < cfg.frames):
        pull_start = time_ms()
        sample = pipeline.run.pull("detector_output", cfg.pull_timeout_ms)
        pull_end = time_ms()

        if sample is None:
            timeouts += 1
            print(
                f"[warn] timed out waiting for detections ({timeouts})",
                file=sys.stderr, flush=True,
            )
            if cfg.source_type == "video":
                print("end of video file or stalled source; stopping", flush=True)
                break
            continue

        boxes = parse_boxes(
            extract_bbox_payload(sample),
            pipeline.frame_w,
            pipeline.frame_h,
            cfg.max_detections,
        )
        frame = frame_to_bgr(first_tensor(joined_field(sample, "frame", 0)))
        decode_end = time_ms()

        # Insight receives the raw frame and draws its own overlay from the
        # metadata stream, so annotate only for the local sinks.
        push_video(pipeline, sample, frame)
        send_metadata(pipeline, sample, boxes)
        processed += 1

        need_jpeg = wants_jpeg(cfg, processed)
        if pipeline.writer is not None or (need_jpeg and cfg.save_overlay):
            annotated = render_annotated(cfg, pipeline, frame, boxes, live_fps)
            if pipeline.writer is not None:
                pipeline.writer.write(annotated)
                pipeline.writer_frames += 1
            if need_jpeg:
                save_frame(cfg, processed, annotated)
        elif need_jpeg:
            save_frame(cfg, processed, frame)
        sink_end = time_ms()

        profile.add(
            pull_end - pull_start, decode_end - pull_end, sink_end - decode_end, len(boxes)
        )

        # Heartbeat, so a healthy run does not look identical to a stalled one.
        heartbeat_boxes += len(boxes)
        if processed % HEARTBEAT_EVERY == 0:
            elapsed = time_ms() - heartbeat_start
            rate = HEARTBEAT_EVERY * 1000.0 / elapsed if elapsed > 0 else 0.0
            live_fps = rate or live_fps
            print(
                f"[{processed}] {rate:.1f} fps, "
                f"{heartbeat_boxes / HEARTBEAT_EVERY:.1f} detections/frame avg",
                flush=True,
            )
            heartbeat_start = time_ms()
            heartbeat_boxes = 0

    profile.flush()
    print(f"processed={processed} timeouts={timeouts}")
    if pipeline.metadata_sender is not None:
        stats = pipeline.metadata_sender.stats()
        print(
            f"metadata: sent={stats.datagrams_sent} failures={stats.send_failures} "
            f"would_block={stats.would_block}"
        )
    return processed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SiMa Neat YOLO object detector")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Parse and validate the config without loading pyneat or the model.",
    )
    args = parser.parse_args(argv)

    pipeline = None
    try:
        cfg = load_app_config(args.config)
        if args.validate_config:
            print(f"config OK: {args.config}")
            print(f"  family={cfg.family} -> BoxDecodeType.{FAMILY_DECODE_TOKENS[cfg.family]}")
            print(f"  {describe_preprocess(cfg, cfg.source_width, cfg.source_height)}")
            return 0

        load_runtime_dependencies()
        if cfg.profile:
            os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
            os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
        if cfg.save_enable:
            Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        stopper = Stopper()
        pipeline = build_pipeline(cfg)
        run_pipeline(pipeline, cfg, stopper)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
