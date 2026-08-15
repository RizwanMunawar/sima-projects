"""Object detection on the SiMa Modalix DevKit.

Reads a video file, RTSP stream or camera, runs a YOLO model on the MLA
accelerator, and publishes results three ways: an annotated video and stills on
the DevKit, plus a live Neat Insight feed over UDP.

Built against the Neat Library public Python API (pyneat) as packaged in
2.1.2_Palette_SDK. Runs on the DevKit, not in the x86 SDK container, because
pyneat is compiled for aarch64.

The pipeline is a Graph rather than a single Model.run call, because it has
multiple stages, named public endpoints and a branch with a fan-in::

    source --> branch --> frame ----------------+
                    |                           +--> combine("detector_output")
                    +---> model --> detections -+

    detector_output --> BBOX parse --> overlay --> video file + stills
                                   +-> MetadataSender  (JSON over UDP)
                                   +-> VideoSender     (H.264 RTP over UDP)

Typical usage::

    python3 src/app.py --config config.yaml
    python3 src/app.py --config config.yaml --validate-config
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import signal
import smtplib
import struct
import subprocess
import sys
import threading
import time
from email.message import EmailMessage
import dataclasses
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

# Imported lazily by load_runtime_dependencies() so --validate-config works off-board.
cv2 = None
np = None
pyneat = None


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing intent, mirroring pyneat.ModelOptions.preprocess.

    This is what the application asks for. The route planner resolves it against
    the model archive's MPK contract and compiles the matching graph, so a value
    here is a request rather than a guarantee.

    Attributes:
        kind: Input kind. One of ``image``, ``tensor`` or ``auto``.
        enable: Master switch. One of ``on``, ``off`` or ``auto``.
        input_format: Colour format the source hands to Preproc, such as
            ``NV12`` for hardware-decoded video or ``BGR`` for cv2 images.
        output_format: Model input colour space, or ``auto`` to take it from the
            model contract.
        input_max_width: Preproc buffer width capacity. 0 uses the probed size.
        input_max_height: Preproc buffer height capacity. 0 uses the probed size.
        resize_enable: Whether to resize. ``auto``, ``on`` or ``off``.
        resize_width: Target width. 0 infers it from the model contract.
        resize_height: Target height. 0 infers it from the model contract.
        resize_mode: One of ``letterbox``, ``stretch`` or ``crop``.
        pad_value: Fill value for letterbox padding. 114 is the YOLO default.
        scaling_type: Interpolation token, for example ``BILINEAR``.
        normalize_enable: Whether to normalize. ``auto``, ``on`` or ``off``.
        normalize_preset: One of ``coco_yolo``, ``imagenet`` or ``none``.
        mean: Per-channel mean, read only when the preset is ``none``.
        stddev: Per-channel divisor, read only when the preset is ``none``.
        quantize_enable: Quantization control. Normally left on ``auto``.
        quantize_zero_point: Explicit zero point, applied only when enabled.
        quantize_scale: Explicit scale. 0.0 uses the model's calibration.
        tessellate_enable: MLA tile-layout control. Normally left on ``auto``.
        tessellate_slice_shape: Tile geometry override. Empty uses the contract.
    """

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
class DrawConfig:
    """Overlay appearance, straight from the ``visualization`` config section.

    Every pixel size here is expressed for a 1080p frame. When ``auto_scale``
    is on they are multiplied by ``min(frame_height, frame_width) /
    reference_height``, so 4K does not get hairlines and 480p does not get
    slabs. Turn ``auto_scale`` off to use the numbers literally.

    Attributes:
        box_thickness: Detection rectangle outline weight, in pixels.
        text_scale: OpenCV font scale for captions.
        text_thickness: Caption stroke weight, in pixels.
        text_padding: Gap between caption text and the edge of its band.
        centre_dot: Whether to mark the centre of each box.
        centre_dot_radius: Radius of that marker, in pixels.
        show_labels: Whether the caption carries the class name.
        show_scores: Whether the caption carries the confidence.
        score_decimals: Decimal places for the confidence, so 2 gives ``0.57``.
        text_color: Caption text colour, BGR.
        hud_text_color: Frame-rate badge text colour, BGR.
        hud_bg_color: Frame-rate badge fill colour, BGR.
        hud_text_scale: Badge font scale. 0 follows ``text_scale``.
        hud_text_thickness: Badge stroke weight. 0 follows ``text_thickness``.
        hud_padding: Gap between badge text and badge edge, on every side. 0
            follows ``text_padding``. This is what sets the badge size when no
            minimum is given.
        hud_padding_x: Left/right gap. 0 follows ``hud_padding``.
        hud_padding_y: Top/bottom gap. 0 follows ``hud_padding``.
        hud_margin_x: Gap between the badge and the left frame edge. 0 follows
            the resolved horizontal padding.
        hud_margin_y: Gap between the badge and the top frame edge. 0 follows
            the resolved vertical padding.
        hud_fps_decimals: Decimal places on the frame rate. 0 gives ``FPS: 25``,
            1 gives ``FPS: 24.8``.
        hud_min_width: Floor on badge width in pixels. 0 fits the text.
        hud_min_height: Floor on badge height in pixels. 0 fits the text.
        show_track_ids: Whether captions carry the track id.
        banner: Whether to draw the full-width alert strip.
        banner_text_scale: Banner font scale. 0 follows ``text_scale``.
        banner_text_thickness: Banner stroke. 0 follows ``text_thickness``.
        banner_padding: Gap between banner text and the strip edge.
        banner_alpha: Strip opacity, 0.0 to 1.0.
        banner_bg_color: Strip fill, BGR.
        banner_text_color: Strip text, BGR.
        auto_scale: Whether sizes scale with frame height.
        reference_height: The frame height the sizes above are tuned for.
    """

    box_thickness: int = 3
    text_scale: float = 1.0
    text_thickness: int = 2
    text_padding: int = 10
    centre_dot: bool = True
    centre_dot_radius: int = 7
    show_labels: bool = True
    show_scores: bool = True
    score_decimals: int = 2
    text_color: tuple[int, int, int] = (255, 255, 255)
    hud_text_color: tuple[int, int, int] = (255, 255, 255)
    hud_bg_color: tuple[int, int, int] = (0, 0, 0)
    hud_text_scale: float = 0.0
    hud_text_thickness: int = 0
    hud_padding: int = 0
    hud_padding_x: int = 0
    hud_padding_y: int = 0
    hud_margin_x: int = 0
    hud_margin_y: int = 0
    hud_fps_decimals: int = 0
    hud_min_width: int = 0
    hud_min_height: int = 0
    show_track_ids: bool = True
    banner: bool = True
    banner_text_scale: float = 0.0
    banner_text_thickness: int = 0
    banner_padding: int = 18
    banner_alpha: float = 0.75
    banner_bg_color: tuple[int, int, int] = (56, 56, 255)
    banner_text_color: tuple[int, int, int] = (255, 255, 255)
    auto_scale: bool = True
    reference_height: float = 1080.0


@dataclass(frozen=True)
class TrackConfig:
    """How detections are followed from frame to frame.

    Attributes:
        classes: Class names or ids that can fall. Empty means every class,
            which is almost never what you want in a warehouse.
        iou_threshold: Minimum overlap to call two boxes the same person.
            Lower it for a low frame rate, where boxes move further per frame.
        max_age: Frames a track survives with no detection before it is dropped.
            This is what carries someone through a brief occlusion.
        min_hits: Frames a track must be seen on before it is reported at all,
            which suppresses one-frame false positives.
        history_seconds: How much per-track history to keep. Must be at least
            ``fall.descent_window``.
        upright_aspect: Aspect ratio at or below which a person counts as
            upright for the purpose of learning their reference height.
    """

    classes: tuple[str, ...] = ("person",)
    iou_threshold: float = 0.30
    max_age: int = 30
    min_hits: int = 3
    history_seconds: float = 3.0
    upright_aspect: float = 0.70


@dataclass(frozen=True)
class FallConfig:
    """When a tracked person counts as fallen.

    Attributes:
        enable: Whether to evaluate falls at all. ``off`` leaves a plain
            person tracker, which is a useful way to tune the tracking first.
        aspect_ratio: Width over height at or above which the box reads as
            lying down. 1.0 is square; 1.2 gives a little margin against a
            crouch or a wide-armed gesture.
        height_drop: Fraction of this person's own learned upright height at or
            below which they count as collapsed. 0.55 means "less than 55% as
            tall as they were".
        descent_rate: Downward speed of the box centre that reads as a fall,
            as a fraction of frame height per second. 0.55 is roughly half the
            frame in one second.
        descent_window: How far back to measure that speed, in seconds.
        confirm_seconds: How long a signal must hold before an alert fires.
            The single most important knob here: too low and a bending forklift
            driver pages someone, too high and the alert is late.
        recover_seconds: How long someone must look upright again before the
            track is re-armed for another alert.
        min_box_height: Ignore boxes shorter than this fraction of the frame,
            which drops distant figures too small to judge.
    """

    enable: bool = True
    aspect_ratio: float = 1.20
    height_drop: float = 0.55
    descent_rate: float = 0.55
    descent_window: float = 0.7
    confirm_seconds: float = 1.5
    recover_seconds: float = 3.0
    min_box_height: float = 0.08


@dataclass(frozen=True)
class AlertConfig:
    """SMTP notification settings.

    There is no password field on purpose. ``config.yaml`` is committed, so a
    password in it is a password on GitHub. The password is read from the
    environment variable named by ``password_env`` instead.

    Attributes:
        enable: Whether to send anything at all.
        dry_run: Compose and log the message without connecting to a server.
            The way to check subject, recipients and attachment wiring.
        host: SMTP server hostname.
        port: SMTP port. 587 for STARTTLS, 465 for implicit SSL, 25 for a relay.
        ssl: Connect with implicit TLS, for port 465.
        starttls: Upgrade a plaintext connection, for port 587.
        username: SMTP login. Empty means an unauthenticated relay.
        password_env: Environment variable holding the password.
        sender: ``From`` address.
        recipients: ``To`` addresses.
        subject: Subject template. ``{label}``, ``{track_id}``, ``{time}`` and
            ``{site}`` are substituted.
        site: A human name for this camera or building, put in the subject and
            the body so an alert says where to go.
        cooldown_seconds: Minimum gap between alerts, across every track. A
            person falling usually produces one event; a camera pointed at a
            busy aisle should not produce forty.
        attach_snapshot: Whether to attach the annotated frame.
        snapshot_dir: Where snapshots are written on the DevKit.
        queue_depth: Pending alerts held for the sender thread. Beyond this,
            alerts are dropped rather than blocking the pipeline.
        timeout: SMTP socket timeout in seconds.
    """

    enable: bool = False
    dry_run: bool = True
    host: str = "smtp.gmail.com"
    port: int = 587
    ssl: bool = False
    starttls: bool = True
    username: str = ""
    password_env: str = "FALL_ALERT_SMTP_PASSWORD"
    sender: str = ""
    recipients: tuple[str, ...] = ()
    subject: str = "[{site}] Fall detected - track #{track_id} at {time}"
    site: str = "Warehouse camera 1"
    cooldown_seconds: float = 60.0
    attach_snapshot: bool = True
    snapshot_dir: str = "alerts"
    queue_depth: int = 8
    timeout: float = 20.0


@dataclass(frozen=True)
class AppConfig:
    """Fully resolved application configuration.

    Built by :func:`load_app_config` from ``config.yaml`` and validated by
    :func:`validate_config` before anything touches the hardware, so a bad
    config fails in under a second instead of part-way through a graph build.

    Attributes:
        model_path: Path to the compiled model archive on the DevKit.
        labels_path: Path to the newline-separated class label file.
        family: Detection head token, mapped by ``FAMILY_DECODE_TOKENS``.
        decode_type_option: Head packing override. Normally ``auto``.
        num_classes: Class count override. 0 reads it from the archive.
        source_type: One of ``video``, ``rtsp`` or ``usb``.
        source_uri: File path or stream URL, as seen on the DevKit.
        source_fps: Frame rate. 0 probes it from the source.
        source_width: Frame width. 0 probes it.
        source_height: Frame height. 0 probes it.
        rtsp_codec: ``h264`` or ``mjpeg``.
        rtsp_tcp: Whether to use TCP transport for RTSP.
        rtsp_latency_ms: Jitter buffer latency.
        usb_camera_name: libcamera device name, or empty for the default.
        usb_format: Pixel format requested from the camera.
        preprocess: Preprocessing intent. See :class:`PreprocessConfig`.
        score_threshold: Minimum detection confidence.
        nms_iou: Non-max suppression IoU threshold.
        max_detections: Top-K cap per frame.
        frames: Frame limit. 0 runs until interrupted.
        pull_timeout_ms: How long to wait for a frame before giving up.
        queue_depth: Runtime queue depth.
        output_buffers: Buffers each public output may hold. Every one of
            them is a frame checked out of the hardware decoder's pool, and
            that pool is small (the boot log prints ``BufferNum=8``). Two
            outputs at 4 is already the whole pool, so a slow consumer
            deadlocks the decoder. Keep the product well under BufferNum.
        run_preset: One of ``realtime``, ``balanced`` or ``reliable``.
        overflow_policy: One of ``keep_latest``, ``block`` or ``drop_incoming``.
        profile: Whether to print per-stage timings.
        profile_interval: Frames per profiling window.
        save_enable: Whether to write annotated stills.
        save_dir: Directory for stills.
        save_every: Write every Nth frame. 0 disables.
        save_overlay: Whether stills carry the overlay.
        save_format: ``jpg`` or ``png``.
        video_enable: Whether to write an annotated video on the DevKit.
        video_path: Output video path.
        video_codec: Four-character FourCC, with an MJPG fallback.
        video_fps: Output frame rate. 0 matches the source.
        video_hud: Whether to draw the summary strip.
        insight_enable: Whether to stream to Neat Insight.
        insight_annotated: Whether Insight receives the annotated frame. False
            sends the raw frame and lets Insight draw its own overlay.
        insight_host: Insight address as the DevKit sees it.
        insight_channel: Channel offset added to both port bases.
        video_port_base: First UDP video port.
        metadata_port_base: First UDP metadata port.
        bitrate_kbps: H.264 encoder bitrate for the Insight feed.
    """

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
    output_buffers: int
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
    insight_annotated: bool
    insight_host: str
    insight_channel: int
    video_port_base: int
    metadata_port_base: int
    bitrate_kbps: int

    track: TrackConfig
    fall: FallConfig
    alerts: AlertConfig
    draw: DrawConfig


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
    """
    Read a tri-state auto/on/off knob. YAML 1.1 resolves 
    bare `on`/`off`/`yes`/`no` to booleans, so `enable: on`
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


def _color(raw: dict, key: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    """Read a ``[B, G, R]`` colour. OpenCV is BGR, so the order is not a typo."""
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"config key `{key}` must be a list of three numbers, [B, G, R]")
    channels = []
    for channel in value:
        if isinstance(channel, bool) or not isinstance(channel, (int, float)):
            raise ValueError(f"config key `{key}` must contain numbers")
        if not 0 <= channel <= 255:
            raise ValueError(f"config key `{key}` channels must be between 0 and 255")
        channels.append(int(channel))
    return (channels[0], channels[1], channels[2])


def _str_list(raw: dict, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a list of names or ids. A bare scalar is accepted as one item."""
    value = raw.get(key)
    if value is None:
        return default
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return (str(value),)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"config key `{key}` must be a list")
    return tuple(str(item) for item in value)


def load_draw_config(raw: dict) -> DrawConfig:
    """Build a :class:`DrawConfig` from the ``visualization`` config section.

    Args:
        raw: The whole parsed config document.

    Returns:
        A populated :class:`DrawConfig`. Every key is optional and falls back to
        the dataclass default, so an absent section is valid.
    """
    section = _section(raw, "visualization")
    hud = _section(section, "hud")
    banner_sec = _section(section, "banner")
    default = DrawConfig()
    return DrawConfig(
        box_thickness=_int(section, "box_thickness", default.box_thickness),
        text_scale=_float(section, "text_scale", default.text_scale),
        text_thickness=_int(section, "text_thickness", default.text_thickness),
        text_padding=_int(section, "text_padding", default.text_padding),
        centre_dot=_flag(section, "centre_dot", "on") == "on",
        centre_dot_radius=_int(section, "centre_dot_radius", default.centre_dot_radius),
        show_labels=_flag(section, "show_labels", "on") == "on",
        show_scores=_flag(section, "show_scores", "on") == "on",
        score_decimals=_int(section, "score_decimals", default.score_decimals),
        text_color=_color(section, "text_color", default.text_color),
        hud_text_color=_color(hud, "text_color", default.hud_text_color),
        hud_bg_color=_color(hud, "bg_color", default.hud_bg_color),
        hud_text_scale=_float(hud, "text_scale", default.hud_text_scale),
        hud_text_thickness=_int(hud, "text_thickness", default.hud_text_thickness),
        hud_padding=_int(hud, "padding", default.hud_padding),
        hud_padding_x=_int(hud, "padding_x", default.hud_padding_x),
        hud_padding_y=_int(hud, "padding_y", default.hud_padding_y),
        hud_margin_x=_int(hud, "margin_x", default.hud_margin_x),
        hud_margin_y=_int(hud, "margin_y", default.hud_margin_y),
        hud_fps_decimals=_int(hud, "fps_decimals", default.hud_fps_decimals),
        hud_min_width=_int(hud, "min_width", default.hud_min_width),
        hud_min_height=_int(hud, "min_height", default.hud_min_height),
        show_track_ids=_flag(section, "show_track_ids", "on") == "on",
        banner=_flag(banner_sec, "enable", "on") == "on",
        banner_text_scale=_float(banner_sec, "text_scale", default.banner_text_scale),
        banner_text_thickness=_int(banner_sec, "text_thickness", default.banner_text_thickness),
        banner_padding=_int(banner_sec, "padding", default.banner_padding),
        banner_alpha=_float(banner_sec, "alpha", default.banner_alpha),
        banner_bg_color=_color(banner_sec, "bg_color", default.banner_bg_color),
        banner_text_color=_color(banner_sec, "text_color", default.banner_text_color),
        auto_scale=_flag(section, "auto_scale", "on") == "on",
        reference_height=_float(section, "reference_height", default.reference_height),
    )


def _triple(raw: dict, key: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"config key `{key}` must be a list of 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def load_track_config(raw: dict) -> TrackConfig:
    section = _section(raw, "tracking")
    d = TrackConfig()
    return TrackConfig(
        classes=_str_list(section, "classes", d.classes),
        iou_threshold=_float(section, "iou_threshold", d.iou_threshold),
        max_age=_int(section, "max_age", d.max_age),
        min_hits=_int(section, "min_hits", d.min_hits),
        history_seconds=_float(section, "history_seconds", d.history_seconds),
        upright_aspect=_float(section, "upright_aspect", d.upright_aspect),
    )


def load_fall_config(raw: dict) -> FallConfig:
    section = _section(raw, "fall")
    d = FallConfig()
    return FallConfig(
        enable=_flag(section, "enable", "on") == "on",
        aspect_ratio=_float(section, "aspect_ratio", d.aspect_ratio),
        height_drop=_float(section, "height_drop", d.height_drop),
        descent_rate=_float(section, "descent_rate", d.descent_rate),
        descent_window=_float(section, "descent_window", d.descent_window),
        confirm_seconds=_float(section, "confirm_seconds", d.confirm_seconds),
        recover_seconds=_float(section, "recover_seconds", d.recover_seconds),
        min_box_height=_float(section, "min_box_height", d.min_box_height),
    )


def load_alert_config(raw: dict) -> AlertConfig:
    section = _section(raw, "alerts")
    smtp = _section(section, "smtp")
    d = AlertConfig()
    return AlertConfig(
        enable=_flag(section, "enable", "off") == "on",
        dry_run=_flag(section, "dry_run", "on") == "on",
        host=_str(smtp, "host", d.host),
        port=_int(smtp, "port", d.port),
        ssl=_flag(smtp, "ssl", "off") == "on",
        starttls=_flag(smtp, "starttls", "on") == "on",
        username=_str(smtp, "username", d.username),
        password_env=_str(smtp, "password_env", d.password_env),
        timeout=_float(smtp, "timeout", d.timeout),
        sender=_str(section, "from", d.sender),
        recipients=_str_list(section, "to", d.recipients),
        subject=_str(section, "subject", d.subject),
        site=_str(section, "site", d.site),
        cooldown_seconds=_float(section, "cooldown_seconds", d.cooldown_seconds),
        attach_snapshot=_flag(section, "attach_snapshot", "on") == "on",
        snapshot_dir=_str(section, "snapshot_dir", d.snapshot_dir),
        queue_depth=_int(section, "queue_depth", d.queue_depth),
    )


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


def resolve_labels_path(value: str, config_path: Path) -> Path:
    """Find the labels file, tolerating where the app was launched from.

    ``model.labels`` is written relative to the app directory, because that is
    where the DevKit launches from. Validating a config off-board is normally
    done from the repo root instead, and failing there would make
    ``--validate-config`` useless in exactly the place it is most useful. So try
    the literal path first, then beside ``config.yaml``, then the copy packaged
    next to this file.

    Args:
        value: The raw ``model.labels`` string.
        config_path: Path to the config file being loaded.

    Returns:
        The first candidate that exists, or the literal path so the caller still
        reports the name the user actually wrote.
    """
    given = Path(value)
    for candidate in (given, config_path.resolve().parent / value):
        if candidate.is_file():
            return candidate
    packaged = Path(__file__).resolve().parent / "coco_labels.txt"
    return packaged if packaged.is_file() else given


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
        labels_path=resolve_labels_path(_str(model, "labels", str(default_labels)), path),
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
        output_buffers=_int(runtime, "output_buffers", 1),
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
        insight_annotated=_bool(insight, "annotated", True),
        insight_host=_str(insight, "host", "127.0.0.1"),
        insight_channel=_int(insight, "channel", 0),
        video_port_base=_int(insight, "video_port_base", 9000),
        metadata_port_base=_int(insight, "metadata_port_base", 9100),
        bitrate_kbps=_int(insight, "bitrate_kbps", 2000),
        track=load_track_config(raw),
        fall=load_fall_config(raw),
        alerts=load_alert_config(raw),
        draw=load_draw_config(raw),
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
    if cfg.output_buffers < 1:
        raise ValueError("runtime.output_buffers must be >= 1")
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
    # Two senders on one port is not a warning-level mistake: the H.264 encoder
    # fails to configure, and because it shares the codec daemon with the
    # decoder feeding the source, the whole pipeline stalls a few frames in.
    # That looks like "the output video is 12 frames long", which is a long way
    # from the actual cause.
    if cfg.insight_enable and cfg.video_port_base == cfg.metadata_port_base:
        raise ValueError(
            f"output.insight.video_port_base and metadata_port_base are both "
            f"{cfg.video_port_base}. They must differ; the defaults are 9000 and 9100.\n"
            f"  Sharing a port wedges the encoder, which stalls the source and "
            f"truncates the recording."
        )
    if cfg.insight_enable and 9900 in (cfg.video_port_base, cfg.metadata_port_base):
        raise ValueError(
            "output.insight port base 9900 is the Neat Insight web UI port, not a "
            "stream port.\n  Use video_port_base: 9000 and metadata_port_base: 9100."
        )
    if not 0.0 <= cfg.track.iou_threshold <= 1.0:
        raise ValueError("tracking.iou_threshold must be in [0.0, 1.0]")
    if cfg.track.max_age < 0:
        raise ValueError("tracking.max_age must be >= 0")
    if cfg.track.min_hits < 1:
        raise ValueError("tracking.min_hits must be >= 1")
    if cfg.track.history_seconds < cfg.fall.descent_window:
        raise ValueError(
            f"tracking.history_seconds ({cfg.track.history_seconds}) is shorter than "
            f"fall.descent_window ({cfg.fall.descent_window}), so the descent test "
            f"would never see far enough back to fire."
        )
    if cfg.fall.aspect_ratio <= 0:
        raise ValueError("fall.aspect_ratio must be > 0")
    if not 0.0 < cfg.fall.height_drop <= 1.0:
        raise ValueError("fall.height_drop must be in (0.0, 1.0]")
    if cfg.fall.descent_rate < 0:
        raise ValueError("fall.descent_rate must be >= 0")
    if cfg.fall.descent_window <= 0:
        raise ValueError("fall.descent_window must be > 0")
    if cfg.fall.confirm_seconds < 0:
        raise ValueError("fall.confirm_seconds must be >= 0")
    if cfg.fall.recover_seconds < 0:
        raise ValueError("fall.recover_seconds must be >= 0")
    if not 0.0 <= cfg.fall.min_box_height < 1.0:
        raise ValueError("fall.min_box_height must be in [0.0, 1.0)")
    if cfg.alerts.enable and not cfg.alerts.dry_run:
        if not cfg.alerts.host:
            raise ValueError("alerts.smtp.host must be set when alerts are enabled")
        if not cfg.alerts.sender:
            raise ValueError("alerts.from must be set when alerts are enabled")
        if not cfg.alerts.recipients:
            raise ValueError("alerts.to must list at least one recipient")
        if cfg.alerts.ssl and cfg.alerts.starttls:
            raise ValueError(
                "alerts.smtp.ssl and starttls are both on. Use ssl for port 465 "
                "or starttls for port 587, not both."
            )
        if not cfg.alerts.password_env:
            raise ValueError("alerts.smtp.password_env must name an environment variable")
    if cfg.alerts.queue_depth < 1:
        raise ValueError("alerts.queue_depth must be >= 1")
    if cfg.alerts.cooldown_seconds < 0:
        raise ValueError("alerts.cooldown_seconds must be >= 0")
    if not (cfg.save_enable or cfg.insight_enable or cfg.video_enable):
        raise ValueError(
            "enable at least one of output.save, output.video or output.insight"
        )


# Config token -> pyneat enum mapping, Kept as string tokens so the tables 
# can be validated before pyneat is imported (the wheel is aarch64-only 
# and does not load in the SDK container). model.family -> BoxDecodeType 
# attribute name. `yolo11` intentionally maps to YoloV8: BoxDecodeType has 
# no YOLO11 member, and Ultralytics YOLO11 exports the same decoupled DFL detect 
# head as YOLOv8.
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


def apply_preprocess_options(opt, pre: PreprocessConfig, frame_w: int, frame_h: int) -> None:
    """
    Translate the config's preprocess block onto pyneat ModelOptions.

    ``Model`` resolves this against the archive's MPK contract and builds the
    matching Preproc, Quant, Tess or QuantTess graph family. Anything left on
    ``auto`` is decided by the planner rather than here.

    Args:
        opt: A ``pyneat.ModelOptions`` to populate in place.
        pre: Preprocessing intent from the config file.
        frame_w: Probed source width, used when no capacity is configured.
        frame_h: Probed source height, used when no capacity is configured.
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


class BitReader:
    """Minimal MSB-first bit reader with Exp-Golomb support, for SPS parsing."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def u(self, n: int) -> int:
        value = 0
        for _ in range(n):
            byte = self.pos >> 3
            if byte >= len(self.data):
                raise ValueError("SPS truncated")
            bit = (self.data[byte] >> (7 - (self.pos & 7))) & 1
            value = (value << 1) | bit
            self.pos += 1
        return value

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 32:
                raise ValueError("SPS Exp-Golomb overrun")
        return (1 << zeros) - 1 + (self.u(zeros) if zeros else 0)

    def se(self) -> int:
        k = self.ue()
        return (k + 1) // 2 if k % 2 else -(k // 2)


def unescape_rbsp(data: bytes) -> bytes:
    """Strip H.264 emulation prevention bytes (00 00 03 -> 00 00)."""
    out = bytearray()
    zeros = 0
    for byte in data:
        if zeros >= 2 and byte == 0x03:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0x00 else 0
    return bytes(out)


# Profiles whose SPS carries the chroma_format_idc block.
HIGH_PROFILES = {100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135}


def skip_scaling_list(reader: BitReader, size: int) -> None:
    last = next_scale = 8
    for _ in range(size):
        if next_scale:
            next_scale = (last + reader.se() + 256) % 256
        last = last if next_scale == 0 else next_scale


def parse_sps(rbsp: bytes) -> tuple[int, int, int]:
    """Decode width, height and frame rate from one SPS payload.

    Args:
        rbsp: SPS NAL payload with the header byte removed and emulation
            prevention bytes already stripped.

    Returns:
        A ``(width, height, fps)`` triple. ``fps`` is 0 when the stream carries
        no VUI timing information, which is common for camera captures and for
        anything remuxed with ``-c:v copy``.
    """
    r = BitReader(rbsp)
    profile_idc = r.u(8)
    r.u(8)  # constraint flags and reserved bits
    r.u(8)  # level_idc
    r.ue()  # seq_parameter_set_id

    chroma_format_idc = 1
    separate_colour_plane = 0
    if profile_idc in HIGH_PROFILES:
        chroma_format_idc = r.ue()
        if chroma_format_idc == 3:
            separate_colour_plane = r.u(1)
        r.ue()  # bit_depth_luma_minus8
        r.ue()  # bit_depth_chroma_minus8
        r.u(1)  # qpprime_y_zero_transform_bypass_flag
        if r.u(1):  # seq_scaling_matrix_present_flag
            for i in range(8 if chroma_format_idc != 3 else 12):
                if r.u(1):
                    skip_scaling_list(r, 16 if i < 6 else 64)

    r.ue()  # log2_max_frame_num_minus4
    pic_order_cnt_type = r.ue()
    if pic_order_cnt_type == 0:
        r.ue()  # log2_max_pic_order_cnt_lsb_minus4
    elif pic_order_cnt_type == 1:
        r.u(1)  # delta_pic_order_always_zero_flag
        r.se()  # offset_for_non_ref_pic
        r.se()  # offset_for_top_to_bottom_field
        for _ in range(r.ue()):
            r.se()  # offset_for_ref_frame[i]

    r.ue()  # max_num_ref_frames
    r.u(1)  # gaps_in_frame_num_value_allowed_flag
    width_mbs = r.ue() + 1
    height_map_units = r.ue() + 1
    frame_mbs_only = r.u(1)
    if not frame_mbs_only:
        r.u(1)  # mb_adaptive_frame_field_flag
    r.u(1)  # direct_8x8_inference_flag

    crop_left = crop_right = crop_top = crop_bottom = 0
    if r.u(1):  # frame_cropping_flag
        crop_left, crop_right = r.ue(), r.ue()
        crop_top, crop_bottom = r.ue(), r.ue()

    width = width_mbs * 16
    height = (2 - frame_mbs_only) * height_map_units * 16
    if chroma_format_idc == 0 or separate_colour_plane:
        unit_x, unit_y = 1, 2 - frame_mbs_only
    else:
        sub_w, sub_h = {1: (2, 2), 2: (2, 1), 3: (1, 1)}.get(chroma_format_idc, (2, 2))
        unit_x, unit_y = sub_w, sub_h * (2 - frame_mbs_only)
    width -= unit_x * (crop_left + crop_right)
    height -= unit_y * (crop_top + crop_bottom)

    fps = 0
    if r.u(1):  # vui_parameters_present_flag
        try:
            fps = parse_vui_fps(r)
        except ValueError:
            fps = 0
    return width, height, fps


def parse_vui_fps(r: BitReader) -> int:
    """Read timing_info out of a VUI block. Returns 0 when it is absent."""
    if r.u(1):  # aspect_ratio_info_present_flag
        if r.u(8) == 255:  # Extended_SAR
            r.u(16)
            r.u(16)
    if r.u(1):  # overscan_info_present_flag
        r.u(1)
    if r.u(1):  # video_signal_type_present_flag
        r.u(3)
        r.u(1)
        if r.u(1):  # colour_description_present_flag
            r.u(24)
    if r.u(1):  # chroma_loc_info_present_flag
        r.ue()
        r.ue()
    if not r.u(1):  # timing_info_present_flag
        return 0
    num_units_in_tick = r.u(32)
    time_scale = r.u(32)
    if num_units_in_tick <= 0 or time_scale <= 0:
        return 0
    return int(round(time_scale / (2.0 * num_units_in_tick)))


def probe_h264_sps(path: str, scan_bytes: int = 4 << 20) -> tuple[int, int, int]:
    """Read geometry straight out of the first SPS in an Annex-B stream.

    ``ffprobe`` and OpenCV both routinely fail on raw elementary streams, which
    is what pushes people into hand-writing ``source.width`` and ``source.height``
    in the config. A wrong value there is silent and fatal: the source caps
    filter stops negotiating and the run reports zero frames after a pull
    timeout. The bytes on disk are authoritative, so read them.

    Args:
        path: Path to a raw Annex-B H.264 file.
        scan_bytes: How much of the head of the file to search.

    Returns:
        A ``(width, height, fps)`` triple, or zeros if no SPS was found.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(scan_bytes)
    except OSError:
        return 0, 0, 0

    pos = 0
    while True:
        idx = head.find(b"\x00\x00\x01", pos)
        if idx < 0:
            return 0, 0, 0
        start = idx + 3
        if start >= len(head):
            return 0, 0, 0
        if head[start] & 0x1F == 7:  # nal_unit_type 7 == SPS
            end = head.find(b"\x00\x00\x01", start)
            payload = head[start + 1 : end if end > 0 else len(head)]
            try:
                return parse_sps(unescape_rbsp(payload))
            except (ValueError, IndexError):
                return 0, 0, 0
        pos = start


def resolve_source_geometry(cfg: AppConfig) -> tuple[int, int, int]:
    """Return (width, height, fps). Config values win; anything left at 0 is probed."""
    width, height, fps = cfg.source_width, cfg.source_height, cfg.source_fps

    if cfg.source_type == "video" and is_elementary_h264(cfg.source_uri):
        sps_w, sps_h, sps_fps = probe_h264_sps(cfg.source_uri)
        if sps_w > 0 and sps_h > 0:
            if (width > 0 and width != sps_w) or (height > 0 and height != sps_h):
                print(
                    f"[warn] config says {width}x{height} but the stream's SPS says "
                    f"{sps_w}x{sps_h}. Using the stream.\n"
                    f"       Fix source.width and source.height in config.yaml, or set "
                    f"them to 0 to always read the stream."
                )
            width, height = sps_w, sps_h
        if sps_fps > 0:
            if fps > 0 and fps != sps_fps:
                print(
                    f"[warn] config says {fps} fps but the stream is {sps_fps} fps. "
                    f"Using the stream.\n"
                    f"       Set source.fps to 0 in config.yaml to always read the stream."
                )
            fps = sps_fps
        if fps <= 0:
            fps = 25
            print("[warn] stream carries no frame rate, assuming 25. Set source.fps to override.")
        return width, height, fps

    if cfg.source_type == "usb":
        # libcamera is queried at build time, not probeable here, so fall back to
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
        # Elementary H.264 never reaches here: that path reads the SPS and
        # returns above.
        hint = (
            "\nNeither ffprobe nor OpenCV could read this source. Set the values "
            "explicitly:\n  source:\n    width: 1920\n    height: 1080\n    fps: 25"
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
    """Build a file source chain without a demuxer.

    This is ``VideoInputGroup`` rebuilt by hand to work around a Neat 0.3.0 bug.
    ``VideoTrackSelect`` emits ``qtdemux name=<base> <base>.video_0``, which is
    internally consistent, but the graph then appends an instance suffix to
    element *names* only. The declaration becomes ``name=n1_demux_8`` while the
    pad reference stays ``n1_demux.video_0``, so gst_parse_launch fails with
    ``No src-element named "n1_demux"``. ``element_names()`` reports only the one
    name, so the renamer never learns to rewrite the pad reference, and any
    non-empty suffix triggers it. Reordering graph construction does not help.

    Dropping the container removes the demuxer, and with it the bug::

        ffmpeg -i input.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 output.h264

    Args:
        cfg: Application configuration, for ``source_uri``.
        width: Unused. Geometry comes from the stream, see the caps note below.
        height: Unused.
        fps: Unused.

    Returns:
        A ``pyneat.Graph`` of FileInput, H264Parse, Queue, SimaDecode and a
        format-only CapsRaw, producing NV12 frames.
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

    # Format only. CapsRawNode emits a field for every argument greater than
    # zero, so passing width, height and fps here would add
    # `width=W,height=H,framerate=F/1` to the capsfilter. A raw elementary
    # stream has no container to state its frame rate, so h264parse publishes
    # `framerate=0/1`, which cannot intersect with any fixed rate. Negotiation
    # then fails upstream of the appsink and the run reports zero frames after
    # a pull timeout, with nothing on the bus to explain it. The decoder already
    # emits NV12 in system memory at the stream's own geometry, so there is
    # nothing left to constrain.
    #
    # nodes.caps_raw takes the format as a plain string, unlike the *Options
    # `format` properties which accept the pyneat.Format enum.
    graph.add(pyneat.nodes.caps_raw("NV12", -1, -1, -1, pyneat.CapsMemory.Any))
    return graph


def check_source_file(cfg: AppConfig) -> None:
    """Fail fast when a file source is missing, empty or a container.

    ``filesrc`` reports a missing file on the GStreamer bus rather than raising,
    so without this the run looks healthy right up to a 20 second pull timeout
    reporting zero frames, which is indistinguishable from a stall.

    Args:
        cfg: Application configuration.

    Raises:
        RuntimeError: If the file is missing or empty.
    """
    if cfg.source_type != "video":
        return

    path = Path(cfg.source_uri)
    if not path.exists():
        listing = ""
        parent = path.parent
        for candidate in (parent, Path("assets/video"), Path("assets/videos")):
            if candidate.is_dir():
                names = sorted(p.name for p in candidate.iterdir() if p.is_file())
                if names:
                    listing += f"\n  {candidate}/ contains: {', '.join(names[:8])}"
        raise RuntimeError(
            f"source file not found: {path}\n"
            f"  looked in: {path.resolve().parent}\n"
            f"  launched from: {Path.cwd()}"
            f"{listing}\n"
            "source.uri is relative to the directory you launch app.py from."
        )

    size = path.stat().st_size
    if size == 0:
        raise RuntimeError(f"source file is empty: {path}")

    if is_elementary_h264(cfg.source_uri):
        head = path.open("rb").read(12)
        # Annex-B streams open with a 3 or 4 byte start code. An MP4 carries
        # "ftyp" at offset 4, which is what a rename rather than a convert looks
        # like, and h264parse would simply never produce a frame.
        annex_b = head.startswith(b"\x00\x00\x00\x01") or head.startswith(b"\x00\x00\x01")
        if not annex_b:
            hint = (
                " That looks like an MP4 container renamed to .h264."
                if b"ftyp" in head
                else ""
            )
            raise RuntimeError(
                f"{path} is not a raw H.264 elementary stream.{hint}\n"
                f"  first bytes: {head[:8].hex(' ')}\n"
                "Convert rather than rename:\n"
                "  ffmpeg -i clip.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 clip.h264"
            )

    print(f"source file: {path} ({size / 1e6:.1f} MB)", flush=True)


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

    # usb / on-board camera, libcamera-backed. Confirm the device is visible
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
    """Everything a running pipeline owns, so teardown has one place to look.

    Attributes:
        model: The loaded ``pyneat.Model``.
        graph: The detector ``Graph``.
        run: Live ``Run`` handle for the detector graph.
        video_graph: Separate graph that encodes frames for Insight.
        video_run: Live ``Run`` handle for the video graph.
        metadata_sender: ``MetadataSender`` publishing detections as JSON.
        labels: Class names, indexed by class id.
        frame_w: Source frame width in pixels.
        frame_h: Source frame height in pixels.
        fps: Source frame rate.
        video_port: Resolved UDP port for the Insight video feed.
        writer: OpenCV ``VideoWriter`` for the on-device recording.
        writer_path: Path the writer actually opened, after any fallback.
        writer_frames: Frames written so far.
        video_dropped: Preview frames the Insight feed refused.
        tracker: The :class:`Tracker` following people across frames.
        alerts: The :class:`AlertSender` owning the SMTP thread.
        fall_class_ids: Class ids that can fall, or None for every class.
        falls: Confirmed falls so far this run.
    """

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
    video_dropped: int = 0
    tracker: object = None
    alerts: object = None
    fall_class_ids: object = None
    falls: int = 0

    def close(self) -> None:
        if self.alerts is not None:
            # Drain first: an alert queued microseconds before Ctrl-C is still
            # a fall that happened, and the thread is a daemon so nothing else
            # would wait for it.
            self.alerts.close()
            stats = self.alerts
            if stats.sent or stats.failed or stats.dropped or stats.suppressed:
                print(
                    f"alerts: sent={stats.sent} failed={stats.failed} "
                    f"dropped={stats.dropped} suppressed_by_cooldown={stats.suppressed}",
                    flush=True,
                )
            self.alerts = None
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


def resolve_fall_classes(cfg: AppConfig, labels: list[str]):
    """Turn ``tracking.classes`` into a set of class ids.

    Accepts names as they appear in the labels file or bare ids. An unknown
    name is a config error rather than a silently empty filter: the alternative
    is an app that tracks nothing and never explains why.
    """
    if not cfg.track.classes:
        return None
    lookup = {name.lower(): i for i, name in enumerate(labels)}
    ids = set()
    for token in cfg.track.classes:
        text = token.strip()
        if text.isdigit():
            index = int(text)
            if not 0 <= index < len(labels):
                raise ValueError(
                    f"tracking.classes has class id {index}, but the labels file "
                    f"only has {len(labels)} classes (0 to {len(labels) - 1})"
                )
            ids.add(index)
            continue
        index = lookup.get(text.lower())
        if index is None:
            near = [n for n in labels if text.lower() in n.lower()][:5]
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            raise ValueError(
                f"tracking.classes has unknown class {text!r}, which is not in "
                f"{cfg.labels_path}.{hint}"
            )
        ids.add(index)
    return ids


def resolve_flow_control(cfg: AppConfig) -> tuple[str, str]:
    """Resolve ``auto`` flow-control settings to concrete tokens.

    A file and a live camera want opposite things, so ``auto`` splits on source
    type:

    * **File** -> ``reliable`` + ``block``. A file has no deadline. Every frame
      matters, and the source can be made to wait.
    * **RTSP or USB** -> ``realtime`` + ``keep_latest``. A camera has no pause
      button, so blocking only buys unbounded latency. Staying current is worth
      more than completeness.

    Getting this wrong on a file is what shortens the recording. ``keep_latest``
    makes the runtime discard buffers whenever inference falls behind the
    decoder, and inference is much slower than decode. Only the survivors reach
    the writer, which then stamps them at the source rate, so the output plays
    fast and ends early: a 15 second clip processed at a quarter of realtime
    becomes a 4 second video. ``block`` instead lets backpressure reach
    ``filesrc``, so decoding slows to the speed of inference and every frame
    survives. The run takes longer than the clip. That is correct, not a stall.

    Args:
        cfg: Application configuration.

    Returns:
        A ``(preset, overflow_policy)`` pair of config tokens.
    """
    live = cfg.source_type in {"rtsp", "usb"}
    preset = cfg.run_preset
    policy = cfg.overflow_policy
    if preset == "auto":
        preset = "realtime" if live else "reliable"
    if policy == "auto":
        policy = "keep_latest" if live else "block"
    return preset, policy


def make_run_options(cfg: AppConfig):
    """Build RunOptions, resolving any ``auto`` flow-control settings.

    Args:
        cfg: Application configuration.

    Returns:
        A populated ``pyneat.RunOptions``.
    """
    preset, policy = resolve_flow_control(cfg)
    run_options = pyneat.RunOptions()
    run_options.preset = enum_value(pyneat.RunPreset, preset, RUN_PRESETS, "runtime.preset")
    run_options.queue_depth = cfg.queue_depth
    run_options.overflow_policy = enum_value(
        pyneat.OverflowPolicy, policy, OVERFLOW_POLICIES, "runtime.overflow_policy"
    )
    # Block is not honoured unconditionally. The runtime quietly rewrites it to
    # KeepLatest when the public output is zero-copy and carries no explicit
    # OutputOptions, so asking for every frame while also asking for zero-copy
    # gets you neither. Auto lets the preset choose, and `reliable` chooses
    # owned buffers, which keeps Block meaning Block. Live sources drop by
    # design, so they keep zero-copy and its lower latency.
    run_options.output_memory = (
        pyneat.OutputMemory.Auto if policy == "block" else pyneat.OutputMemory.ZeroCopy
    )
    return run_options


def build_detector_graph(cfg: AppConfig, model, width: int, height: int, fps: int):
    """source -> branch -> {frame, model->detections} -> combine("detector_output")."""
    source = make_source_graph(cfg, width, height, fps)
    branch = pyneat.graphs.branch("source", ["frame", "model"])

    frame_graph = pyneat.Graph("frame")
    frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(cfg.output_buffers)))

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)

    detections_graph = pyneat.Graph("detections")
    detections_graph.add(
        pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(cfg.output_buffers))
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
    # Never block the caller. The default appsrc is block=true, so once the
    # encoder or UDP egress falls behind, push() stops returning, the run loop
    # never reaches its next pull(), and the whole detector graph stalls. The
    # Insight feed is best effort, so a dropped preview frame is the right
    # trade for keeping inference and the recording running.
    input_options.block = False

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


def describe_fall(cfg: AppConfig, labels: list[str]) -> str:
    watched = ", ".join(cfg.track.classes) if cfg.track.classes else "every class"
    if not cfg.fall.enable:
        return f"fall: off, tracking {watched} only"
    return (
        f"fall: watching {watched} | aspect>={cfg.fall.aspect_ratio} "
        f"height<={cfg.fall.height_drop:.0%} descent>={cfg.fall.descent_rate:.0%}/s "
        f"| confirm={cfg.fall.confirm_seconds}s recover={cfg.fall.recover_seconds}s"
    )


def describe_alerts(cfg: AppConfig) -> str:
    a = cfg.alerts
    if not a.enable:
        return "alerts: off"
    if a.dry_run:
        return (
            f"alerts: DRY RUN, nothing is sent | would mail {len(a.recipients)} "
            f"recipient(s) | cooldown={a.cooldown_seconds}s"
        )
    mode = "ssl" if a.ssl else ("starttls" if a.starttls else "plain")
    return (
        f"alerts: {a.host}:{a.port} {mode} as {a.username or '<anonymous>'} "
        f"-> {', '.join(a.recipients)} | cooldown={a.cooldown_seconds}s "
        f"snapshot={'on' if a.attach_snapshot else 'off'}"
    )


def build_pipeline(cfg: AppConfig) -> Pipeline:
    step = lambda msg: print(msg, flush=True)

    check_source_file(cfg)
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

    step(describe_fall(cfg, labels))
    step(describe_alerts(cfg))

    preset, policy = resolve_flow_control(cfg)
    step(f"runtime: preset={preset} overflow={policy} queue_depth={cfg.queue_depth} "
         f"output_buffers={cfg.output_buffers} (2 outputs -> "
         f"{2 * cfg.output_buffers} decoder buffers in flight)")
    if policy == "block":
        step(
            "       block keeps every frame, so the run takes longer than the clip.\n"
            "       Output length matches the input. This is the right mode for a file."
        )
    elif cfg.source_type == "video":
        step(
            "[warn] overflow_policy drops frames, so the recording will be shorter\n"
            "       than the input and will play fast. Use auto for a file source."
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
        tracker=Tracker(cfg.track),
        alerts=AlertSender(cfg.alerts),
        fall_class_ids=resolve_fall_classes(cfg, labels),
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
            "the route did not include BoxDecode. Check model.family and the model archive."
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
# Tracking
#
# A fall is a change over time, so a per-frame detection is not enough: the same
# person has to be recognisable from one frame to the next before "they were
# upright and now they are not" means anything. This is greedy IoU association,
# which is the cheapest thing that works when people are walking rather than
# crossing at speed.
# ─────────────────────────────────────────────────────────────────────────────


UPRIGHT, FALLING, FALLEN, RECOVERING = "upright", "falling", "fallen", "recovering"


@dataclass
class Track:
    """One person followed across frames, with just enough history to judge them.

    Attributes:
        track_id: Stable id, assigned once and never reused within a run.
        box: Most recent detection, in source-image pixels.
        hits: How many frames this track has been matched on.
        misses: Consecutive frames with no match. Drops the track past a limit.
        history: Recent ``(timestamp_s, centre_y, height, aspect)`` samples,
            oldest first, trimmed to the window the fall rules need.
        upright_height: Rolling reference height from frames where the person
            looked upright. A fallen box is short as well as wide, but only
            relative to how tall *that* person was, which is why this is
            per-track rather than a constant.
        state: One of ``upright``, ``falling``, ``fallen`` or ``recovering``.
        state_since: Timestamp the current state began.
        alerted_at: When an alert was last raised for this track, or 0.0.
    """

    track_id: int
    box: dict
    # 0, not 1: the frame that creates a track also runs it through _advance,
    # so counting it here too would make min_hits mean one frame fewer than it
    # says.
    hits: int = 0
    misses: int = 0
    history: list = field(default_factory=list)
    upright_height: float = 0.0
    state: str = UPRIGHT
    state_since: float = 0.0
    alerted_at: float = 0.0

    @property
    def width(self) -> float:
        return max(1.0, self.box["x2"] - self.box["x1"])

    @property
    def height(self) -> float:
        return max(1.0, self.box["y2"] - self.box["y1"])

    @property
    def aspect(self) -> float:
        """Width over height. Standing is well under 1; lying down is over it."""
        return self.width / self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.box["x1"] + self.box["x2"]) / 2.0,
                (self.box["y1"] + self.box["y2"]) / 2.0)


def box_iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    """Greedy IoU tracker with stable ids.

    Deliberately simple. It has no motion model, so it will swap ids when two
    people cross while overlapping heavily. That costs a duplicate alert at
    worst, which is the right way round for a safety feature: a Kalman filter
    would be more correct and considerably more to get wrong.

    Attributes:
        cfg: Tracking settings.
        tracks: Live tracks.
    """

    def __init__(self, cfg: "TrackConfig") -> None:
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: list[dict], now: float) -> list[Track]:
        """Associate this frame's boxes with existing tracks.

        Args:
            boxes: Detections for this frame, already filtered to the classes
                that count as people.
            now: Monotonic timestamp in seconds.

        Returns:
            Every live track seen often enough to be trusted.
        """
        pairs = sorted(
            (
                (box_iou(t.box, b), ti, bi)
                for ti, t in enumerate(self.tracks)
                for bi, b in enumerate(boxes)
            ),
            key=lambda p: p[0],
            reverse=True,
        )
        used_t: set[int] = set()
        used_b: set[int] = set()
        for score, ti, bi in pairs:
            if score < self.cfg.iou_threshold:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            self._advance(self.tracks[ti], boxes[bi], now)

        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.misses += 1

        for bi, box in enumerate(boxes):
            if bi not in used_b:
                track = Track(track_id=self._next_id, box=box, state_since=now)
                self._next_id += 1
                self._advance(track, box, now)
                self.tracks.append(track)

        self.tracks = [t for t in self.tracks if t.misses <= self.cfg.max_age]
        return [t for t in self.tracks if t.hits >= self.cfg.min_hits and t.misses == 0]

    def _advance(self, track: Track, box: dict, now: float) -> None:
        track.box = box
        track.misses = 0
        track.hits += 1
        _, cy = track.centre
        track.history.append((now, cy, track.height, track.aspect))
        # Keep only what the rules can still look at, so a long run does not
        # accumulate a per-track list the length of the video.
        cutoff = now - self.cfg.history_seconds
        while len(track.history) > 2 and track.history[0][0] < cutoff:
            track.history.pop(0)
        if track.aspect <= self.cfg.upright_aspect:
            # An exponential rather than a max: someone briefly clipped by the
            # frame edge should not raise the bar for the rest of the run.
            track.upright_height = (
                track.height
                if track.upright_height <= 0
                else 0.9 * track.upright_height + 0.1 * track.height
            )


# ─────────────────────────────────────────────────────────────────────────────
# Fall rules
#
# Three signals, all available from a plain bounding box:
#
#   aspect    a standing person is tall and narrow, a fallen one is wide and
#             short. The box aspect ratio crossing 1.0 is the strongest single
#             indicator there is without pose keypoints.
#   collapse  the box height drops well below what this person's own upright
#             height has been, which separates lying down from crouching.
#   descent   the centre of the box moves down fast. This is what distinguishes
#             a fall from someone lying down deliberately.
#
# Any one of them can fire spuriously for a frame or two, so nothing is reported
# until the condition has held for `confirm_seconds`. That delay is the whole
# difference between a useful alert and one nobody reads.
# ─────────────────────────────────────────────────────────────────────────────


def descent_rate(track: Track, window: float) -> float:
    """Downward speed of the box centre, in pixels per second.

    Args:
        track: The track to measure.
        window: How far back to look, in seconds.

    Returns:
        Pixels per second, positive downwards. 0.0 when there is too little
        history to say.
    """
    if len(track.history) < 2:
        return 0.0
    now = track.history[-1][0]
    oldest = track.history[0]
    for sample in track.history:
        if now - sample[0] <= window:
            oldest = sample
            break
    dt = now - oldest[0]
    if dt <= 1e-3:
        return 0.0
    return (track.history[-1][1] - oldest[1]) / dt


def fall_signals(track: Track, fall: "FallConfig", frame_h: int) -> dict:
    """Evaluate all three signals for one track, for judging and for reporting."""
    collapsed = (
        track.upright_height > 0
        and track.height <= track.upright_height * fall.height_drop
    )
    rate = descent_rate(track, fall.descent_window)
    return {
        "aspect": track.aspect >= fall.aspect_ratio,
        "collapse": bool(collapsed),
        "descent": rate >= fall.descent_rate * frame_h,
        "aspect_value": round(track.aspect, 2),
        "descent_value": round(rate, 1),
    }


def looks_fallen(track: Track, fall: "FallConfig", frame_h: int) -> bool:
    """Whether this track currently satisfies any of the fall signals."""
    s = fall_signals(track, fall, frame_h)
    return s["aspect"] or s["collapse"] or s["descent"]


def update_fall_states(
    tracks: list[Track], fall: "FallConfig", frame_h: int, now: float
) -> list[Track]:
    """Advance every track's state machine and return the ones that just fell.

    The machine is::

        upright ──looks_fallen──> falling ──held confirm_seconds──> fallen
           ^                         |                                 |
           |                     recovered                         recovered
           |                         v                                 v
           └──── held recover_seconds ──── recovering <────────────────┘

    Args:
        tracks: Live, confirmed tracks.
        fall: Fall rule settings.
        frame_h: Frame height, so descent thresholds stay resolution independent.
        now: Monotonic timestamp in seconds.

    Returns:
        Tracks that transitioned into ``fallen`` on this frame.
    """
    newly = []
    for track in tracks:
        down = looks_fallen(track, fall, frame_h)
        if track.state in (UPRIGHT, RECOVERING):
            if down:
                track.state, track.state_since = FALLING, now
            elif (
                track.state == RECOVERING
                and now - track.state_since >= fall.recover_seconds
            ):
                track.state, track.state_since = UPRIGHT, now
        elif track.state == FALLING:
            if not down:
                track.state, track.state_since = RECOVERING, now
            elif now - track.state_since >= fall.confirm_seconds:
                track.state, track.state_since = FALLEN, now
                newly.append(track)
        elif track.state == FALLEN:
            if not down:
                track.state, track.state_since = RECOVERING, now
    return newly


# ─────────────────────────────────────────────────────────────────────────────
# SMTP alerts
#
# Sending mail takes anything from tens of milliseconds to a TCP timeout, and
# the run loop cannot afford either. Alerts are handed to a background thread
# through a bounded queue: if the queue is full the alert is dropped and
# counted, which is the correct trade. A stalled pipeline stops detecting the
# next fall, and that is worse than missing one notification.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Alert:
    """One pending notification."""

    track_id: int
    when: float
    label: str
    box: tuple[int, int, int, int]
    signals: dict
    frame_index: int
    snapshot_path: str = ""


class AlertSender:
    """Queues fall alerts and sends them as email from a background thread.

    Attributes:
        cfg: Alert settings.
        sent: Successful sends.
        failed: Sends that raised.
        dropped: Alerts discarded because the queue was full.
        suppressed: Alerts skipped by the cooldown.
    """

    def __init__(self, cfg: "AlertConfig") -> None:
        self.cfg = cfg
        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self.suppressed = 0
        self._queue: "queue.Queue[Alert | None]" = queue.Queue(maxsize=cfg.queue_depth)
        self._last_global = 0.0
        self._thread = None
        if cfg.enable:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    # ── producer side, called from the run loop ──
    def offer(self, alert: Alert, now: float) -> bool:
        """Queue an alert unless the cooldown or a full queue says otherwise."""
        if not self.cfg.enable:
            return False
        if now - self._last_global < self.cfg.cooldown_seconds:
            self.suppressed += 1
            return False
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            self.dropped += 1
            return False
        self._last_global = now
        return True

    def close(self, timeout: float = 10.0) -> None:
        """Drain the queue and stop the worker."""
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout)

    # ── consumer side ──
    def _worker(self) -> None:
        while True:
            alert = self._queue.get()
            if alert is None:
                return
            try:
                self.send_now(alert)
                self.sent += 1
            except Exception as exc:  # pragma: no cover - depends on the network
                self.failed += 1
                print(f"[warn] alert send failed: {exc}", file=sys.stderr, flush=True)

    def build_message(self, alert: Alert):
        """Compose the email, attaching the snapshot when there is one."""
        msg = EmailMessage()
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        msg["Subject"] = self.cfg.subject.format(
            label=alert.label, track_id=alert.track_id, time=when, site=self.cfg.site
        )
        msg["From"] = self.cfg.sender
        msg["To"] = ", ".join(self.cfg.recipients)
        x1, y1, x2, y2 = alert.box
        msg.set_content(
            f"A fall was detected.\n\n"
            f"  site      : {self.cfg.site}\n"
            f"  time      : {when}\n"
            f"  track     : #{alert.track_id} ({alert.label})\n"
            f"  frame     : {alert.frame_index}\n"
            f"  box       : x={x1} y={y1} w={x2 - x1} h={y2 - y1}\n"
            f"  aspect    : {alert.signals.get('aspect_value')} "
            f"(triggered: {alert.signals.get('aspect')})\n"
            f"  collapsed : {alert.signals.get('collapse')}\n"
            f"  descent   : {alert.signals.get('descent_value')} px/s "
            f"(triggered: {alert.signals.get('descent')})\n\n"
            f"Sent by the SiMa Modalix fall-detection app.\n"
        )
        if alert.snapshot_path and Path(alert.snapshot_path).is_file():
            data = Path(alert.snapshot_path).read_bytes()
            msg.add_attachment(
                data, maintype="image", subtype="jpeg",
                filename=Path(alert.snapshot_path).name,
            )
        return msg

    def send_now(self, alert: Alert) -> None:
        """Send one alert synchronously. Raises on any SMTP failure."""
        msg = self.build_message(alert)
        if self.cfg.dry_run:
            print(
                f"[alert:dry-run] would email {len(self.cfg.recipients)} recipient(s): "
                f"{msg['Subject']}",
                flush=True,
            )
            return
        password = alert_password(self.cfg)
        if self.cfg.ssl:
            server = smtplib.SMTP_SSL(self.cfg.host, self.cfg.port, timeout=self.cfg.timeout)
        else:
            server = smtplib.SMTP(self.cfg.host, self.cfg.port, timeout=self.cfg.timeout)
        try:
            if self.cfg.starttls and not self.cfg.ssl:
                server.starttls()
            if self.cfg.username:
                server.login(self.cfg.username, password)
            server.send_message(msg)
        finally:
            try:
                server.quit()
            except Exception:
                pass


def alert_password(cfg: "AlertConfig") -> str:
    """Read the SMTP password from the environment.

    Deliberately not a config key. ``config.yaml`` is committed to the repo, and
    a password in it is a password on GitHub. The variable is read at send time
    so rotating it does not need a restart.
    """
    if not cfg.username:
        return ""
    password = os.environ.get(cfg.password_env, "")
    if not password:
        raise RuntimeError(
            f"alerts.username is set but ${cfg.password_env} is empty.\n"
            f"  export {cfg.password_env}='...' before running, or set "
            f"alerts.dry_run: true to test without sending."
        )
    return password


# ─────────────────────────────────────────────────────────────────────────────
# Frames, overlay, sinks
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FrameStamp:
    """The timing fields a sink needs, copied out of a sample as plain ints.

    The point is to stop holding the sample itself. A decoded frame is a buffer
    from the hardware decoder's pool, and that pool is small: this board reports
    ``BufferNum=8``. Anything still referencing a sample is still holding one of
    those eight, so keeping one alive across the next ``pull()`` -- while
    compositing, encoding a JPEG and writing a video frame -- starves the
    decoder and deadlocks the pipeline. The run stops with a pull timeout after
    almost exactly as many frames as the pool has buffers.
    """

    pts_ns: int = -1
    dts_ns: int = -1
    duration_ns: int = 0
    frame_id: int = -1
    stream_id: int = 0

    @classmethod
    def of(cls, sample) -> "FrameStamp":
        return cls(
            pts_ns=getattr(sample, "pts_ns", -1),
            dts_ns=getattr(sample, "dts_ns", -1),
            duration_ns=getattr(sample, "duration_ns", 0),
            frame_id=getattr(sample, "frame_id", -1),
            stream_id=getattr(sample, "stream_id", 0),
        )


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

FONT = cv2.FONT_HERSHEY_SIMPLEX if cv2 is not None else 0

# All digits share one vertical extent in this font, so folding them to a single
# digit keeps the cache to one entry per distinct caption rather than one per
# confidence value.
DIGIT_FOLD = str.maketrans("123456789", "000000000")

_ink_cache: dict[tuple[str, int, int], tuple[int, int]] = {}


def text_ink_extent(text: str, text_scale: float, text_thickness: int) -> tuple[int, int]:
    """Measure how far this string's ink reaches above and below the baseline.

    ``cv2.getTextSize`` reports a height that includes internal leading and a
    baseline drop sized for descenders, so a band sized from it holds the glyphs
    visibly off centre in whichever direction the string happens to lack. The
    correction has to be per string: ``FPS: 24.7`` has no descender, ``person``
    does. Rendering once and reading back the occupied rows measures exactly
    what will be drawn.

    Args:
        text: The string that will be drawn.
        text_scale: OpenCV font scale.
        text_thickness: Stroke weight in pixels.

    Returns:
        An ``(above, below)`` pair of pixel counts relative to the baseline.
    """
    key = (text.translate(DIGIT_FOLD), int(round(text_scale * 1000)), int(text_thickness))
    cached = _ink_cache.get(key)
    if cached is not None:
        return cached

    (width, height), baseline = cv2.getTextSize(text, FONT, text_scale, text_thickness)
    margin = max(4, text_thickness * 3)
    canvas = np.zeros((height + baseline + margin * 2, width + margin * 2), np.uint8)
    origin_y = margin + height
    cv2.putText(
        canvas, text, (margin, origin_y), FONT, text_scale, 255,
        text_thickness, cv2.LINE_AA,
    )
    rows = np.where(canvas.max(axis=1) > 0)[0]
    if rows.size == 0:                          # blank or degenerate, fall back
        extent = (height, baseline)
    else:
        extent = (origin_y - int(rows.min()), max(0, int(rows.max()) - origin_y + 1))
    _ink_cache[key] = extent
    return extent


def class_color(class_id: int) -> tuple[int, int, int]:
    """Pick a stable colour for a class.

    Args:
        class_id: Model class id.

    Returns:
        A BGR tuple, repeating every 20 classes.
    """
    return BOX_COLORS[class_id % len(BOX_COLORS)]


def draw_scale(frame, draw: DrawConfig) -> float:
    """Scale factor that keeps strokes and text proportional to the frame.

    Args:
        frame: The image being annotated.
        draw: Visualization settings.

    Returns:
        A multiplier, 1.0 at the reference height, floored so small frames stay
        readable. Always 1.0 when ``auto_scale`` is off.
    """
    if not draw.auto_scale or draw.reference_height <= 0:
        return 1.0
    return max(0.4, min(frame.shape[:2]) / draw.reference_height)


def caption_text(box: dict, labels: list[str], draw: DrawConfig) -> str:
    """Build the caption for one detection.

    Args:
        box: A single detection.
        labels: Class names indexed by class id.
        draw: Visualization settings.

    Returns:
        The caption, which is empty when both the label and the score are off.
    """
    class_id = int(box["class_id"])
    parts = []
    if draw.show_labels:
        parts.append(labels[class_id] if 0 <= class_id < len(labels) else str(class_id))
    if draw.show_scores:
        parts.append(f"{box['score']:.{max(0, draw.score_decimals)}f}")
    return " ".join(parts)


def draw_boxes(frame, boxes: list[dict], labels: list[str], draw: DrawConfig) -> None:
    """Draw detection boxes, centre markers and labelled captions in place.

    Each box is a plain rectangle in the class colour, with a centre dot and a
    filled caption sitting directly above it. The caption flips below the box
    when it would otherwise clip off the top of the frame.

    Args:
        frame: BGR image, modified in place.
        boxes: Detections with ``x1``, ``y1``, ``x2``, ``y2``, ``score`` and
            ``class_id`` keys, in original-image pixels.
        labels: Class names indexed by class id.
        draw: Visualization settings, from the ``visualization`` config section.
    """
    h, w = frame.shape[:2]
    scale = draw_scale(frame, draw)
    thickness = max(1, int(round(draw.box_thickness * scale)))
    text_scale = draw.text_scale * scale
    text_thickness = max(1, int(round(draw.text_thickness * scale)))
    pad = max(2, int(round(draw.text_padding * scale)))
    radius = max(2, int(round(draw.centre_dot_radius * scale)))

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

        color = class_color(int(box["class_id"]))

        if draw.centre_dot:
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), radius, color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = caption_text(box, labels, draw)
        if not label:
            continue

        (text_w, _), _ = cv2.getTextSize(label, FONT, text_scale, text_thickness)
        above, below = text_ink_extent(label, text_scale, text_thickness)
        band_w = text_w + pad * 2
        band_h = above + below + pad * 2

        top = y1 - band_h
        if top < 0:                      # would clip off the top, so sit inside
            top = y1
        left = max(0, min(x1, w - band_w))

        cv2.rectangle(frame, (left, top), (left + band_w, top + band_h), color, -1)
        cv2.putText(
            frame, label, (left + pad, top + pad + above), FONT, text_scale,
            draw.text_color, text_thickness, cv2.LINE_AA,
        )


# Per-state colours, BGR. Deliberately traffic-light rather than the class
# palette: on this app the state is the information, not which class it is.
STATE_COLORS = {
    UPRIGHT: (98, 205, 0),       # green, nothing to see
    FALLING: (0, 194, 255),      # amber, the timer is running
    FALLEN: (56, 56, 255),       # red, alert raised
    RECOVERING: (227, 195, 0),   # cyan, back up but still being watched
}


def state_color(state: str) -> tuple[int, int, int]:
    return STATE_COLORS.get(state, (200, 200, 200))


def track_caption(track: Track, draw: DrawConfig) -> str:
    """Build the caption for one tracked person."""
    parts = []
    if draw.show_track_ids:
        parts.append(f"#{track.track_id}")
    if draw.show_labels:
        parts.append(track.state)
    if draw.show_scores:
        parts.append(f"{track.box['score']:.{max(0, draw.score_decimals)}f}")
    return " ".join(parts)


def draw_tracks(frame, tracks: list[Track], draw: DrawConfig, fall: FallConfig) -> None:
    """Draw every tracked person, coloured by state, in place.

    Args:
        frame: BGR image, modified in place.
        tracks: Live tracks with their fall state already resolved.
        draw: Visualization settings.
        fall: Fall settings, for the countdown readout on a pending fall.
    """
    height, width = frame.shape[:2]
    scale = draw_scale(frame, draw)
    thickness = max(1, int(round(draw.box_thickness * scale)))
    text_scale = draw.text_scale * scale
    text_thickness = max(1, int(round(draw.text_thickness * scale)))
    pad = max(2, int(round(draw.text_padding * scale)))
    radius = max(2, int(round(draw.centre_dot_radius * scale)))

    # Paint larger boxes first, so a small figure in front stays legible.
    ordered = sorted(tracks, key=lambda t: t.width * t.height, reverse=True)
    for track in ordered:
        x1 = max(0, int(round(track.box["x1"])))
        y1 = max(0, int(round(track.box["y1"])))
        x2 = min(width - 1, int(round(track.box["x2"])))
        y2 = min(height - 1, int(round(track.box["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue

        color = state_color(track.state)
        # A fallen person gets a heavier box, so the frame reads correctly even
        # in a thumbnail or a greyscale printout.
        weight = thickness * 2 if track.state == FALLEN else thickness
        if draw.centre_dot:
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), radius, color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, weight)

        label = track_caption(track, draw)
        if track.state == FALLING and fall.confirm_seconds > 0:
            held = max(0.0, time_ms() / 1000.0 - track.state_since)
            label = f"{label} {min(held, fall.confirm_seconds):.1f}/{fall.confirm_seconds:.1f}s"
        if not label:
            continue

        (text_w, _), _ = cv2.getTextSize(label, FONT, text_scale, text_thickness)
        above, below = text_ink_extent(label, text_scale, text_thickness)
        band_w = text_w + pad * 2
        band_h = above + below + pad * 2

        top = y1 - band_h
        if top < 0:
            top = y1
        left = max(0, min(x1, width - band_w))

        cv2.rectangle(frame, (left, top), (left + band_w, top + band_h), color, -1)
        cv2.putText(
            frame, label, (left + pad, top + pad + above), FONT, text_scale,
            draw.text_color, text_thickness, cv2.LINE_AA,
        )


def draw_banner(frame, text: str, draw: DrawConfig) -> None:
    """Draw a full-width alert strip across the bottom of the frame.

    A red box around one person is easy to miss on a wall of camera tiles. A
    band across the whole frame is not, which is the point of it.

    Args:
        frame: BGR image, modified in place.
        text: Banner text.
        draw: Visualization settings.
    """
    height, width = frame.shape[:2]
    scale = draw_scale(frame, draw)
    text_scale = (draw.banner_text_scale or draw.text_scale) * scale
    text_thickness = max(1, int(round((draw.banner_text_thickness or draw.text_thickness) * scale)))
    pad = max(4, int(round(draw.banner_padding * scale)))

    (text_w, _), _ = cv2.getTextSize(text, FONT, text_scale, text_thickness)
    above, below = text_ink_extent(text, text_scale, text_thickness)
    band_h = above + below + pad * 2
    top = height - band_h

    strip = frame[top:height, 0:width]
    tint = np.empty_like(strip)
    tint[:] = draw.banner_bg_color
    # Translucent rather than solid, so the band never hides the thing it is
    # drawing attention to.
    cv2.addWeighted(tint, draw.banner_alpha, strip, 1.0 - draw.banner_alpha, 0.0, dst=strip)
    cv2.putText(
        frame, text, (max(pad, (width - text_w) // 2), top + pad + above),
        FONT, text_scale, draw.banner_text_color, text_thickness, cv2.LINE_AA,
    )


def draw_fps(frame, fps: float, draw: DrawConfig) -> None:
    """Draw the frame rate centred in a filled badge at the top left.

    Font size, stroke weight and padding fall back to the shared caption
    settings when their ``hud`` counterparts are 0, so the badge matches the
    captions unless it is deliberately given its own look.

    Args:
        frame: BGR image, modified in place.
        fps: Frames per second to display.
        draw: Visualization settings.
    """
    scale = draw_scale(frame, draw)
    text_scale = (draw.hud_text_scale or draw.text_scale) * scale
    raw_thickness = draw.hud_text_thickness or draw.text_thickness
    text_thickness = max(1, int(round(raw_thickness * scale)))

    # padding_x / padding_y fall back to the shared hud padding, which itself
    # falls back to the caption padding, so one number still styles everything
    # and two numbers give independent control of the horizontal and vertical
    # breathing room.
    base_pad = draw.hud_padding or draw.text_padding
    pad_x = max(0, int(round((draw.hud_padding_x or base_pad) * scale)))
    pad_y = max(0, int(round((draw.hud_padding_y or base_pad) * scale)))

    text = f"FPS: {fps:.{max(0, draw.hud_fps_decimals)}f}"
    (text_w, _), _ = cv2.getTextSize(text, FONT, text_scale, text_thickness)
    above, below = text_ink_extent(text, text_scale, text_thickness)

    # Padding sets the badge size; a minimum can only grow it. The text is then
    # centred in whatever box results, so a forced size never pushes it off.
    box_w = max(text_w + pad_x * 2, int(round(draw.hud_min_width * scale)))
    box_h = max(above + below + pad_y * 2, int(round(draw.hud_min_height * scale)))
    left = max(0, int(round((draw.hud_margin_x or (draw.hud_padding_x or base_pad)) * scale))) or 1
    top = max(0, int(round((draw.hud_margin_y or (draw.hud_padding_y or base_pad)) * scale))) or 1

    cv2.rectangle(
        frame, (left, top), (left + box_w, top + box_h), draw.hud_bg_color, -1
    )
    cv2.putText(
        frame,
        text,
        (
            left + (box_w - text_w) // 2,
            top + (box_h - (above + below)) // 2 + above,
        ),
        FONT,
        text_scale,
        draw.hud_text_color,
        text_thickness,
        cv2.LINE_AA,
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


def track_objects(tracks, labels: list[str]) -> list[dict]:
    objects = []
    for track in tracks:
        class_id = int(track.box["class_id"])
        objects.append(
            {
                "id": f"track_{track.track_id}",
                "label": labels[class_id] if 0 <= class_id < len(labels) else "unknown",
                "confidence": float(track.box["score"]),
                "bbox": [
                    float(track.box["x1"]), float(track.box["y1"]),
                    float(track.box["x2"] - track.box["x1"]),
                    float(track.box["y2"] - track.box["y1"]),
                ],
                "state": track.state,
                "aspect": round(track.aspect, 2),
            }
        )
    return objects


def send_metadata(pipeline: Pipeline, sample, boxes: list[dict]) -> None:
    if pipeline.metadata_sender is None:
        return
    timestamp_ms = int(stamp.pts_ns // 1_000_000) if stamp.pts_ns >= 0 else -1
    frame_id = str(stamp.frame_id) if stamp.frame_id >= 0 else ""
    pipeline.metadata_sender.send_metadata(
        "fall-detection",
        json.dumps(
            {"objects": track_objects(boxes, pipeline.labels)},
            separators=(",", ":"),
        ),
        timestamp_ms,
        frame_id,
    )


def push_video(pipeline: Pipeline, stamp: FrameStamp, frame_bgr) -> None:
    """Send one frame to the Insight preview, dropping it if the feed is busy.

    Best effort by design. A refused push means the encoder or UDP egress is
    behind; skipping that frame keeps inference and the recording at full rate,
    which matters more than a complete preview.

    Args:
        pipeline: Live pipeline, whose ``video_run`` may be None.
        stamp: Timing fields copied from the source sample.
        frame_bgr: BGR image to send.
    """
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
    video_sample.pts_ns = stamp.pts_ns
    video_sample.dts_ns = stamp.dts_ns
    video_sample.duration_ns = stamp.duration_ns
    video_sample.frame_id = stamp.frame_id
    video_sample.stream_id = stamp.stream_id
    try:
        if not pipeline.video_run.push([video_sample]):
            pipeline.video_dropped += 1
    except Exception:
        pipeline.video_dropped += 1


def wants_jpeg(cfg: AppConfig, index: int) -> bool:
    return cfg.save_enable and cfg.save_every > 0 and index % cfg.save_every == 0


def person_boxes(cfg: AppConfig, pipeline: Pipeline, boxes: list[dict], frame_h: int):
    """Keep only the classes that can fall, and only boxes big enough to judge."""
    floor = cfg.fall.min_box_height * frame_h
    kept = []
    for box in boxes:
        if pipeline.fall_class_ids is not None:
            if int(box["class_id"]) not in pipeline.fall_class_ids:
                continue
        if (box["y2"] - box["y1"]) < floor:
            continue
        kept.append(box)
    return kept


def write_snapshot(cfg: AppConfig, track, frame_index: int, frame) -> str:
    """Write the annotated frame that an alert refers to. Returns the path."""
    if frame is None or not cfg.alerts.attach_snapshot:
        return ""
    directory = Path(cfg.alerts.snapshot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"fall_track{track.track_id:03d}_frame{frame_index:06d}.jpg"
    if not cv2.imwrite(str(out), frame):
        print(f"[warn] failed to write snapshot {out}", file=sys.stderr)
        return ""
    return str(out)


def render_annotated(cfg: AppConfig, pipeline: Pipeline, frame, tracks, fps: float):
    """Draw once per frame and share the result between the video and JPEG sinks."""
    annotated = frame.copy()
    # FPS first, so a person in the top-left corner is never hidden by it.
    if cfg.video_hud:
        draw_fps(annotated, fps, cfg.draw)
    draw_tracks(annotated, tracks, cfg.draw, cfg.fall)
    down = [t for t in tracks if t.state == FALLEN]
    if cfg.draw.banner and down:
        ids = ", ".join(f"#{t.track_id}" for t in down[:6])
        more = "" if len(down) <= 6 else f" +{len(down) - 6}"
        draw_banner(
            annotated,
            f"FALL DETECTED  {ids}{more}  -  {cfg.alerts.site}",
            cfg.draw,
        )
    return annotated


def save_frame(cfg: AppConfig, index: int, frame) -> None:
    out_path = Path(cfg.save_dir) / f"frame_{index:06d}.{cfg.save_format}"
    if not cv2.imwrite(str(out_path), frame):
        print(f"[warn] failed to write {out_path}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Run loop
# ─────────────────────────────────────────────────────────────────────────────


class Stopper:
    """Cooperative stop flag driven by SIGINT, SIGTERM and SIGHUP.

    ``dk`` and ``devkit-run`` invoke over SSH without a pty, so a terminal
    Ctrl-C is never forwarded and the app can be orphaned on the DevKit while
    still holding the MLA. The next run then fails with a busy device. Launch
    interactive runs with ``ssh -tt`` and let these handlers close the Run.

    Attributes:
        stop: Set to True once a signal has been received.
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
    """Rolling per-stage timing accumulator.

    Sums pull, decode and sink latencies over a fixed number of frames, then
    prints one averaged line and resets. Averaging avoids the noise of
    per-frame timings without needing to keep every sample.

    Attributes:
        enabled: Whether profiling output is on at all.
        interval: Frames per window before a flush.
    """

    def __init__(self, enabled: bool, interval: int) -> None:
        """Initialise the window.

        Args:
            enabled: Whether to collect and print timings.
            interval: Frames to accumulate before printing a summary.
        """
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
                print(
                    f"source produced nothing for {cfg.pull_timeout_ms} ms after "
                    f"{processed} frames; stopping.\n"
                    "If that is far short of the clip length the source stalled "
                    "rather than ended. In order of likelihood:\n"
                    "  1. the hardware decoder ran out of buffers. Its pool is small "
                    "(the boot log\n     prints BufferNum), so anything holding a "
                    "decoded sample starves it. Cut\n     per-frame work: "
                    "output.save.every: 10 rather than 1 is the usual win, and\n"
                    "     lower runtime.queue_depth.\n"
                    "  2. output.insight.enable is on and its encoder wedged the "
                    "shared codec daemon.\n"
                    "  3. the clip really ended.",
                    flush=True,
                )
                break
            continue

        stamp = FrameStamp.of(sample)
        boxes = parse_boxes(
            extract_bbox_payload(sample),
            pipeline.frame_w,
            pipeline.frame_h,
            cfg.max_detections,
        )
        frame = frame_to_bgr(first_tensor(joined_field(sample, "frame", 0)))
        # `boxes` and `frame` are copies, so give the decoder its buffer back
        # before the sinks rather than after. See FrameStamp for why.
        sample = None
        decode_end = time_ms()

        # Track, then judge. Both need a clock, and the source's own timestamps
        # are the honest one: with overflow_policy block the run is slower than
        # realtime, so wall-clock seconds would make every descent look slow.
        now = (
            stamp.pts_ns / 1e9 if stamp.pts_ns >= 0
            else processed / float(pipeline.fps or 25)
        )
        people = person_boxes(cfg, pipeline, boxes, pipeline.frame_h)
        tracks = pipeline.tracker.update(people, now)
        fallen_now = (
            update_fall_states(tracks, cfg.fall, pipeline.frame_h, now)
            if cfg.fall.enable else []
        )

        processed += 1
        need_jpeg = wants_jpeg(cfg, processed)
        need_annotated = (
            pipeline.writer is not None
            or (cfg.insight_enable and cfg.insight_annotated)
            or (need_jpeg and cfg.save_overlay)
        )

        # A confirmed fall always needs a rendered frame, whatever the sinks
        # asked for: it is what gets attached to the alert.
        need_annotated = need_annotated or bool(fallen_now)

        # Render once, then share the result across every sink that wants it.
        annotated = (
            render_annotated(cfg, pipeline, frame, tracks, live_fps)
            if need_annotated
            else None
        )

        for track in fallen_now:
            pipeline.falls += 1
            signals = fall_signals(track, cfg.fall, pipeline.frame_h)
            snapshot = write_snapshot(cfg, track, processed, annotated)
            label = (
                pipeline.labels[int(track.box["class_id"])]
                if 0 <= int(track.box["class_id"]) < len(pipeline.labels) else "person"
            )
            print(
                f"[FALL] track #{track.track_id} ({label}) at frame {processed} "
                f"aspect={signals['aspect_value']} descent={signals['descent_value']}px/s"
                + (f" snapshot={snapshot}" if snapshot else ""),
                flush=True,
            )
            queued = pipeline.alerts.offer(
                Alert(
                    track_id=track.track_id, when=now, label=label,
                    box=(int(track.box["x1"]), int(track.box["y1"]),
                         int(track.box["x2"]), int(track.box["y2"])),
                    signals=signals, frame_index=processed, snapshot_path=snapshot,
                ),
                now,
            )
            track.alerted_at = now if queued else track.alerted_at

        # With insight_annotated the viewer shows our overlay. Without it,
        # Insight receives the raw frame and draws its own from the metadata.
        push_video(
            pipeline, stamp,
            annotated if (cfg.insight_annotated and annotated is not None) else frame,
        )
        send_metadata(pipeline, stamp, tracks)

        if pipeline.writer is not None:
            pipeline.writer.write(annotated)
            pipeline.writer_frames += 1
        if need_jpeg:
            save_frame(cfg, processed, annotated if cfg.save_overlay else frame)
        sink_end = time_ms()

        profile.add(
            pull_end - pull_start, decode_end - pull_end, sink_end - decode_end, len(boxes)
        )

        # Heartbeat, so a healthy run does not look identical to a stalled one.
        heartbeat_boxes += len(tracks)
        if processed % HEARTBEAT_EVERY == 0:
            elapsed = time_ms() - heartbeat_start
            rate = HEARTBEAT_EVERY * 1000.0 / elapsed if elapsed > 0 else 0.0
            live_fps = rate or live_fps
            print(
                f"[{processed}] {rate:.1f} fps, "
                f"{heartbeat_boxes / HEARTBEAT_EVERY:.1f} people/frame avg, "
                f"{pipeline.falls} fall(s) so far",
                flush=True,
            )
            heartbeat_start = time_ms()
            heartbeat_boxes = 0

    profile.flush()
    print(f"processed={processed} timeouts={timeouts} falls={pipeline.falls}")

    # A recording that is over almost before it starts is the most commonly
    # reported symptom, and it has three quite different causes. Rank them here
    # rather than leaving the frame count to be interpreted.
    out_fps = cfg.video_fps or pipeline.fps or 25
    seconds = pipeline.writer_frames / out_fps if out_fps else 0.0
    if pipeline.writer is not None and pipeline.writer_frames and seconds < 2.0:
        causes = []
        if cfg.insight_enable:
            causes.append(
                "output.insight.enable is true. Its H.264 encoder shares the codec "
                "daemon with the decoder feeding the source, so a failing encoder "
                "stalls the run. Set it to false; the recording does not need it."
            )
        if cfg.frames:
            causes.append(f"runtime.frames is {cfg.frames}, which capped the run.")
        if timeouts:
            causes.append(
                f"the source stopped producing frames ({timeouts} timeout(s)), so "
                "the run ended before the clip did."
            )
        if not causes:
            causes.append("the source clip really is that short.")
        listed = "\n".join(f"       {i}. {c}" for i, c in enumerate(causes, 1))
        print(
            f"[warn] the recording is only {seconds:.1f}s "
            f"({pipeline.writer_frames} frames at {out_fps} fps).\n{listed}",
            file=sys.stderr, flush=True,
        )
    if pipeline.metadata_sender is not None:
        stats = pipeline.metadata_sender.stats()
        print(
            f"metadata: sent={stats.datagrams_sent} failures={stats.send_failures} "
            f"would_block={stats.would_block}"
        )
    if pipeline.video_dropped:
        print(
            f"insight: dropped {pipeline.video_dropped} preview frames because the "
            f"feed was busy. The recording is unaffected."
        )
    return processed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SiMa Neat fall detection with SMTP alerts"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Parse and validate the config without loading pyneat or the model.",
    )
    parser.add_argument(
        "--test-alert",
        action="store_true",
        help="Send one fake alert and exit. Proves the SMTP settings and the "
             "password environment variable work, without waiting for a fall.",
    )
    args = parser.parse_args(argv)

    pipeline = None
    try:
        cfg = load_app_config(args.config)
        labels = load_labels(cfg.labels_path)
        if args.validate_config:
            watched = resolve_fall_classes(cfg, labels)
            print(f"config OK: {args.config}")
            print(f"  family={cfg.family} -> BoxDecodeType.{FAMILY_DECODE_TOKENS[cfg.family]}")
            print(f"  {describe_preprocess(cfg, cfg.source_width, cfg.source_height)}")
            print(f"  {describe_fall(cfg, labels)}")
            print(f"  {describe_alerts(cfg)}")
            if watched is not None:
                print(f"  watched class ids: {sorted(watched)}")
            return 0

        if args.test_alert:
            # Deliberately synchronous and outside the pipeline: this is the
            # command you run when mail is not arriving, so it must report the
            # real exception rather than queue the work and return 0.
            if not cfg.alerts.enable:
                print("[ERR] alerts.enable is off, so there is nothing to test.",
                      file=sys.stderr)
                return 1
            sender = AlertSender(dataclasses.replace(cfg.alerts, enable=False))
            probe = Alert(
                track_id=0, when=0.0, label="person", box=(100, 100, 400, 700),
                signals={"aspect": True, "collapse": False, "descent": False,
                         "aspect_value": 1.35, "descent_value": 0.0},
                frame_index=0,
            )
            print(describe_alerts(cfg))
            sender.send_now(probe)
            print("test alert sent." if not cfg.alerts.dry_run
                  else "test alert composed (dry_run is on, nothing left the board).")
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
