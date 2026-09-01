"""Instance segmentation with a background blur on the SiMa Modalix DevKit.

Reads a video file, RTSP stream or camera, runs a YOLO segmentation model on the
MLA accelerator, rebuilds a per-instance mask for every detection, and composites
the frame so the instances stay sharp while everything behind them is blurred.
Results are published three ways: an annotated video and stills on the DevKit,
plus a live Neat Insight feed over UDP.

Built against the Neat Library public Python API (pyneat) as packaged in
2.1.2_Palette_SDK. Runs on the DevKit, not in the x86 SDK container, because
pyneat is compiled for aarch64.

The pipeline is a Graph rather than a single Model.run call, because it has
multiple stages, named public endpoints and a branch with a fan-in::

    source --> branch --> frame ---------------+
                    |                          +--> combine("segmenter_output")
                    +---> model --> instances -+

    segmenter_output --> BBOX + mask parse --> foreground mask
                                           --> blur composite --> video + stills
                                                              +-> MetadataSender
                                                              +-> VideoSender

The compositing half is the part that is specific to this app::

    frame ──┬─────────────────────────────► kept where mask == 1 ──┐
            │                                                      ├──► output
            └─► gaussian / pixelate ──────► kept where mask == 0 ──┘

Typical usage::

    python3 src/app.py --config config.yaml
    python3 src/app.py --config config.yaml --validate-config
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import queue
import signal
import struct
import subprocess
import sys
import threading
import time
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
class SegmentConfig:
    """How per-instance masks are recovered from the model output.

    A detect head emits one BBOX tensor and nothing else. A segment head emits
    mask data as well, and there are three shapes it comes in:

    * **packed** - what SiMa's ``neatobjectdecode`` actually produces: a single
      UInt8 tensor whose head is the ordinary BBOX array and whose tail is one
      finished mask per *box slot*. There are ``top_k`` slots, not ``count``, so
      the buffer is a fixed size every frame::

          [uint32 count][RawBox 24B * top_k][mask side*side uint8 * top_k]

      With ``top_k=50`` and a 640-input model that is
      ``4 + 50*24 + 50*160*160 = 1,281,204`` bytes.
    * **planes** - the same finished masks, but in a tensor of their own,
      ``N x mh x mw``.
    * **proto** - the Ultralytics YOLO-seg encoding. A bank of ``C`` prototype
      planes shared by the whole frame, ``C x mh x mw``, plus ``C`` coefficients
      per instance. An instance's mask is the coefficient-weighted sum of the
      prototypes, so it costs one matmul per frame.

    ``auto`` tries packed first, because it is the layout the shipped model
    packs use, then falls back to looking at any separate tensors. ``describe``
    prints what actually arrived on the first frame, so pinning ``source``
    afterwards is a one-line change.

    Attributes:
        masks: Master switch. ``off`` skips mask decoding entirely and blurs
            around plain bounding boxes, which needs no segment head at all.
        source: ``auto``, ``packed``, ``proto`` or ``planes``.
        space: Which coordinate space a mask lives in. ``net`` means the
            model's own letterboxed input space, which is the YOLO convention
            and needs the letterbox undone. ``box`` means the mask is already
            cropped to its own detection and only needs scaling. ``auto``
            decides on the first frame by checking whether the mask's occupied
            region lands where its box says it should, and prints the verdict.
        threshold: Mask cut-off as a probability, so 0.5 is the YOLO default.
            Lower it to grow instances, raise it to tighten them.
        stride: Model input pixels per mask pixel, used to infer the network
            input size when ``preprocess.resize`` is left at 0. 4 for YOLO-seg,
            whose 640x640 input gives 160x160 prototypes.
        net_width: Model input width. 0 derives it, see ``stride``.
        net_height: Model input height. 0 derives it.
        coeff_counts: Prototype bank sizes to recognise while auto-detecting.
        mask_sides: Mask edge lengths to try when solving a packed buffer whose
            slot count cannot be taken from ``decode.max_detections``.
        fallback_to_boxes: Whether to fall back to box-shaped regions when no
            mask data can be found, rather than failing the run. The app says so
            once, loudly, and keeps going.
        describe: Whether to print every tensor in the first frame's output,
            with its tag, dtype and shape. This is how you learn what your model
            pack emits without reading any SDK source.
        dilate: Grow the finished mask by this many pixels, expressed for a
            1080p frame. Small positive values cover the halo a tight mask
            leaves around hair and limbs.
        blur_mask: Smooth the mask over this many pixels before it is used, also
            expressed for a 1080p frame. Takes the staircase off the edge that a
            160x160 mask shows once it is scaled up. 0 disables.
    """

    masks: str = "on"
    source: str = "auto"
    space: str = "auto"
    threshold: float = 0.5
    stride: int = 4
    net_width: int = 0
    net_height: int = 0
    coeff_counts: tuple[int, ...] = (16, 32, 64)
    mask_sides: tuple[int, ...] = (
        64, 80, 88, 96, 104, 112, 120, 128, 144, 152, 160, 176, 192, 208, 224, 240, 256, 320
    )
    fallback_to_boxes: bool = True
    describe: bool = True
    dilate: int = 0
    blur_mask: int = 0


@dataclass(frozen=True)
class BlurConfig:
    """The background treatment, straight from the ``blur`` config section.

    Pixel sizes here are expressed for a 1080p frame and follow
    ``visualization.auto_scale``, so a kernel that looks right on a test clip
    still looks right at 4K.

    This whole block is **optional**. The app is a segmenter: it finds instances
    and draws their masks whatever happens here. ``enable: off`` turns the
    background treatment off and leaves a plain segmentation overlay, and
    ``opacity`` is the dial between the two rather than an on/off switch.

    Attributes:
        enable: Whether to composite at all. ``off`` leaves the frame untouched
            and the app becomes a plain segmentation overlay.
        opacity: How much of the treated background to keep, 0.0 to 1.0. 1.0 is
            a full-strength blur, 0.4 is a hint of one, 0.0 is indistinguishable
            from ``enable: off``. Applied after the method, so it dials
            ``grayscale`` and ``dim`` down too.
        method: ``gaussian``, ``pixelate`` or ``none``. ``none`` still applies
            ``dim`` and ``grayscale``, which is how you get a darkened rather
            than blurred background.
        kernel: Gaussian kernel width in pixels. Forced odd. Bigger is blurrier
            and slower, though ``downscale`` absorbs most of the cost.
        sigma: Gaussian sigma. 0 derives it from ``kernel``, which is what you
            want almost always.
        downscale: Blur at 1/N resolution and scale back up. The dominant speed
            knob: at 1080p, 2 roughly quarters the work and is visually free,
            because the result is a blur either way.
        pixel_size: Mosaic block size for ``pixelate``, in pixels.
        dim: Darken the background, 0.0 to 1.0. 0 leaves brightness alone.
        grayscale: Whether to desaturate the background.
        feather: Soften the mask edge over this many pixels. 0 gives a hard
            cut, which is faster but shows every jag in the mask.
        invert: Blur the instances instead of the background. This is the
            anonymiser: ``keep_classes: [person]`` plus ``invert: on`` blurs
            people and leaves the scene sharp.
        keep_classes: Class names or ids that count as foreground. Empty keeps
            every detected class.
    """

    enable: bool = True
    opacity: float = 1.0
    method: str = "gaussian"
    kernel: int = 41
    sigma: float = 0.0
    downscale: int = 2
    pixel_size: int = 24
    dim: float = 0.0
    grayscale: bool = False
    feather: int = 9
    invert: bool = False
    keep_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DrawConfig:
    """Overlay appearance, straight from the ``visualization`` config section.

    Every pixel size here is expressed for a 1080p frame. When ``auto_scale``
    is on they are multiplied by ``min(frame_height, frame_width) /
    reference_height``, so 4K does not get hairlines and 480p does not get
    slabs. Turn ``auto_scale`` off to use the numbers literally.

    Attributes:
        mask_alpha: Strength of the class-coloured tint painted over each
            instance, 0.0 to 1.0. 0 leaves the instance untouched, which is
            what you want when the blur alone should carry the effect.
        mask_outline: Whether to trace the mask edge.
        outline_thickness: Weight of that trace, in pixels.
        show_boxes: Whether to draw the bounding rectangle as well. Off by
            default: the mask already shows the extent, and the box mostly adds
            clutter.
        box_thickness: Rectangle outline weight, in pixels.
        text_scale: OpenCV font scale for captions.
        text_thickness: Caption stroke weight, in pixels.
        text_padding: Gap between caption text and the edge of its band.
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
        auto_scale: Whether sizes scale with frame height.
        reference_height: The frame height the sizes above are tuned for.
    """

    mask_alpha: float = 0.35
    mask_outline: bool = True
    outline_thickness: int = 3
    show_boxes: bool = False
    box_thickness: int = 2
    text_scale: float = 1.0
    text_thickness: int = 2
    text_padding: int = 10
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
    auto_scale: bool = True
    reference_height: float = 1080.0


@dataclass(frozen=True)
class AppConfig:
    """Fully resolved application configuration.

    Built by :func:`load_app_config` from ``config.yaml`` and validated by
    :func:`validate_config` before anything touches the hardware, so a bad
    config fails in under a second instead of part-way through a graph build.

    Attributes:
        model_path: Path to the compiled model archive on the DevKit.
        labels_path: Path to the newline-separated class label file.
        family: Segment head token, mapped by ``FAMILY_DECODE_TOKENS``.
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
        queue_depth: Depth of the runtime's own queues, and of the sink
            thread's queue. It does not change the ``max-buffers`` and
            ``num-buffers`` in the printed pipeline, which pyneat fixes at 4.
        output_buffers: Buffers each public output may hold. Every one of them
            is a frame checked out of the hardware decoder's pool, that pool is
            small (the boot log prints ``BufferNum=8``), and there are two
            outputs. Counts against the same budget as the decoded path in
            front of the source appsink; see ``make_elementary_h264_source``.
        run_preset: One of ``realtime``, ``balanced`` or ``reliable``.
        overflow_policy: One of ``keep_latest``, ``block`` or ``drop_incoming``.
        profile: Whether to print per-stage timings.
        profile_interval: Frames per profiling window.
        save_enable: Whether to write annotated stills.
        save_dir: Directory for stills.
        save_every: Write every Nth frame. 0 disables.
        save_overlay: Whether stills carry the composite and overlay.
        save_format: ``jpg`` or ``png``.
        video_enable: Whether to write an annotated video on the DevKit.
        video_path: Output video path.
        video_codec: Four-character FourCC, with an MJPG fallback.
        video_fps: Output frame rate. 0 matches the source.
        video_hud: Whether to draw the frame-rate badge.
        insight_enable: Whether to stream to Neat Insight.
        insight_annotated: Whether Insight receives the composited frame. False
            sends the raw frame and lets Insight draw its own overlay.
        insight_host: Insight address as the DevKit sees it.
        insight_channel: Channel offset added to both port bases.
        video_port_base: First UDP video port.
        metadata_port_base: First UDP metadata port.
        bitrate_kbps: H.264 encoder bitrate for the Insight feed.
        segment: Mask recovery settings. See :class:`SegmentConfig`.
        blur: Background treatment. See :class:`BlurConfig`.
        draw: Overlay appearance. See :class:`DrawConfig`.
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

    segment: SegmentConfig
    blur: BlurConfig
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


def _triple(raw: dict, key: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"config key `{key}` must be a list of 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _int_list(raw: dict, key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"config key `{key}` must be a list of integers")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"config key `{key}` must contain integers")
        out.append(int(item))
    return tuple(out)


def _str_list(raw: dict, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a list of class names or ids. A bare scalar is accepted as one item."""
    value = raw.get(key)
    if value is None:
        return default
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return (str(value),)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"config key `{key}` must be a list of class names or ids")
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
    default = DrawConfig()
    return DrawConfig(
        mask_alpha=_float(section, "mask_alpha", default.mask_alpha),
        mask_outline=_flag(section, "mask_outline", "on") == "on",
        outline_thickness=_int(section, "outline_thickness", default.outline_thickness),
        show_boxes=_flag(section, "show_boxes", "off") == "on",
        box_thickness=_int(section, "box_thickness", default.box_thickness),
        text_scale=_float(section, "text_scale", default.text_scale),
        text_thickness=_int(section, "text_thickness", default.text_thickness),
        text_padding=_int(section, "text_padding", default.text_padding),
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
        auto_scale=_flag(section, "auto_scale", "on") == "on",
        reference_height=_float(section, "reference_height", default.reference_height),
    )


def load_blur_config(raw: dict) -> BlurConfig:
    section = _section(raw, "blur")
    default = BlurConfig()
    return BlurConfig(
        enable=_flag(section, "enable", "on") == "on",
        opacity=_float(section, "opacity", default.opacity),
        method=_str(section, "method", default.method).lower(),
        kernel=_int(section, "kernel", default.kernel),
        sigma=_float(section, "sigma", default.sigma),
        downscale=_int(section, "downscale", default.downscale),
        pixel_size=_int(section, "pixel_size", default.pixel_size),
        dim=_float(section, "dim", default.dim),
        grayscale=_flag(section, "grayscale", "off") == "on",
        feather=_int(section, "feather", default.feather),
        invert=_flag(section, "invert", "off") == "on",
        keep_classes=_str_list(section, "keep_classes", default.keep_classes),
    )


def load_segment_config(raw: dict) -> SegmentConfig:
    section = _section(raw, "segmentation")
    default = SegmentConfig()
    return SegmentConfig(
        masks=_flag(section, "masks", default.masks),
        source=_str(section, "source", default.source).lower(),
        space=_str(section, "space", default.space).lower(),
        threshold=_float(section, "threshold", default.threshold),
        stride=_int(section, "stride", default.stride),
        net_width=_int(section, "net_width", default.net_width),
        net_height=_int(section, "net_height", default.net_height),
        coeff_counts=_int_list(section, "coeff_counts", default.coeff_counts),
        mask_sides=_int_list(section, "mask_sides", default.mask_sides),
        fallback_to_boxes=_flag(section, "fallback_to_boxes", "on") == "on",
        describe=_flag(section, "describe", "on") == "on",
        dilate=_int(section, "dilate", default.dilate),
        blur_mask=_int(section, "blur_mask", default.blur_mask),
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
        family=_str(model, "family", "yolo26-seg").lower(),
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
        run_preset=_str(runtime, "preset", "auto").lower(),
        overflow_policy=_str(runtime, "overflow_policy", "auto").lower(),
        profile=_bool(runtime, "profile", False),
        profile_interval=_int(runtime, "profile_interval", 100),
        save_enable=_bool(save, "enable", True),
        save_dir=_str(save, "dir", "frames"),
        save_every=_int(save, "every", 10),
        save_overlay=_bool(save, "overlay", True),
        save_format=_str(save, "format", "jpg").lower().lstrip("."),
        video_enable=_bool(video, "enable", True),
        video_path=_str(video, "path", "segmentation.mp4"),
        video_codec=_str(video, "codec", "mp4v"),
        video_fps=_int(video, "fps", 0),
        video_hud=_bool(video, "hud", True),
        insight_enable=_bool(insight, "enable", False),
        insight_annotated=_bool(insight, "annotated", True),
        insight_host=_str(insight, "host", "127.0.0.1"),
        insight_channel=_int(insight, "channel", 0),
        video_port_base=_int(insight, "video_port_base", 9000),
        metadata_port_base=_int(insight, "metadata_port_base", 9100),
        bitrate_kbps=_int(insight, "bitrate_kbps", 2000),
        segment=load_segment_config(raw),
        blur=load_blur_config(raw),
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
    # A detect head emits boxes and nothing else, so asking it for masks gives a
    # run that starts cleanly and then blurs nothing. Catch it here rather than
    # three minutes into a clip.
    if cfg.segment.masks == "on" and cfg.family not in SEG_FAMILIES:
        raise ValueError(
            f"model.family `{cfg.family}` is a detect head, which emits no mask data.\n"
            f"  Either point model.family at a segment head "
            f"({', '.join(sorted(SEG_FAMILIES))}),\n"
            f"  or set segmentation.masks: off to blur around plain bounding boxes."
        )
    if cfg.segment.source not in {"auto", "packed", "proto", "planes"}:
        raise ValueError("segmentation.source must be auto, packed, proto or planes")
    if cfg.segment.space not in {"auto", "net", "box"}:
        raise ValueError("segmentation.space must be auto, net or box")
    if not 0.0 < cfg.segment.threshold < 1.0:
        raise ValueError("segmentation.threshold must be in (0.0, 1.0), exclusive")
    if cfg.segment.stride <= 0:
        raise ValueError("segmentation.stride must be > 0")
    if cfg.segment.net_width < 0 or cfg.segment.net_height < 0:
        raise ValueError("segmentation.net_width and net_height must be >= 0")
    if cfg.segment.dilate < 0:
        raise ValueError("segmentation.dilate must be >= 0")
    if cfg.segment.blur_mask < 0:
        raise ValueError("segmentation.blur_mask must be >= 0")
    if not 0.0 <= cfg.blur.opacity <= 1.0:
        raise ValueError("blur.opacity must be in [0.0, 1.0]")
    if cfg.blur.method not in {"gaussian", "pixelate", "none"}:
        raise ValueError("blur.method must be gaussian, pixelate or none")
    if cfg.blur.kernel < 1:
        raise ValueError("blur.kernel must be >= 1")
    if cfg.blur.sigma < 0:
        raise ValueError("blur.sigma must be >= 0")
    if cfg.blur.downscale < 1:
        raise ValueError("blur.downscale must be >= 1")
    if cfg.blur.pixel_size < 2:
        raise ValueError("blur.pixel_size must be >= 2")
    if not 0.0 <= cfg.blur.dim <= 1.0:
        raise ValueError("blur.dim must be in [0.0, 1.0]")
    if cfg.blur.feather < 0:
        raise ValueError("blur.feather must be >= 0")
    if not 0.0 <= cfg.draw.mask_alpha <= 1.0:
        raise ValueError("visualization.mask_alpha must be in [0.0, 1.0]")
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
    if cfg.queue_depth < 1:
        raise ValueError("runtime.queue_depth must be >= 1")
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
    if not (cfg.save_enable or cfg.insight_enable or cfg.video_enable):
        raise ValueError(
            "enable at least one of output.save, output.video or output.insight"
        )


# Config token -> pyneat enum mapping, Kept as string tokens so the tables
# can be validated before pyneat is imported (the wheel is aarch64-only
# and does not load in the SDK container). model.family -> BoxDecodeType
# attribute name. `yolo11` intentionally maps to YoloV8: BoxDecodeType has
# no YOLO11 member, and Ultralytics YOLO11 exports the same decoupled DFL
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

# Which of those actually carry mask data. Everything else is a detect or pose
# head, and asking it for masks is a config error rather than a runtime one.
SEG_FAMILIES = {name for name in FAMILY_DECODE_TOKENS if name.endswith("-seg")}

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
    global cv2, np, pyneat, FONT
    if pyneat is not None:
        return
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)
    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module

    cv2, np, pyneat = cv2_module, np_module, pyneat_module
    FONT = cv2.FONT_HERSHEY_SIMPLEX


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
    # hardware H.264 decoder and libcamera, BGR for cv2 images.
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


def describe_blur(cfg: AppConfig) -> str:
    blur = cfg.blur
    if not blur.enable or blur.opacity <= 0.0:
        return "blur: off, segmentation overlay only"
    if blur.method == "gaussian":
        shape = f"gaussian kernel={blur.kernel} sigma={blur.sigma or 'auto'} down={blur.downscale}"
    elif blur.method == "pixelate":
        shape = f"pixelate block={blur.pixel_size}"
    else:
        shape = "none"
    extras = []
    if blur.grayscale:
        extras.append("grayscale")
    if blur.dim > 0:
        extras.append(f"dim={blur.dim}")
    if blur.feather:
        extras.append(f"feather={blur.feather}")
    target = "instances" if blur.invert else "background"
    keep = ", ".join(blur.keep_classes) if blur.keep_classes else "every class"
    if blur.opacity < 1.0:
        extras.append(f"opacity={blur.opacity}")
    return f"blur: {shape} on the {target} | keep={keep}" + (
        f" | {' '.join(extras)}" if extras else ""
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


# A slice NAL whose first_mb_in_slice is 0 begins a new primary coded picture;
# any other slice continues the one before it. 24 bytes of RBSP is far more than
# an Exp-Golomb first_mb_in_slice can occupy, so it is always enough to decide.
SLICE_NAL_TYPES = frozenset({1, 5})
SLICE_HEADER_BYTES = 24
PICTURE_SCAN_LIMIT = 512 << 20


def count_pictures_in(buf: bytes, final: bool) -> tuple[int, int]:
    """Count picture starts in one buffer.

    Args:
        buf: Annex-B bytes, starting on a start-code boundary or earlier.
        final: True when no more bytes follow, so a NAL near the end can be
            parsed from what is there rather than deferred.

    Returns:
        A ``(count, consumed)`` pair. ``consumed`` is how many leading bytes are
        finished with; the caller carries the remainder into the next chunk.
    """
    count = 0
    pos = 0
    while True:
        idx = buf.find(b"\x00\x00\x01", pos)
        if idx < 0:
            # Two trailing bytes could still be the head of a split start code.
            return count, len(buf) if final else max(0, len(buf) - 2)
        start = idx + 3
        if start + SLICE_HEADER_BYTES > len(buf):
            if not final:
                return count, idx
            if start >= len(buf):
                return count, len(buf)
        if buf[start] & 0x1F in SLICE_NAL_TYPES:
            rbsp = unescape_rbsp(buf[start + 1 : start + 1 + SLICE_HEADER_BYTES])
            try:
                if BitReader(rbsp).ue() == 0:
                    count += 1
            except (ValueError, IndexError):
                pass
        pos = start


def count_h264_pictures(path: str, limit_bytes: int = PICTURE_SCAN_LIMIT) -> int:
    """Count the coded pictures in a raw Annex-B stream.

    Without this number, "the clip ended" and "the source stalled" are the same
    event from the pull loop: both are silence. That is what let a run stop at 83
    frames of a 379 frame clip and still write a plausible-looking
    ``segmentation.mp4``. The bytes on disk settle it before the run even
    starts, and they need neither a container nor a decoder to do it.

    ``ffprobe -count_frames`` would answer the same question, but it is not on
    the DevKit and it is unreliable on elementary streams, which is the same
    reason :func:`probe_h264_sps` exists.

    Args:
        path: Path to a raw Annex-B H.264 file.
        limit_bytes: Stop scanning after this many bytes, so a very large file
            cannot hold up startup. The result is then a lower bound.

    Returns:
        The number of coded pictures, or 0 if the file could not be read.
    """
    count = 0
    carry = b""
    scanned = 0
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1 << 20) if scanned < limit_bytes else b""
                scanned += len(chunk)
                buf = carry + chunk
                if not buf:
                    break
                found, consumed = count_pictures_in(buf, final=not chunk)
                count += found
                if not chunk:
                    break
                carry = buf[consumed:]
    except OSError:
        return 0
    return count


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
        A ``pyneat.Graph`` of FileInput, H264Parse, Queue and SimaDecode,
        producing NV12 frames.
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

    # No CapsRaw node here, deliberately, and this is the difference between a
    # run that finishes the clip and one that dies on a pull timeout part-way
    # through.
    #
    # `raw_output = False` already appends `videoconvert ! capsfilter
    # caps="video/x-raw(memory:SystemMemory),format=NV12"` to the decoder, so a
    # CapsRaw("NV12") after it constrains nothing that is not already fixed. It
    # is not free, though: the Graph inserts `queue max-size-buffers=5` between
    # adjacent nodes, and never before a terminal appsink. With the extra node
    # the decoded path was
    #
    #   neatdecoder ! videoconvert ! capsfilter ! queue(5) ! capsfilter ! appsink(4)
    #
    # so 5 + 4 = 9 decoded frames could sit downstream at once. The hardware
    # decoder's pool is 8 (`BufferNum=8` in the boot log) and it needs several
    # of those for its own reference frames, so the path could swallow the
    # entire pool. The decoder then cannot produce, the app cannot consume, and
    # nothing is ever released. Without the node the path is
    #
    #   neatdecoder ! videoconvert ! capsfilter ! appsink(4)
    #
    # which caps it at 4 and leaves the rest of the pool to the decoder.
    #
    # If negotiation ever does fail here, the node comes back as
    # `graph.add(pyneat.nodes.caps_raw("NV12", -1, -1, -1, pyneat.CapsMemory.Any))`
    # -- format only. Passing width, height or fps instead would add
    # `framerate=F/1`, and a raw elementary stream has no container to state its
    # rate, so h264parse publishes `framerate=0/1`, which intersects with
    # nothing. That fails silently: zero frames and a pull timeout, with nothing
    # on the bus to explain it.
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
        graph: The segmenter ``Graph``.
        run: Live ``Run`` handle for the segmenter graph.
        video_graph: Separate graph that encodes frames for Insight.
        video_run: Live ``Run`` handle for the video graph.
        metadata_sender: ``MetadataSender`` publishing instances as JSON.
        labels: Class names, indexed by class id.
        keep_ids: Class ids that count as foreground, or None for every class.
        frame_w: Source frame width in pixels.
        frame_h: Source frame height in pixels.
        fps: Source frame rate.
        source_frames: Coded pictures in the source file, counted before the
            run. 0 for a live source, where there is no such number.
        net_w: Model input width, used to invert the letterbox for masks.
        net_h: Model input height.
        video_port: Resolved UDP port for the Insight video feed.
        writer: OpenCV ``VideoWriter`` for the on-device recording.
        writer_path: Path the writer actually opened, after any fallback.
        writer_frames: Frames written so far.
        video_dropped: Preview frames the Insight feed refused.
        mask_kind: What the mask decoder settled on: ``proto``, ``planes`` or
            ``none``. Filled in on the first frame that carries instances.
        described: Whether the first-frame tensor dump has already been printed.
        warned_no_masks: Whether the "no mask data" warning has been printed.
    """

    model: object
    graph: object
    run: object
    video_graph: object = None
    video_run: object = None
    metadata_sender: object = None
    labels: list[str] = field(default_factory=list)
    keep_ids: object = None
    frame_w: int = 0
    frame_h: int = 0
    fps: int = 0
    source_frames: int = 0
    net_w: int = 0
    net_h: int = 0
    video_port: int = 0
    writer: object = None
    writer_path: str = ""
    writer_frames: int = 0
    video_dropped: int = 0
    mask_kind: str = ""
    mask_space: str = ""
    described: bool = False
    warned_no_masks: bool = False

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


def resolve_keep_classes(cfg: AppConfig, labels: list[str]):
    """Turn ``blur.keep_classes`` into a set of class ids.

    Accepts either names as they appear in the labels file or bare numeric ids,
    so ``[person, 2]`` is valid. A name that is not in the labels file is a
    config error, not a silently empty filter: the alternative is a run that
    blurs the whole frame and gives no clue why.

    Args:
        cfg: Application configuration.
        labels: Class names indexed by class id.

    Returns:
        A set of class ids, or None when every class counts as foreground.
    """
    if not cfg.blur.keep_classes:
        return None
    lookup = {name.lower(): index for index, name in enumerate(labels)}
    ids = set()
    for token in cfg.blur.keep_classes:
        text = token.strip()
        if text.isdigit():
            index = int(text)
            if not 0 <= index < len(labels):
                raise ValueError(
                    f"blur.keep_classes has class id {index}, but the labels file "
                    f"only has {len(labels)} classes (0 to {len(labels) - 1})"
                )
            ids.add(index)
            continue
        index = lookup.get(text.lower())
        if index is None:
            near = [name for name in labels if text.lower() in name.lower()][:5]
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            raise ValueError(
                f"blur.keep_classes has unknown class {text!r}, which is not in "
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


def build_segmenter_graph(cfg: AppConfig, model, width: int, height: int, fps: int):
    """source -> branch -> {frame, model->instances} -> combine("segmenter_output")."""
    source = make_source_graph(cfg, width, height, fps)
    branch = pyneat.graphs.branch("source", ["frame", "model"])

    frame_graph = pyneat.Graph("frame")
    frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(cfg.output_buffers)))

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)

    instances_graph = pyneat.Graph("instances")
    instances_graph.add(
        pyneat.nodes.output("instances", pyneat.OutputOptions.every_frame(cfg.output_buffers))
    )

    joined = pyneat.graphs.combine(
        ["frame", "instances"], "segmenter_output", pyneat.CombinePolicy.ByFrame
    )

    graph = pyneat.Graph("yolo_segmenter")
    graph.connect(source, branch)
    graph.connect(branch, frame_graph)
    graph.connect(branch, model_graph)
    graph.connect(model_graph, instances_graph)
    graph.connect(frame_graph, joined)
    graph.connect(instances_graph, joined)
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
    # never reaches its next pull(), and the whole segmenter graph stalls. The
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


def resolve_net_size(cfg: AppConfig) -> tuple[int, int]:
    """Model input geometry, needed to map a mask back onto the frame.

    Boxes come back in original-image pixels because Preproc writes the resize
    and letterbox metadata onto the tensor and BoxDecode reads it. Masks do not
    get that treatment: they live in the network's own letterboxed space, so the
    app has to invert the letterbox itself, and that needs the network input
    size. ``preprocess.resize`` supplies it when set; otherwise the mask
    geometry does, once the first frame arrives.

    Args:
        cfg: Application configuration.

    Returns:
        A ``(width, height)`` pair, or ``(0, 0)`` when it must wait for a mask.
    """
    if cfg.segment.net_width and cfg.segment.net_height:
        return cfg.segment.net_width, cfg.segment.net_height
    pre = cfg.preprocess
    if pre.resize_width and pre.resize_height:
        return pre.resize_width, pre.resize_height
    return 0, 0


def build_pipeline(cfg: AppConfig) -> Pipeline:
    step = lambda msg: print(msg, flush=True)

    check_source_file(cfg)
    width, height, fps = resolve_source_geometry(cfg)

    # Counted up front so the end of the run can say "83 of 379" rather than
    # leaving a short recording to be interpreted. Only a file has the number.
    source_frames = (
        count_h264_pictures(cfg.source_uri)
        if cfg.source_type == "video" and is_elementary_h264(cfg.source_uri)
        else 0
    )
    length = (
        f" frames={source_frames} ({source_frames / fps:.1f}s)"
        if source_frames and fps
        else f" frames={source_frames}" if source_frames
        else ""
    )
    step(f"source: type={cfg.source_type} uri={cfg.source_uri or '<default camera>'} "
         f"stream={width}x{height}@{fps}{length}")
    step(describe_preprocess(cfg, width, height))

    step("loading model (first load unpacks the archive, this can take a minute)...")
    model = make_model(cfg, width, height)
    labels = load_labels(cfg.labels_path)
    step(
        f"model: {cfg.model_path} family={cfg.family} "
        f"decode_type={FAMILY_DECODE_TOKENS[cfg.family]} labels={len(labels)}"
    )

    keep_ids = resolve_keep_classes(cfg, labels)
    net_w, net_h = resolve_net_size(cfg)
    step(
        f"segmentation: masks={cfg.segment.masks} source={cfg.segment.source} "
        f"space={cfg.segment.space} threshold={cfg.segment.threshold} "
        f"net={f'{net_w}x{net_h}' if net_w else '<from the first mask>'}"
    )
    step(describe_blur(cfg))

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
    graph = build_segmenter_graph(cfg, model, width, height, fps)
    if cfg.profile:
        step(f"Backend:\n{graph.describe_backend()}")
    run = graph.build(make_run_options(cfg))
    step("graph built")

    pipeline = Pipeline(
        model=model, graph=graph, run=run, labels=labels, keep_ids=keep_ids,
        frame_w=width, frame_h=height, fps=fps, source_frames=source_frames,
        net_w=net_w, net_h=net_h,
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
#
# A segment head emits that same tensor plus mask data in one or more further
# tensors, which is what the mask decoder below picks apart.
# ─────────────────────────────────────────────────────────────────────────────

BBOX_RECORD = struct.Struct("<iiiifi")
BBOX_RECORD_SIZE = BBOX_RECORD.size  # 24

# Tags a BBOX-carrying tensor is allowed to have. A segment head may label its
# box tensor differently from a detect head, and refusing an unfamiliar-but-
# correct tag would be worse than accepting it.
BBOX_TAGS = {"BBOX", "BBOXSEG", "BBOX_SEG", "BBOXSEGMENT", "DETECTION", "DETECTIONS"}


def sample_payload_tag(sample, tensor) -> str:
    """Best-effort payload tag for one tensor, upper-cased. Empty when unknown."""
    tag = getattr(sample, "payload_tag", "") or getattr(sample, "format", "")
    if not tag:
        semantic = getattr(tensor, "semantic", None)
        tess = getattr(semantic, "tess", None)
        if tess is not None:
            tag = getattr(tess, "format", "")
    return str(tag or "").upper()


def tensor_bbox_payload(sample, tensor=None) -> bytes:
    tensor = tensor if tensor is not None else getattr(sample, "tensor", None)
    if tensor is None:
        raise RuntimeError("instance sample carries no tensor")
    fmt = sample_payload_tag(sample, tensor)
    if fmt and fmt not in BBOX_TAGS:
        raise RuntimeError(
            f"expected a BBOX tensor but got {fmt}. If this is `raw_output_heads`, "
            "the route did not include BoxDecode. Check model.family and the model archive."
        )
    payload = tensor.copy_payload_bytes()
    if not payload:
        raise RuntimeError("empty BBOX payload")
    return payload


def extract_bbox_payload(sample) -> tuple[bytes, object]:
    """Find the BBOX tensor in a sample tree.

    Returns:
        A ``(payload, tensor)`` pair. The tensor is handed back so the mask
        decoder can exclude it when it goes looking for mask data.
    """
    if sample.kind == pyneat.SampleKind.Bundle:
        for candidate in sample.fields:
            try:
                return extract_bbox_payload(candidate)
            except RuntimeError:
                continue
        raise RuntimeError("bundle has no BBOX field")
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        # A segment head packs several tensors into one set. The box tensor is
        # normally first, but scan rather than assume: an unexpected order would
        # otherwise be parsed as boxes and produce garbage coordinates.
        for tensor in sample.tensors:
            try:
                return tensor_bbox_payload(sample, tensor), tensor
            except RuntimeError:
                continue
        raise RuntimeError("tensor set has no BBOX tensor")
    if sample.kind != pyneat.SampleKind.Tensor:
        raise RuntimeError(f"unexpected sample kind {sample.kind}")
    return tensor_bbox_payload(sample), sample.tensor


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
# Mask payload
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MaskBundle:
    """Mask data for one frame, already reduced to numpy.

    Attributes:
        kind: ``proto``, ``planes`` or ``none``.
        protos: Prototype bank, ``(C, mh, mw)`` float32. Only for ``proto``.
        coeffs: Per-instance coefficients, ``(N, C)`` float32. Only for ``proto``.
        planes: Per-instance masks, ``(N, mh, mw)``. Only for ``planes``.
        probabilities: Whether the values are already in 0..1 (or 0..255). When
            False they are treated as logits and compared against
            ``logit(threshold)``.
        origin: Where the data came from, ``packed`` or ``tensors``. Reporting
            only; the decode path is the same once it is in numpy.
        peak: Largest value seen across the used planes, which separates a
            0/1 binary mask from a 0..255 quantised one.
    """

    kind: str = "none"
    protos: object = None
    coeffs: object = None
    planes: object = None
    probabilities: bool = False
    origin: str = ""
    peak: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        """Mask-space ``(height, width)``, or ``(0, 0)`` when there is no mask."""
        if self.kind == "proto":
            return int(self.protos.shape[1]), int(self.protos.shape[2])
        if self.kind == "planes":
            return int(self.planes.shape[1]), int(self.planes.shape[2])
        return 0, 0


def iter_tensors(sample):
    """Yield every ``(sample, tensor)`` pair in a sample tree, depth first."""
    kind = getattr(sample, "kind", None)
    if kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample, sample.tensor
    elif kind == pyneat.SampleKind.TensorSet:
        for tensor in sample.tensors or []:
            yield sample, tensor
    for candidate in getattr(sample, "fields", []) or []:
        yield from iter_tensors(candidate)


def describe_tensors(sample) -> str:
    """One line per tensor: tag, dtype, shape and byte length.

    Printed once, on the first frame that carries instances. Which tensors a
    segment head emits, and in what shape, is the one thing this app cannot know
    ahead of time, so it reports what actually arrived instead of guessing
    silently. Everything the mask decoder needs to be pinned in ``config.yaml``
    is visible in this dump.
    """
    lines = []
    for index, (owner, tensor) in enumerate(iter_tensors(sample)):
        tag = sample_payload_tag(owner, tensor) or "<untagged>"
        label = getattr(owner, "stream_label", "") or "<unlabelled>"
        try:
            array = np.asarray(tensor.to_numpy(copy=False))
            shape, dtype = tuple(int(d) for d in array.shape), str(array.dtype)
        except Exception as exc:
            shape, dtype = f"<to_numpy failed: {exc}>", "?"
        try:
            nbytes = len(tensor.copy_payload_bytes())
        except Exception:
            nbytes = -1
        lines.append(
            f"  [{index}] stream={label} tag={tag} dtype={dtype} shape={shape} bytes={nbytes}"
        )
    return "\n".join(lines) or "  <no tensors>"


def squeeze_leading(array):
    """Drop leading length-1 axes, so ``(1, 32, 160, 160)`` becomes ``(32, 160, 160)``."""
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return array


def classify_mask_tensors(arrays, count: int, seg: SegmentConfig) -> MaskBundle:
    """Work out which of the leftover tensors are protos, coefficients or planes.

    Args:
        arrays: Every non-BBOX tensor from the instances field, as numpy arrays.
        count: How many instances the BBOX tensor reported.
        seg: Segmentation settings, for ``source`` and ``coeff_counts``.

    Returns:
        A populated :class:`MaskBundle`, or one with ``kind="none"`` when
        nothing recognisable was found.
    """
    planes = protos = coeffs = None
    for array in arrays:
        array = squeeze_leading(array)
        if array.ndim == 3:
            # A plane bank has one plane per instance; a prototype bank has one
            # per prototype, and those counts only collide by coincidence. When
            # they do, the configured coeff_counts breaks the tie.
            if count > 0 and array.shape[0] == count and array.shape[0] not in seg.coeff_counts:
                planes = array
            elif array.shape[0] in seg.coeff_counts:
                protos = array
            elif count > 0 and array.shape[0] == count:
                planes = array
        elif array.ndim == 2:
            if count > 0 and array.shape == (count, count):
                continue  # ambiguous, and no real model emits this
            if count > 0 and array.shape[0] == count and array.shape[1] in seg.coeff_counts:
                coeffs = array
            elif count > 0 and array.shape[1] == count and array.shape[0] in seg.coeff_counts:
                coeffs = array.T

    if seg.source in {"auto", "planes"} and planes is not None:
        planes = np.ascontiguousarray(planes)
        return MaskBundle(
            kind="planes",
            planes=np.array(planes, copy=True),
            probabilities=plane_values_are_probabilities(planes),
            origin="tensors",
            # Same 0/1-versus-0..255 question as the packed path, so answer it
            # the same way rather than defaulting and thresholding to nothing.
            peak=int(planes[:count].max()) if count and planes.shape[0] >= count else 0,
        )
    if seg.source in {"auto", "proto"} and protos is not None and coeffs is not None:
        return MaskBundle(
            kind="proto",
            protos=np.array(protos, dtype=np.float32, copy=True),
            coeffs=np.array(coeffs, dtype=np.float32, copy=True),
            probabilities=False,
            origin="tensors",
        )
    return MaskBundle()


def plane_values_are_probabilities(planes) -> bool:
    """Whether a plane bank is already 0..1 (or 0..255) rather than raw logits.

    Sampling the first plane is enough: a bank is homogeneous, and reducing over
    all of them every frame would cost more than the masks themselves.
    """
    if planes.dtype == np.uint8:
        return True
    sample = planes[0] if planes.shape[0] else planes
    return bool(sample.min() >= 0.0 and sample.max() <= 1.0)


MIN_MASK_SIDE = 16
MAX_MASK_SIDE = 1024


def packed_mask_layout(payload_len: int, top_k: int, seg: SegmentConfig):
    """Solve a BBOX buffer that carries its masks in its own tail.

    ``neatobjectdecode`` emits one UInt8 tensor per frame for a segment head::

        [uint32 count][RawBox 24B * slots][mask side*side uint8 * slots]

    ``slots`` is the top-K capacity rather than the number of detections, so the
    buffer is the same size on every frame and the arithmetic below is exact
    rather than a guess. With ``top_k=50`` and a 640-input model:
    ``4 + 50*24 + 50*160*160 = 1281204``.

    Args:
        payload_len: Total bytes in the BBOX tensor.
        top_k: The configured ``decode.max_detections``, or 0 when the archive's
            own value is in force and we do not know it.
        seg: Segmentation settings, for ``mask_sides``.

    Returns:
        A ``(slots, side)`` pair, or None when the length does not decompose.
    """
    # The top-K we asked for is the answer whenever it was honoured, so try it
    # before searching. One exact hit beats a table of plausible ones.
    if top_k > 0:
        tail = payload_len - 4 - top_k * BBOX_RECORD_SIZE
        if tail > 0 and tail % top_k == 0:
            per = tail // top_k
            side = math.isqrt(per)
            if side * side == per and MIN_MASK_SIDE <= side <= MAX_MASK_SIDE:
                return top_k, side

    # top_k unknown, or the decoder used its own. Solve for the slot count from
    # each plausible mask edge instead; a wrong side almost never divides.
    for side in seg.mask_sides:
        per = side * side + BBOX_RECORD_SIZE
        if (payload_len - 4) % per == 0:
            slots = (payload_len - 4) // per
            if slots > 0:
                return slots, side
    return None


def masks_from_packed_payload(payload: bytes, count: int, top_k: int, seg: SegmentConfig):
    """Read per-instance masks out of the tail of the BBOX buffer."""
    layout = packed_mask_layout(len(payload), top_k, seg)
    if layout is None:
        return MaskBundle()
    slots, side = layout
    if slots < count:
        return MaskBundle()
    offset = 4 + slots * BBOX_RECORD_SIZE
    planes = np.frombuffer(
        payload, dtype=np.uint8, count=slots * side * side, offset=offset
    ).reshape(slots, side, side)
    # Only the first `count` slots hold a detection; the rest are unwritten
    # capacity, so peak is measured over the used ones alone.
    peak = int(planes[:count].max()) if count else 0
    return MaskBundle(
        kind="planes", planes=planes, probabilities=True, origin="packed", peak=peak
    )


def extract_masks(
    sample, bbox_tensor, payload: bytes, count: int, seg: SegmentConfig, top_k: int
) -> MaskBundle:
    """Pull mask data out of the instances field of one joined sample.

    Packed first, because that is what the shipped SiMa model packs emit and it
    needs no tensor sniffing at all: the byte length either decomposes or it
    does not. Separate tensors are the fallback.
    """
    if seg.masks != "on" or count <= 0:
        return MaskBundle()

    if seg.source in {"auto", "packed"}:
        bundle = masks_from_packed_payload(payload, count, top_k, seg)
        if bundle.kind != "none":
            return bundle
        if seg.source == "packed":
            return MaskBundle()

    arrays = []
    for owner, tensor in iter_tensors(sample):
        if tensor is bbox_tensor:
            continue
        try:
            arrays.append(np.asarray(tensor.to_numpy(copy=False)))
        except Exception:
            # A tessellated or otherwise opaque tensor cannot become an array.
            # It is not mask data we can use, so skip it rather than fail.
            continue
    return classify_mask_tensors(arrays, count, seg)


@dataclass(frozen=True)
class Letterbox:
    """Affine map from frame pixels to network pixels.

    ``nx = fx * sx + pad_x`` and ``ny = fy * sy + pad_y``. Letterbox and crop
    share one scale; stretch has a different one per axis and no padding.
    """

    sx: float
    sy: float
    pad_x: float
    pad_y: float


def letterbox_transform(frame_w: int, frame_h: int, net_w: int, net_h: int, mode: str) -> Letterbox:
    """Recreate what Preproc did to the frame before the model saw it.

    Boxes need none of this: Preproc records the transform on the tensor and
    BoxDecode undoes it, which is why detections arrive in source-image pixels.
    Masks are produced in the network's own coordinate space, so this rebuilds
    the same transform in order to invert it.

    Args:
        frame_w: Source frame width.
        frame_h: Source frame height.
        net_w: Model input width.
        net_h: Model input height.
        mode: ``letterbox``, ``stretch`` or ``crop``, from ``preprocess.resize``.

    Returns:
        A :class:`Letterbox` mapping frame pixels to network pixels.
    """
    if mode == "stretch":
        return Letterbox(net_w / frame_w, net_h / frame_h, 0.0, 0.0)
    # letterbox fits the whole frame inside the net and pads the shortfall;
    # crop fills the net and spills over the edges. Same formula, min vs max,
    # and crop simply produces negative padding.
    pick = max if mode == "crop" else min
    scale = pick(net_w / frame_w, net_h / frame_h)
    return Letterbox(
        scale,
        scale,
        (net_w - frame_w * scale) / 2.0,
        (net_h - frame_h * scale) / 2.0,
    )


def instance_plane(bundle: MaskBundle, index: int):
    """The mask-space plane for one instance, as float32.

    For ``proto`` this is the coefficient-weighted sum of the prototype bank,
    left as logits. Thresholding a logit at ``logit(t)`` is the same decision as
    thresholding its sigmoid at ``t``, and sigmoid over every prototype-sized
    plane, every frame, is real work for no change in the result.
    """
    if bundle.kind == "planes":
        return bundle.planes[index]
    protos = bundle.protos
    channels, mask_h, mask_w = protos.shape
    flat = protos.reshape(channels, mask_h * mask_w)
    return (bundle.coeffs[index] @ flat).reshape(mask_h, mask_w)


def plane_cutoff(bundle: MaskBundle, threshold: float) -> float:
    """The value to compare a plane against, in that plane's own units.

    A uint8 plane is either a 0/1 binary mask or a 0..255 quantised
    probability, and the two need cut-offs 255x apart. The observed peak tells
    them apart, so a binary mask is not silently thresholded away to nothing.
    """
    if not bundle.probabilities:
        return math.log(threshold / (1.0 - threshold))     # logit
    if bundle.kind == "planes" and bundle.planes.dtype == np.uint8:
        return threshold * (255.0 if bundle.peak > 1 else 1.0)
    return threshold


def warp_plane_to_box(plane, lb: Letterbox, net_w: int, net_h: int, box: tuple[int, int, int, int]):
    """Scale one mask-space plane into a box-sized crop in frame pixels.

    Resampling only the box region rather than the whole frame is what keeps
    this affordable: a 1080p frame with 20 instances would otherwise mean 20
    full-frame resizes per frame, and every one of them would be almost entirely
    empty. The composed map is a pure scale plus translation, so a single
    inverse-mapped affine warp is exact.

    Args:
        plane: ``(mh, mw)`` float32 mask-space plane.
        lb: Frame-to-network transform.
        net_w: Model input width.
        net_h: Model input height.
        box: ``(x1, y1, x2, y2)`` in frame pixels.

    Returns:
        A ``(y2 - y1, x2 - x1)`` float32 array in the plane's own units.
    """
    x1, y1, x2, y2 = box
    mask_h, mask_w = plane.shape
    # frame px -> network px -> mask px, then shift by half a pixel because
    # OpenCV samples at pixel centres.
    ax = lb.sx * mask_w / net_w
    ay = lb.sy * mask_h / net_h
    bx = ((x1 + 0.5) * lb.sx + lb.pad_x) * mask_w / net_w - 0.5
    by = ((y1 + 0.5) * lb.sy + lb.pad_y) * mask_h / net_h - 0.5
    matrix = np.array([[ax, 0.0, bx], [0.0, ay, by]], dtype=np.float32)
    return cv2.warpAffine(
        np.ascontiguousarray(plane, dtype=np.float32),
        matrix,
        (x2 - x1, y2 - y1),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def resize_plane_to_box(plane, box: tuple[int, int, int, int]):
    """Scale a whole plane into the box, for masks already cropped to it."""
    x1, y1, x2, y2 = box
    return cv2.resize(
        np.ascontiguousarray(plane, dtype=np.float32),
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_LINEAR,
    )


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_mask_space(plane, cutoff: float, lb: Letterbox, net_w: int, net_h: int,
                      box: tuple[int, int, int, int]) -> str:
    """Work out whether a plane covers the whole network input or just its box.

    Both conventions exist and they are indistinguishable from shape alone, but
    not from content: map the detection's own box into the plane and compare it
    with where the ink actually is. A network-space mask lines up; a
    box-cropped one fills the plane no matter where its box sits.

    Args:
        plane: One mask-space plane.
        cutoff: Threshold in the plane's own units.
        lb: Frame-to-network transform.
        net_w: Model input width.
        net_h: Model input height.
        box: ``(x1, y1, x2, y2)`` in frame pixels.

    Returns:
        ``net``, ``box``, or ``""`` when the plane is empty and tells us nothing.
    """
    rows = np.nonzero((plane > cutoff).any(axis=1))[0]
    cols = np.nonzero((plane > cutoff).any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return ""
    mask_h, mask_w = plane.shape
    seen = (
        cols[0] / mask_w, rows[0] / mask_h,
        (cols[-1] + 1) / mask_w, (rows[-1] + 1) / mask_h,
    )
    x1, y1, x2, y2 = box
    expected = (
        (x1 * lb.sx + lb.pad_x) / net_w, (y1 * lb.sy + lb.pad_y) / net_h,
        (x2 * lb.sx + lb.pad_x) / net_w, (y2 * lb.sy + lb.pad_y) / net_h,
    )
    return "net" if _iou(seen, expected) >= 0.30 else "box"


@dataclass
class Instance:
    """One detected object, with its mask already in frame coordinates.

    Attributes:
        box: The raw detection dict, for captions and metadata.
        x1: Left edge in frame pixels, clipped to the frame.
        y1: Top edge.
        x2: Right edge, exclusive.
        y2: Bottom edge, exclusive.
        mask: Boolean ``(y2 - y1, x2 - x1)`` array, or None when the run is
            falling back to box-shaped regions.
        keep: Whether this instance counts as foreground for the blur.
    """

    box: dict
    x1: int
    y1: int
    x2: int
    y2: int
    mask: object = None
    keep: bool = True

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def mask_area(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else self.area


def as_cv_mask(mask):
    """View a boolean mask as the uint8 mask OpenCV wants, without copying.

    ``np.bool_`` is one byte holding 0 or 1, and every OpenCV mask argument
    treats non-zero as set, so the view is free and the values already mean the
    right thing.
    """
    return mask.view(np.uint8) if mask.flags["C_CONTIGUOUS"] else mask.astype(np.uint8)


def refine_mask(mask, seg: SegmentConfig, scale: float):
    """Smooth and grow a boolean mask according to the ``segmentation`` settings."""
    if seg.blur_mask > 0:
        k = max(3, int(round(seg.blur_mask * scale))) | 1
        smoothed = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (k, k), 0)
        mask = smoothed > 127
    if seg.dilate > 0:
        radius = max(1, int(round(seg.dilate * scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask


def build_instances(
    pipeline: Pipeline,
    cfg: AppConfig,
    boxes: list[dict],
    bundle: MaskBundle,
    frame_shape,
    scale: float,
) -> list[Instance]:
    """Turn boxes plus mask data into per-instance masks in frame coordinates.

    Args:
        pipeline: Live pipeline, for the frame geometry, the class filter and
            the one-shot fallback warning.
        cfg: Application configuration.
        boxes: Parsed detections, in source-image pixels.
        bundle: Mask data for this frame, possibly empty.
        frame_shape: The frame's ``(h, w, c)``.
        scale: Overlay scale factor for this frame.

    Returns:
        One :class:`Instance` per usable box, in input order.
    """
    height, width = frame_shape[:2]
    lb = None
    cutoff = 0.0
    if bundle.kind != "none":
        mask_h, mask_w = bundle.shape
        # A model pack that does not state its input size still tells us
        # indirectly: the mask is the input divided by the head's stride.
        net_w = pipeline.net_w or mask_w * cfg.segment.stride
        net_h = pipeline.net_h or mask_h * cfg.segment.stride
        pipeline.net_w, pipeline.net_h = net_w, net_h
        lb = letterbox_transform(width, height, net_w, net_h, cfg.preprocess.resize_mode)
        cutoff = plane_cutoff(bundle, cfg.segment.threshold)
    elif boxes and cfg.segment.masks == "on" and not pipeline.warned_no_masks:
        pipeline.warned_no_masks = True
        if cfg.segment.fallback_to_boxes:
            print(
                "[warn] no mask data in the model output, falling back to box-shaped\n"
                "       regions. The blur still works, the edges are just rectangles.\n"
                "       Set segmentation.describe: on and check the tensor dump above\n"
                "       against segmentation.source / coeff_counts.",
                file=sys.stderr, flush=True,
            )
        else:
            raise RuntimeError(
                "no mask data in the model output. Check model.family is a -seg head "
                "and segmentation.source, or set segmentation.fallback_to_boxes: on."
            )

    instances = []
    for index, box in enumerate(boxes):
        x1 = max(0, int(round(box["x1"])))
        y1 = max(0, int(round(box["y1"])))
        x2 = min(width, int(round(box["x2"])))
        y2 = min(height, int(round(box["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue

        mask = None
        if lb is not None and index < instance_count(bundle):
            plane = instance_plane(bundle, index)
            if not pipeline.mask_space:
                # Decided once, from the first instance that has any ink in it.
                space = cfg.segment.space
                if space == "auto":
                    space = detect_mask_space(
                        plane, cutoff, lb, pipeline.net_w, pipeline.net_h, (x1, y1, x2, y2)
                    )
                if space:
                    pipeline.mask_space = space
                    print(f"masks: space={space}", flush=True)
            local = (
                resize_plane_to_box(plane, (x1, y1, x2, y2))
                if pipeline.mask_space == "box"
                else warp_plane_to_box(
                    plane, lb, pipeline.net_w, pipeline.net_h, (x1, y1, x2, y2)
                )
            )
            mask = refine_mask(local > cutoff, cfg.segment, scale)
            if not mask.any():
                # A mask that thresholds to nothing would silently remove the
                # instance from the composite. The box is a worse answer than a
                # mask but a much better one than a hole.
                mask = None

        class_id = int(box["class_id"])
        keep = pipeline.keep_ids is None or class_id in pipeline.keep_ids
        instances.append(Instance(box=box, x1=x1, y1=y1, x2=x2, y2=y2, mask=mask, keep=keep))
    return instances


def instance_count(bundle: MaskBundle) -> int:
    if bundle.kind == "planes":
        return int(bundle.planes.shape[0])
    if bundle.kind == "proto":
        return int(bundle.coeffs.shape[0])
    return 0


def foreground_mask(instances: list[Instance], frame_shape):
    """Union of every kept instance, as a full-frame uint8 mask of 0 and 255."""
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for inst in instances:
        if not inst.keep:
            continue
        region = mask[inst.y1 : inst.y2, inst.x1 : inst.x2]
        if inst.mask is None:
            region[:] = 255
        else:
            # Union rather than assignment: overlapping instances must not
            # punch each other's pixels back out of the foreground.
            cv2.bitwise_or(region, 255, dst=region, mask=as_cv_mask(inst.mask))
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Compositing
# ─────────────────────────────────────────────────────────────────────────────


def blurred_background(frame, blur: BlurConfig, scale: float):
    """Build the treated copy of the frame that shows through outside the mask.

    Always returns a new array, never a view of ``frame``, because the caller
    writes the sharp pixels back into it.

    Args:
        frame: BGR source frame.
        blur: Background settings.
        scale: Overlay scale factor, so sizes tuned at 1080p hold at any height.

    Returns:
        A BGR image the same size as ``frame``.
    """
    height, width = frame.shape[:2]

    if blur.method == "pixelate":
        block = max(2, int(round(blur.pixel_size * scale)))
        small = cv2.resize(
            frame,
            (max(1, width // block), max(1, height // block)),
            interpolation=cv2.INTER_AREA,
        )
        out = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    elif blur.method == "gaussian":
        # Blurring at 1/N resolution is the whole reason this runs at frame
        # rate on 1080p. The downscale is itself a low-pass, so what comes back
        # is a blur either way; the kernel just has less area to cover.
        down = max(1, blur.downscale)
        small = (
            cv2.resize(
                frame,
                (max(1, width // down), max(1, height // down)),
                interpolation=cv2.INTER_AREA,
            )
            if down > 1
            else frame
        )
        kernel = max(1, int(round(blur.kernel * scale / down))) | 1
        small = cv2.GaussianBlur(small, (kernel, kernel), blur.sigma * scale / down)
        out = (
            cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
            if down > 1
            else small
        )
    else:
        out = frame.copy()

    if blur.grayscale:
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if blur.dim > 0:
        out = cv2.convertScaleAbs(out, alpha=1.0 - blur.dim, beta=0.0)

    # Opacity last, so one dial covers the blur, the desaturation and the
    # dimming together instead of needing three of them turned down in step.
    if blur.opacity < 1.0:
        out = cv2.addWeighted(out, blur.opacity, frame, 1.0 - blur.opacity, 0.0)
    return out


def composite(frame, mask, blur: BlurConfig, scale: float):
    """Keep the frame where the mask is set and the treated copy everywhere else.

    Args:
        frame: BGR source frame, left unmodified.
        mask: Full-frame uint8 foreground mask, 0 or 255.
        blur: Background settings.
        scale: Overlay scale factor.

    Returns:
        A new BGR image.
    """
    background = blurred_background(frame, blur, scale)
    if blur.invert:
        # Blur the instances instead: same composite, opposite mask.
        mask = cv2.bitwise_not(mask)

    feather = int(round(blur.feather * scale))
    if feather > 0:
        # A hard cut shows every jag a 160x160 mask has after scaling to 1080p.
        # Blurring the mask turns that edge into a short cross-fade, which is
        # both cheaper and better looking than trying to refine the mask itself.
        kernel = max(3, feather) | 1
        alpha = cv2.cvtColor(cv2.GaussianBlur(mask, (kernel, kernel), 0), cv2.COLOR_GRAY2BGR)
        # Blend in uint8 through OpenCV rather than promoting two 1080p frames
        # to float32 and back. The arithmetic is identical to within a rounding
        # step, and the float path costs more than the blur, the mask decode and
        # every other sink put together.
        return cv2.add(
            cv2.multiply(frame, alpha, scale=1.0 / 255.0),
            cv2.multiply(background, cv2.bitwise_not(alpha), scale=1.0 / 255.0),
        )

    # copyTo writes only where the mask is set and leaves the rest of the
    # background untouched, which boolean fancy indexing does far more slowly.
    return cv2.copyTo(frame, mask, background)


# ─────────────────────────────────────────────────────────────────────────────
# Frames and overlay
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
MASK_COLORS = [
    (56, 56, 255), (49, 210, 207), (10, 249, 72), (227, 195, 0), (255, 112, 132),
    (144, 31, 255), (29, 178, 255), (49, 121, 255), (0, 194, 255), (98, 205, 0),
    (185, 243, 52), (255, 156, 87), (255, 88, 178), (184, 61, 245), (86, 96, 255),
    (0, 151, 255), (0, 229, 178), (146, 255, 51), (255, 194, 26), (255, 92, 92),
]

FONT = 0  # replaced with cv2.FONT_HERSHEY_SIMPLEX by load_runtime_dependencies()

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
    return MASK_COLORS[class_id % len(MASK_COLORS)]


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
    """Build the caption for one instance.

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


def draw_instances(frame, instances: list[Instance], labels: list[str], draw: DrawConfig) -> None:
    """Tint, outline and caption every instance in place.

    Each instance gets a translucent fill in its class colour, a traced edge and
    a filled caption sitting directly above its box. The caption flips inside
    the box when it would otherwise clip off the top of the frame.

    Args:
        frame: BGR image, modified in place.
        instances: Instances with masks already in frame coordinates.
        labels: Class names indexed by class id.
        draw: Visualization settings, from the ``visualization`` config section.
    """
    height, width = frame.shape[:2]
    scale = draw_scale(frame, draw)
    box_thickness = max(1, int(round(draw.box_thickness * scale)))
    outline_thickness = max(1, int(round(draw.outline_thickness * scale)))
    text_scale = draw.text_scale * scale
    text_thickness = max(1, int(round(draw.text_thickness * scale)))
    pad = max(2, int(round(draw.text_padding * scale)))

    # Paint larger instances first, so small foreground objects stay legible.
    for inst in sorted(instances, key=lambda i: i.area, reverse=True):
        color = class_color(int(inst.box["class_id"]))
        region = frame[inst.y1 : inst.y2, inst.x1 : inst.x2]

        if draw.mask_alpha > 0 and region.size:
            alpha = draw.mask_alpha
            tinted = cv2.addWeighted(
                region, 1.0 - alpha, np.full_like(region, color, dtype=np.uint8), alpha, 0.0
            )
            if inst.mask is None:
                region[:] = tinted
            else:
                cv2.copyTo(tinted, as_cv_mask(inst.mask), region)

        if draw.mask_outline and inst.mask is not None:
            contours, _ = cv2.findContours(
                as_cv_mask(inst.mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                frame, contours, -1, color, outline_thickness, cv2.LINE_AA,
                offset=(inst.x1, inst.y1),
            )
        if draw.show_boxes or inst.mask is None:
            cv2.rectangle(frame, (inst.x1, inst.y1), (inst.x2, inst.y2), color, box_thickness)

        label = caption_text(inst.box, labels, draw)
        if not label:
            continue

        (text_w, _), _ = cv2.getTextSize(label, FONT, text_scale, text_thickness)
        above, below = text_ink_extent(label, text_scale, text_thickness)
        band_w = text_w + pad * 2
        band_h = above + below + pad * 2

        top = inst.y1 - band_h
        if top < 0:                      # would clip off the top, so sit inside
            top = inst.y1
        left = max(0, min(inst.x1, width - band_w))

        cv2.rectangle(frame, (left, top), (left + band_w, top + band_h), color, -1)
        cv2.putText(
            frame, label, (left + pad, top + pad + above), FONT, text_scale,
            draw.text_color, text_thickness, cv2.LINE_AA,
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


# ─────────────────────────────────────────────────────────────────────────────
# Sinks
# ─────────────────────────────────────────────────────────────────────────────


def metadata_objects(instances: list[Instance], labels: list[str]) -> list[dict]:
    objects = []
    for index, inst in enumerate(instances, start=1):
        class_id = int(inst.box["class_id"])
        objects.append(
            {
                "id": f"obj_{index}",
                "label": labels[class_id] if 0 <= class_id < len(labels) else "unknown",
                "confidence": float(inst.box["score"]),
                "bbox": [
                    float(inst.x1),
                    float(inst.y1),
                    float(inst.x2 - inst.x1),
                    float(inst.y2 - inst.y1),
                ],
                # Pixels the mask actually covers, which is what separates a
                # thin diagonal object from the box that contains it.
                "mask_area": inst.mask_area,
                "foreground": bool(inst.keep),
            }
        )
    return objects


def send_metadata(pipeline: Pipeline, stamp: FrameStamp, instances: list[Instance]) -> None:
    if pipeline.metadata_sender is None:
        return
    timestamp_ms = int(stamp.pts_ns // 1_000_000) if stamp.pts_ns >= 0 else -1
    frame_id = str(stamp.frame_id) if stamp.frame_id >= 0 else ""
    pipeline.metadata_sender.send_metadata(
        "instance-segmentation",
        json.dumps(
            {"objects": metadata_objects(instances, pipeline.labels)},
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


def render_annotated(cfg: AppConfig, pipeline: Pipeline, frame, instances, fps: float):
    """Composite the blur, then draw over it, once per frame for every sink."""
    scale = draw_scale(frame, cfg.draw)
    if cfg.blur.enable:
        annotated = composite(frame, foreground_mask(instances, frame.shape), cfg.blur, scale)
    else:
        annotated = frame.copy()
    # FPS first, so an instance in the top-left corner is never hidden by it.
    if cfg.video_hud:
        draw_fps(annotated, fps, cfg.draw)
    draw_instances(annotated, instances, pipeline.labels, cfg.draw)
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

    Sums pull, decode, mask and sink latencies over a fixed number of frames,
    then prints one averaged line and resets. Averaging avoids the noise of
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
        self.instances = 0
        self.start_ms = 0.0
        self.pull_ms = 0.0
        self.decode_ms = 0.0
        self.mask_ms = 0.0
        self.sink_ms = 0.0

    def add(
        self,
        pull_ms: float,
        decode_ms: float,
        mask_ms: float,
        sink_ms: float,
        instance_count: int,
    ) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.instances += instance_count
        self.pull_ms += pull_ms
        self.decode_ms += decode_ms
        self.mask_ms += mask_ms
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
            f"masks={self.mask_ms / self.frames:.1f}ms "
            f"sinks={self.sink_ms / self.frames:.1f}ms "
            f"instances={self.instances / self.frames:.1f}",
            flush=True,
        )
        self.reset()


@dataclass
class SinkJob:
    """One finished frame, handed to the sink thread.

    Every field is plain numpy or plain Python. Nothing here references a
    ``pyneat`` sample, so the decoder's buffer is already back in its pool by
    the time a job is queued.

    Attributes:
        index: 1-based frame number, used for the stills filename and ``every``.
        stamp: Timing fields copied out of the source sample.
        frame: Untouched BGR frame.
        instances: Instances detected on it.
        fps: Rate to print in the HUD badge.
    """

    index: int
    stamp: FrameStamp
    frame: object
    instances: list
    fps: float


class SinkWorker:
    """Runs compositing, stills and the recording on a thread of its own.

    The pull loop used to blur, draw, JPEG-encode and write the video before
    asking for the next frame. All of that is pure numpy and needs no buffer
    from the decoder, but it still gated ``pull()``, and a consumer that pauses
    for a few hundred milliseconds per frame is what lets decoded frames pile up
    in the queues between the decoder and the app. The pool is eight buffers on
    this board, so a big enough pile starves the decoder, which then cannot
    produce the frame that would release the pile: the run stops with a pull
    timeout part-way through the clip.

    Moving that work here means the loop pulls again immediately, so buffers go
    back at the rate the decoder can reuse them. Ordering is preserved because
    there is exactly one worker draining a FIFO, so the recording still has the
    source's frame order.

    The queue is bounded. When the sinks fall behind, ``submit`` blocks, which is
    the backpressure that keeps memory flat -- but it blocks after ``depth``
    frames of slack rather than on every single one.

    Attributes:
        error: First exception raised on the worker, re-raised by ``close``.
        blocked_ms: Total time ``submit`` spent waiting for a free slot.
    """

    def __init__(self, cfg: AppConfig, pipeline: Pipeline, depth: int) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self.queue: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self.error: BaseException | None = None
        self.blocked_ms = 0.0
        self.thread = threading.Thread(target=self._run, name="sinks", daemon=True)
        self.thread.start()

    def submit(self, job: SinkJob) -> None:
        start = time_ms()
        self.queue.put(job)
        self.blocked_ms += time_ms() - start

    def drain(self) -> None:
        """Block until every queued frame has been written."""
        self.queue.join()

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            try:
                if job is None:
                    return
                self._handle(job)
            except BaseException as exc:  # noqa: BLE001 - reported by close()
                if self.error is None:
                    self.error = exc
            finally:
                self.queue.task_done()

    def _handle(self, job: SinkJob) -> None:
        cfg, pipeline = self.cfg, self.pipeline
        need_jpeg = wants_jpeg(cfg, job.index)
        need_annotated = (
            pipeline.writer is not None
            or (cfg.insight_enable and cfg.insight_annotated)
            or (need_jpeg and cfg.save_overlay)
        )

        # Render once, then share the result across every sink that wants it.
        annotated = (
            render_annotated(cfg, pipeline, job.frame, job.instances, job.fps)
            if need_annotated
            else None
        )

        # With insight_annotated the viewer shows our composite. Without it,
        # Insight receives the raw frame and draws its own overlay from the
        # metadata, which has boxes but no masks.
        push_video(
            pipeline, job.stamp,
            annotated if (cfg.insight_annotated and annotated is not None) else job.frame,
        )
        send_metadata(pipeline, job.stamp, job.instances)

        if pipeline.writer is not None:
            pipeline.writer.write(annotated)
            pipeline.writer_frames += 1
        if need_jpeg:
            save_frame(cfg, job.index, annotated if cfg.save_overlay else job.frame)


HEARTBEAT_EVERY = 50


def source_stopped_message(cfg: AppConfig, pipeline: Pipeline, processed: int) -> str:
    """Explain a source that went quiet, ruling out what the frame count rules out.

    The clip's length is known before the run starts, so "it just ended" is
    either the whole answer or not on the list at all. Saying which turns a
    short recording from something to be interpreted into something decided.
    """
    total = pipeline.source_frames
    head = (
        f"source produced nothing for {cfg.pull_timeout_ms} ms twice in a row after "
        f"{processed} frames"
    )
    if total and processed >= total:
        return (
            f"{head}, which is the whole clip ({total} frames). Nothing is wrong: "
            "the run is complete."
        )

    if total:
        head += (
            f", {processed / total:.0%} of the way through a {total} frame clip. "
            "The source stalled; it did not end."
        )
    else:
        head += ". If that is far short of the clip, the source stalled rather than ended."

    return (
        f"{head}\nIn order of likelihood:\n"
        "  1. the hardware decoder ran out of buffers. Its pool is small (the boot log\n"
        "     prints BufferNum), and every element between it and the source appsink\n"
        "     can park one. Count the queues in the first pipeline printed above:\n"
        "     their max-buffers plus the appsink's must stay under BufferNum. Then\n"
        "     lower runtime.output_buffers, which costs two more.\n"
        "  2. output.insight.enable is on and its encoder wedged the shared codec\n"
        "     daemon.\n"
        "Run again with --minimal to tell 1 apart from how much work this app does\n"
        "per frame: the same stall means the graph, a complete run means the app."
    )


def pull_frame(pipeline: Pipeline, cfg: AppConfig, sinks: SinkWorker, processed: int):
    """Pull one joined sample, flushing our own backlog before giving up.

    A starved decoder and a finished clip look identical from here: both are
    silence. So on the first timeout, hand back everything the app is still
    holding -- the sink queue is several decoded frames deep -- and ask again. A
    pool that refills answers straight away; a clip that ended stays quiet.

    Args:
        pipeline: Live pipeline.
        cfg: Application configuration, for ``pull_timeout_ms``.
        sinks: Sink worker to drain before the retry.
        processed: Frames processed so far, for the message only.

    Returns:
        A ``(sample, timed_out, recovered)`` triple. ``sample`` is None only
        when both attempts came back empty.
    """
    sample = pipeline.run.pull("segmenter_output", cfg.pull_timeout_ms)
    if sample is not None:
        return sample, False, False

    print(
        f"[warn] timed out waiting for instances after {processed} frames; "
        "flushing the sink queue and retrying once",
        file=sys.stderr, flush=True,
    )
    sinks.drain()
    sample = pipeline.run.pull("segmenter_output", cfg.pull_timeout_ms)
    if sample is None:
        return None, True, False

    print(
        f"[warn] the source recovered once the backlog was flushed. That was "
        f"back-pressure from this app rather than the end of the clip; lower "
        f"runtime.queue_depth if it keeps happening.",
        file=sys.stderr, flush=True,
    )
    return sample, True, True


def consume_frames(
    pipeline: Pipeline, cfg: AppConfig, stopper: Stopper, sinks: SinkWorker,
    profile: ProfileWindow,
) -> tuple[int, int, int]:
    """The pull loop. Returns ``(processed, timeouts, recovered)``.

    Everything expensive happens on ``sinks``, so the only work between two
    ``pull`` calls is copying the frame out, parsing boxes and rebuilding masks.
    That is what keeps the decoder's pool turning over.
    """
    processed = 0
    timeouts = 0
    recovered = 0
    heartbeat_start = time_ms()
    heartbeat_instances = 0
    live_fps = float(pipeline.fps or 25)   # HUD value, refreshed each heartbeat

    while not stopper.stop and (cfg.frames <= 0 or processed < cfg.frames):
        pull_start = time_ms()
        sample, timed_out, came_back = pull_frame(pipeline, cfg, sinks, processed)
        pull_end = time_ms()
        timeouts += int(timed_out)
        recovered += int(came_back)

        if sample is None:
            if cfg.source_type == "video":
                print(source_stopped_message(cfg, pipeline, processed), flush=True)
                break
            continue

        stamp = FrameStamp.of(sample)
        instances_field = joined_field(sample, "instances", 1)
        payload, bbox_tensor = extract_bbox_payload(instances_field)
        boxes = parse_boxes(payload, pipeline.frame_w, pipeline.frame_h, cfg.max_detections)
        frame = frame_to_bgr(first_tensor(joined_field(sample, "frame", 0)))
        decode_end = time_ms()

        if cfg.segment.describe and not pipeline.described and boxes:
            pipeline.described = True
            layout = packed_mask_layout(len(payload), cfg.max_detections, cfg.segment)
            if layout is None:
                solved = (
                    f"  packed layout: {len(payload)} bytes does not decompose into "
                    f"4 + slots*{BBOX_RECORD_SIZE} + slots*side*side"
                )
            else:
                slots, side = layout
                solved = (
                    f"  packed layout: 4 + {slots}*{BBOX_RECORD_SIZE} + "
                    f"{slots}*{side}*{side} = "
                    f"{4 + slots * BBOX_RECORD_SIZE + slots * side * side} bytes"
                )
            print(
                "model output tensors (first frame with instances):\n"
                f"{describe_tensors(instances_field)}\n{solved}",
                flush=True,
            )

        bundle = extract_masks(
            instances_field, bbox_tensor, payload, len(boxes), cfg.segment, cfg.max_detections
        )
        if bundle.kind != "none" and not pipeline.mask_kind:
            pipeline.mask_kind = bundle.origin or bundle.kind
            mask_h, mask_w = bundle.shape
            if bundle.probabilities:
                values = "0/1 binary" if bundle.peak <= 1 else f"0..{bundle.peak}"
            else:
                values = "logits"
            print(
                f"masks: source={pipeline.mask_kind} layout={bundle.kind} "
                f"{mask_w}x{mask_h} slots={instance_count(bundle)} values={values}",
                flush=True,
            )

        # The decoder's buffer is free from here on. `frame` and `payload` are
        # copies; a packed `bundle` is a view on `payload`, and a bundle built
        # from separate tensors is copied out in classify_mask_tensors. Nothing
        # below this line touches pyneat, so let go before doing any of it --
        # the mask warp alone is tens of milliseconds per frame.
        sample = instances_field = bbox_tensor = None

        instances = build_instances(
            pipeline, cfg, boxes, bundle, frame.shape, draw_scale(frame, cfg.draw)
        )
        bundle = None
        mask_end = time_ms()

        processed += 1
        sinks.submit(SinkJob(processed, stamp, frame, instances, live_fps))
        sink_end = time_ms()

        profile.add(
            pull_end - pull_start,
            decode_end - pull_end,
            mask_end - decode_end,
            sink_end - mask_end,
            len(instances),
        )

        # Heartbeat, so a healthy run does not look identical to a stalled one.
        heartbeat_instances += len(instances)
        if processed % HEARTBEAT_EVERY == 0:
            elapsed = time_ms() - heartbeat_start
            rate = HEARTBEAT_EVERY * 1000.0 / elapsed if elapsed > 0 else 0.0
            live_fps = rate or live_fps
            print(
                f"[{processed}] {rate:.1f} fps, "
                f"{heartbeat_instances / HEARTBEAT_EVERY:.1f} instances/frame avg",
                flush=True,
            )
            heartbeat_start = time_ms()
            heartbeat_instances = 0

    return processed, timeouts, recovered


def run_pipeline(pipeline: Pipeline, cfg: AppConfig, stopper: Stopper) -> int:
    profile = ProfileWindow(cfg.profile, cfg.profile_interval)
    sinks = SinkWorker(cfg, pipeline, cfg.queue_depth)
    try:
        processed, timeouts, recovered = consume_frames(
            pipeline, cfg, stopper, sinks, profile
        )
    finally:
        # Ordered before anything that reads writer_frames: frames may still be
        # queued, and they belong in the recording. close() re-raises whatever
        # the worker hit, so a failing sink is not swallowed.
        sinks.close()

    profile.flush()
    total = pipeline.source_frames
    print(
        f"processed={processed}{f' of {total}' if total else ''} timeouts={timeouts} "
        f"recovered={recovered} masks={pipeline.mask_kind or 'none'}"
    )
    if sinks.blocked_ms > 1000.0 and processed:
        print(
            f"sinks: the pull loop waited {sinks.blocked_ms / 1000.0:.1f}s in total for "
            f"the compositing thread ({sinks.blocked_ms / processed:.0f} ms/frame). "
            "Cheaper settings are in the README under \"It runs slower than the detector\"."
        )

    # An incomplete recording is the most commonly reported symptom, and it has
    # several quite different causes. Rank them here rather than leaving the
    # frame count to be interpreted. "Incomplete" is measured against the clip's
    # own length where that is known, not against an arbitrary few seconds: a
    # 15 second clip cut off at 3 seconds used to pass this check in silence.
    out_fps = cfg.video_fps or pipeline.fps or 25
    seconds = pipeline.writer_frames / out_fps if out_fps else 0.0
    short = (
        pipeline.writer_frames < total if total
        else seconds < 2.0
    )
    if pipeline.writer is not None and pipeline.writer_frames and short:
        causes = []
        if cfg.frames:
            causes.append(f"runtime.frames is {cfg.frames}, which capped the run.")
        if cfg.insight_enable:
            causes.append(
                "output.insight.enable is true. Its H.264 encoder shares the codec "
                "daemon with the decoder feeding the source, so a failing encoder "
                "stalls the run. Set it to false; the recording does not need it."
            )
        if timeouts:
            causes.append(
                f"the source stopped producing frames ({timeouts} timeout(s)), so "
                "the run ended before the clip did."
            )
        if not causes:
            causes.append(
                "frames were dropped rather than blocked on. Check that "
                "runtime.overflow_policy resolved to block, as printed at startup."
            )
        listed = "\n".join(f"       {i}. {c}" for i, c in enumerate(causes, 1))
        missing = (
            f"{pipeline.writer_frames} of {total} frames, {seconds:.1f}s of "
            f"{total / out_fps:.1f}s" if total
            else f"only {seconds:.1f}s ({pipeline.writer_frames} frames at {out_fps} fps)"
        )
        print(
            f"[warn] the recording is incomplete: {missing}.\n{listed}",
            file=sys.stderr, flush=True,
        )
    elif pipeline.writer is not None and total and pipeline.writer_frames >= total:
        print(f"video: complete, all {total} frames of the clip.", flush=True)
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
        description="SiMa Neat YOLO instance segmentation with a background blur"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Parse and validate the config without loading pyneat or the model.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Pull frames and do nothing else: no masks, no blur, no overlay, no "
             "video, no stills, no Insight. If a run that stalls part-way through "
             "completes with this, the cause is how much work the app does per "
             "frame; if it stalls at the same frame, the cause is the graph.",
    )
    args = parser.parse_args(argv)

    pipeline = None
    try:
        cfg = load_app_config(args.config)
        if args.validate_config:
            net_w, net_h = resolve_net_size(cfg)
            # Needs the labels file, not the board, so a typo in
            # blur.keep_classes is caught here rather than on the DevKit.
            keep_ids = resolve_keep_classes(cfg, load_labels(cfg.labels_path))
            print(f"config OK: {args.config}")
            print(f"  family={cfg.family} -> BoxDecodeType.{FAMILY_DECODE_TOKENS[cfg.family]}")
            print(f"  {describe_preprocess(cfg, cfg.source_width, cfg.source_height)}")
            print(
                f"  segmentation: masks={cfg.segment.masks} source={cfg.segment.source} "
                f"space={cfg.segment.space} threshold={cfg.segment.threshold} "
                f"net={f'{net_w}x{net_h}' if net_w else '<from the first mask>'}"
            )
            print(f"  {describe_blur(cfg)}")
            if keep_ids is not None:
                print(f"  foreground class ids: {sorted(keep_ids)}")
            return 0

        if args.minimal:
            # Strip the consumer back to a bare pull loop. Nothing here changes
            # the graph, so it isolates "we are too slow" from "the graph is
            # wrong" in a single run.
            cfg = dataclasses.replace(
                cfg,
                segment=dataclasses.replace(cfg.segment, masks="off", describe=False),
                blur=dataclasses.replace(cfg.blur, enable=False),
                save_enable=False, video_enable=False, insight_enable=False,
            )
            print(
                "[minimal] masks, blur, overlay, video, stills and Insight are all "
                "off.\n          Reaching the end of the clip means the graph is fine "
                "and the app was\n          simply holding buffers too long.",
                flush=True,
            )

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
