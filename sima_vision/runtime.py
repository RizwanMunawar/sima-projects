"""Deferred third-party imports and the config-token to pyneat-enum tables.

``pyneat`` is an aarch64 wheel that only exists on the DevKit, and ``cv2`` comes
from the board's system packages rather than pip. Importing either at module
scope would make ``sima-vision --help`` and ``sima-vision detect --validate``
fail on a laptop, which is exactly where you want to check a config before
copying it to the board. So both are loaded by :func:`load_runtime_dependencies`
at the point of first real use, and the enum tables below are plain strings so
they can be validated without either.

Neither lives where pip would put it, and they do not live in the same place as
each other:

* ``cv2`` is in the board's system packages, ``/usr/lib/python3*/dist-packages``.
* ``pyneat`` is in a **virtualenv of its own**, which ``sima-cli sdk setup``
  creates at ``~/pyneat``. Nothing puts it on the default path.

So ``pip install sima-vision`` followed by ``sima-vision detect`` used to fail
with ``ModuleNotFoundError: pyneat`` whenever the install went anywhere but that
venv, which is every install that does not know to look for it. The honest fix
would be to tell everyone to type ``~/pyneat/bin/pip install sima-vision``, and
that still works and is still the least surprising thing to do. But the plain
command is what people type, so :func:`find_pyneat_env` goes and finds the venv
instead, and only after the ordinary import has already failed. A working
interpreter is never second-guessed.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

# Populated by load_runtime_dependencies(). Modules read them through this
# module (``from . import runtime`` then ``runtime.cv2``) rather than importing
# the names directly, because a `from runtime import cv2` binds None forever.
cv2 = None
np = None
pyneat = None

#: ``cv2.FONT_HERSHEY_SIMPLEX``, filled in once cv2 is available.
FONT = 0

#: Points at the pyneat virtualenv when it is somewhere unusual.
PYNEAT_ENV = "SIMA_VISION_PYNEAT"

#: Where the pyneat venv ends up, tried first because they cost one stat each.
#: ``~/pyneat`` is what ``sima-cli sdk setup`` creates; ``/media/nvme`` is where
#: you are told to put it by hand, because the board's root filesystem is too
#: small for it.
PYNEAT_HOMES = (
    "~/pyneat",
    "/media/nvme/neat/pyneat",
    "/media/nvme/pyneat",
    "/opt/pyneat",
)

#: Searched when none of the above has it. A board that was set up by hand, or
#: by a different SDK version, puts it somewhere else entirely, and a fixed list
#: of guesses is exactly the thing that then says "not found" about a venv
#: sitting two directories away.
PYNEAT_SEARCH_ROOTS = (
    "~",
    "/opt",
    "/media/nvme",
    "/media/nvme/neat",
    "/usr/local",
    "/srv",
)


def find_pyneat_env() -> tuple[Path | None, str]:
    """Locate a pyneat virtualenv this interpreter could actually import from.

    Returns:
        A ``(site_packages, note)`` pair. ``site_packages`` is None when there is
        nothing usable, and ``note`` always says why in one line, so ``doctor``
        and the run error can print the same explanation.

    The version check is the point. ``pyneat`` is a compiled extension built for
    one CPython, so putting a 3.10 venv on a 3.12 path swaps
    ``ModuleNotFoundError`` for an undefined-symbol crash out of the dynamic
    linker, which is a far worse thing to hand someone.
    """
    want = f"python{sys.version_info.major}.{sys.version_info.minor}"
    override = os.environ.get(PYNEAT_ENV, "")

    wrong_version: list[str] = []

    def is_dir(path: Path) -> bool:
        """`Path.is_dir()` that answers "no" instead of raising.

        pathlib only swallows ENOENT, ENOTDIR, EBADF and ELOOP; EACCES comes
        straight back out. Since this walks directories nobody promised us
        access to -- and on the board plenty of them are root's -- one
        unreadable path would otherwise end a run with a traceback rather than
        with the search continuing past it.
        """
        try:
            return path.is_dir()
        except OSError:
            return False

    def globs(path: Path, pattern: str) -> list[Path]:
        """`Path.glob()` with the same promise, for the same reason."""
        try:
            return sorted(path.glob(pattern))
        except OSError:
            return []

    def look_in(root: Path) -> Path | None:
        """Is there a pyneat for *this* Python under this venv root?"""
        site = root / "lib" / want / "site-packages"
        if globs(site, "pyneat*"):
            return site
        # There, but built for another CPython. Note the interpreter that can
        # use it rather than leaving someone to work it out.
        for other in globs(root / "lib", "python*"):
            if other.name != want and globs(other / "site-packages", "pyneat*"):
                wrong_version.append(f"{root}/bin/python3 ({other.name})")
        return None

    if override:
        root = Path(override).expanduser()
        found = look_in(root) if is_dir(root) else None
        if found:
            return found, f"using pyneat from {root}"
        if wrong_version:
            return None, f"found pyneat, but built for {wrong_version[0]}, not {want}"
        return None, f"${PYNEAT_ENV} is {override}, which has no pyneat for {want}"

    for home in PYNEAT_HOMES:
        root = Path(home).expanduser()
        if is_dir(root):
            found = look_in(root)
            if found:
                return found, f"using pyneat from {root}"

    # Nothing where it is supposed to be, so go and look. One level down from a
    # handful of roots finds every venv anyone actually makes, and stops well
    # short of walking the filesystem.
    for base in PYNEAT_SEARCH_ROOTS:
        parent = Path(base).expanduser()
        if not is_dir(parent):
            continue
        try:
            children = sorted(child for child in parent.iterdir() if is_dir(child))
        except OSError:                       # unreadable, not our problem
            continue
        for child in children:
            if not is_dir(child / "lib"):     # not a venv, skip cheaply
                continue
            found = look_in(child)
            if found:
                return found, f"using pyneat from {child}"

    if wrong_version:
        return None, (
            f"found pyneat, but built for {', '.join(sorted(set(wrong_version)))} "
            f"and this is {want}"
        )
    return None, f"no pyneat for {want} anywhere under {', '.join(PYNEAT_SEARCH_ROOTS)}"


def missing_pyneat_message(note: str) -> str:
    """What to print when pyneat cannot be imported and cannot be found."""
    on_board = bool(glob.glob("/usr/lib/python3*/dist-packages")) and sys.platform.startswith(
        "linux"
    )
    if not on_board:
        return (
            "pyneat is missing, and this does not look like a DevKit.\n"
            "  It is an aarch64 wheel that ships with the Palette SDK, not "
            "something pip can\n  install, so inference only runs on the board. "
            "Everything else works here:\n"
            "    sima-vision <task> --validate     check a config\n"
            "    sima-vision watch -- <task>       run it on the board, watch it here"
        )
    return (
        f"pyneat is missing: {note}.\n"
        "  `sima-cli sdk setup` puts it in a virtualenv of its own, and pip "
        "installs into\n  whichever Python you ran pip with. Install into that "
        "venv instead:\n"
        "    ~/pyneat/bin/pip install sima-vision\n"
        "    ~/pyneat/bin/sima-vision detect\n"
        f"  Or point at it: export {PYNEAT_ENV}=/path/to/the/venv\n"
        "  If pairing never ran, `sima-cli sdk setup --devkit <ip>` from your PC "
        "installs it."
    )


def load_runtime_dependencies() -> None:
    """Import cv2, numpy and pyneat, finding the board's copies of each.

    The board ships OpenCV in ``/usr/lib/python3*/dist-packages`` rather than in
    any venv, so that directory goes on ``sys.path`` before the import. Doing it
    here rather than in the venv keeps ``opencv-python`` -- which would drag in
    numpy 2.x and break pyneat -- off the board entirely.

    pyneat is looked for only if importing it the ordinary way fails, so an
    interpreter that already has it is never interfered with.

    Raises:
        ImportError: With the interpreter to use, when pyneat cannot be found.
    """
    global cv2, np, pyneat, FONT
    if pyneat is not None:
        return
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        import pyneat as pyneat_module
    except ImportError:
        site, note = find_pyneat_env()
        if site is None:
            raise ImportError(missing_pyneat_message(note)) from None
        # Ahead of the current environment: this venv also holds the numpy<2
        # that pyneat was built against, and that is the one it has to get.
        sys.path.insert(0, str(site))
        print(f"[pyneat] {note}", flush=True)
        import pyneat as pyneat_module

    import cv2 as cv2_module
    import numpy as np_module

    cv2, np, pyneat = cv2_module, np_module, pyneat_module
    FONT = cv2.FONT_HERSHEY_SIMPLEX


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
