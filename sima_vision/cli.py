"""The ``sima-vision`` command.

One subcommand per task. Every flag that corresponds to a config key declares
its dotted path as its argparse ``dest``, so the whole override mechanism is
this::

    parser.add_argument("--source", dest="source.uri")
    ...
    {"source.uri": "clip.h264"}  ->  raw["source"]["uri"] = "clip.h264"

Overrides are written into the parsed YAML *before* the loaders run, so a CLI
flag goes through exactly the same defaulting and validation a config file
does, and cannot reach a state a config file could not.

Config is optional. The dataclass defaults are a complete configuration, so
``--model`` and ``--source`` are enough to run with no YAML at all; a file adds
to that, and flags win over both.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__, runtime
from .neat import describe_preprocess
from .runloop import Stopper
from .runtime import FAMILY_DECODE_TOKENS, load_runtime_dependencies
from .setup_commands import run_fetch, run_init
from .tasks import TASKS

EPILOG = """\
examples:
  sima-vision detect  --source clip.h264 --model yolo26m-det.tar.gz
  sima-vision segment --source clip.h264 --model yolo26m-seg.tar.gz --blur
  sima-vision segment --source clip.h264 --anonymise --keep-classes person
  sima-vision fall    --source rtsp://cam/live --alert-to ops@example.com
  sima-vision detect  --config object-detection/config.yaml --validate

Run from inside an app folder, or from the repo root, and config.yaml is found
automatically. Everything runs on the DevKit; --validate works anywhere.
"""


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags every task understands. The dest is the config key it writes."""
    source = parser.add_argument_group("source")
    source.add_argument(
        "--source", "-s", dest="source.uri", metavar="URI",
        help="Video file, RTSP URL, or empty for the DevKit camera. Raw .h264 only "
             "for files; see the README on converting.",
    )
    source.add_argument(
        "--source-type", dest="source.type", choices=("video", "rtsp", "usb"),
        help="Where frames come from. Default video.",
    )
    source.add_argument(
        "--fps", dest="source.fps", type=int, metavar="N",
        help="Source frame rate. Default 0, which reads it from the stream.",
    )
    source.add_argument(
        "--width", dest="source.width", type=int, metavar="PX",
        help="Source width. Default 0, which reads it from the stream's SPS.",
    )
    source.add_argument(
        "--height", dest="source.height", type=int, metavar="PX",
        help="Source height. Default 0, which reads it from the stream's SPS.",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model", "-m", dest="model.path", metavar="PATH",
        help="Compiled model archive (.tar.gz) as seen on the DevKit.",
    )
    model.add_argument(
        "--labels", dest="model.labels", metavar="PATH",
        help="Newline-separated class names. Defaults to the packaged COCO list.",
    )
    model.add_argument(
        "--family", dest="model.family", metavar="NAME",
        choices=sorted(FAMILY_DECODE_TOKENS),
        help="Detection head. Must match the model or you get no detections.",
    )
    model.add_argument(
        "--conf", dest="decode.score_threshold", type=float, metavar="T",
        help="Minimum detection confidence. Default 0.30.",
    )
    model.add_argument(
        "--iou", dest="decode.nms_iou", type=float, metavar="T",
        help="Non-max suppression IoU threshold. Default 0.60.",
    )
    model.add_argument(
        "--max-det", dest="decode.max_detections", type=int, metavar="N",
        help="Top-K cap per frame. Default 50.",
    )

    run = parser.add_argument_group("runtime")
    run.add_argument(
        "--frames", "-n", dest="runtime.frames", type=int, metavar="N",
        help="Stop after N frames. Default 0, which runs until interrupted.",
    )
    run.add_argument(
        "--timeout", dest="runtime.pull_timeout_ms", type=int, metavar="MS",
        help="How long to wait for a frame before giving up. Default 20000.",
    )
    run.add_argument(
        "--queue-depth", dest="runtime.queue_depth", type=int, metavar="N",
        help="How far ahead of the sinks the pull loop may run. Default 1.",
    )
    run.add_argument(
        "--profile", dest="runtime.profile", action="store_const", const=True,
        help="Print per-stage timings every runtime.profile_interval frames.",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--video", dest="output.video.path", metavar="PATH",
        help="Where to write the annotated recording on the DevKit.",
    )
    out.add_argument(
        "--no-video", dest="output.video.enable", action="store_const", const=False,
        help="Do not record.",
    )
    out.add_argument(
        "--save-dir", dest="output.save.dir", metavar="DIR",
        help="Where to write annotated stills.",
    )
    out.add_argument(
        "--save-every", dest="output.save.every", type=int, metavar="N",
        help="Write every Nth still. Default 10; 0 disables.",
    )
    out.add_argument(
        "--no-save", dest="output.save.enable", action="store_const", const=False,
        help="Do not write stills.",
    )
    out.add_argument(
        "--no-hud", dest="output.video.hud", action="store_const", const=False,
        help="Leave the frame-rate badge off the overlay.",
    )
    out.add_argument(
        "--insight", dest="output.insight.enable", action="store_const", const=True,
        help="Stream to Neat Insight over UDP. Off by default: its encoder shares "
             "the codec daemon with the decoder, so it can stall a file run.",
    )
    out.add_argument(
        "--insight-host", dest="output.insight.host", metavar="HOST",
        help="Insight address as the DevKit sees it. Default 127.0.0.1.",
    )

    config = parser.add_mutually_exclusive_group()
    config.add_argument(
        "--config", "-c", type=Path, metavar="PATH",
        help="Config file. Defaults to ./config.yaml, then <app-dir>/config.yaml.",
    )
    config.add_argument(
        "--no-config", action="store_true",
        help="Ignore any config file and run on the built-in defaults plus these "
             "flags, even when a config.yaml is sitting right there.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Parse and check the config, print what it resolved to, and exit. "
             "Needs neither pyneat nor the board, so it works on a laptop.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sima-vision",
        description="Live YOLO computer vision on a SiMa Modalix DevKit 3.0.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"sima-vision {__version__}")
    subparsers = parser.add_subparsers(dest="task", metavar="COMMAND")

    for name, task_cls in TASKS.items():
        task = task_cls()
        sub = subparsers.add_parser(
            name,
            help=task.help,
            description=task.help + ".",
            epilog=EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        add_shared_arguments(sub)
        group_title = f"{name} options"
        task_group = sub.add_argument_group(group_title)
        task.add_arguments(task_group)
        sub.set_defaults(_task=task_cls)

    add_init_parser(subparsers)
    add_fetch_parser(subparsers)
    add_preview_parser(subparsers)
    add_doctor_parser(subparsers)
    return parser


def add_init_parser(subparsers) -> None:
    """``init`` -- write a commented starter config into the working directory."""
    sub = subparsers.add_parser(
        "init",
        help="Write a documented config.yaml you can edit",
        description=(
            "Copy this task's starter config into the working directory. It is "
            "the same file the repo ships, with every setting commented, and it "
            "comes out of the installed package -- no clone needed."
        ),
    )
    sub.add_argument("task", choices=list(TASKS), help="Which app to configure.")
    sub.add_argument(
        "--out", "-o", type=Path, default=Path("config.yaml"), metavar="PATH",
        help="Where to write it. Default ./config.yaml, which every command finds "
             "on its own.",
    )
    sub.add_argument(
        "--force", "-f", action="store_true", help="Overwrite an existing file.",
    )
    sub.set_defaults(_command="init")


def add_fetch_parser(subparsers) -> None:
    """``fetch`` -- download the sample clips and print the model command."""
    sub = subparsers.add_parser(
        "fetch",
        help="Download the sample clips, and say how to get the model",
        description=(
            "Download the two sample videos into ./assets/videos/. They are on a "
            "public GitHub release, so they need no login. The model packs do "
            "need a community.sima.ai login, so that command is printed for you "
            "to run rather than attempted here."
        ),
    )
    sub.add_argument(
        "task", choices=list(TASKS), nargs="?", default="detect",
        help="Which model to print the download command for. Default detect.",
    )
    sub.add_argument(
        "--into", type=Path, default=Path("assets"), metavar="DIR",
        help="Where to put them. Default ./assets.",
    )
    sub.set_defaults(_command="fetch")


def add_preview_parser(subparsers) -> None:
    """``preview`` -- render the overlay a config produces, with no board."""
    sub = subparsers.add_parser(
        "preview",
        help="Render what your config looks like, with no board and no model",
        description=(
            "Draw one frame the way a real run would, so visualization and blur "
            "settings can be tuned on a laptop. No model is run: the detections "
            "are synthetic and only exist to give the drawing code something to "
            "draw. Needs numpy and OpenCV (pip install 'sima-vision[preview]')."
        ),
        epilog=(
            "examples:\n"
            "  sima-vision preview                                  # detect, defaults\n"
            "  sima-vision preview --task segment -o blur.png\n"
            "  sima-vision preview --task segment "
            "-c instance-segmentation/config.yaml\n"
            "  sima-vision preview --task fall --source my-photo.jpg\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub.add_argument(
        "--task", "-t", choices=list(TASKS), default="detect",
        help="Which app's overlay to draw. Default detect.",
    )
    group = sub.add_mutually_exclusive_group()
    group.add_argument(
        "--config", "-c", type=Path, metavar="PATH",
        help="Config to preview. Defaults to ./config.yaml, then <app-dir>/config.yaml.",
    )
    group.add_argument(
        "--no-config", action="store_true",
        help="Preview the built-in defaults, ignoring any config file.",
    )
    sub.add_argument(
        "--source", "-s", metavar="PATH",
        help="Image or video to draw on. Anything OpenCV can open; raw .h264 "
             "cannot be, and falls back to the synthetic scene.",
    )
    sub.add_argument(
        "--out", "-o", type=Path, default=Path("preview.png"), metavar="PATH",
        help="Where to write the PNG. Default preview.png.",
    )
    sub.add_argument(
        "--size", default="1280x720", metavar="WxH",
        help="Synthetic scene size. Default 1280x720.",
    )
    sub.set_defaults(_command="preview")


def add_doctor_parser(subparsers) -> None:
    """``doctor`` -- say what is installed and what each part enables."""
    sub = subparsers.add_parser(
        "doctor",
        help="Check what is installed and what you can do with it",
        description=(
            "Report which pieces are present. Nothing here is fatal on its own: "
            "the parts needed to check a config and preview an overlay are "
            "separate from the parts needed to run inference on the DevKit."
        ),
    )
    sub.set_defaults(_command="doctor")


def parse_size(text: str) -> tuple[int, int]:
    """Read a ``WxH`` size. Raises ValueError on anything else."""
    width, _, height = text.lower().partition("x")
    try:
        size = (int(width), int(height))
    except ValueError:
        raise ValueError(f"--size must look like 1280x720, got {text!r}") from None
    if size[0] < 64 or size[1] < 64:
        raise ValueError(f"--size must be at least 64x64, got {text!r}")
    return size


def collect_overrides(args: argparse.Namespace) -> dict:
    """Every dotted-dest flag the user actually gave, as config paths.

    ``None`` means the flag was not given, which is how an unset flag defers to
    the config file rather than overwriting it with an argparse default.
    """
    return {
        key: value
        for key, value in vars(args).items()
        if "." in key and value is not None
    }


def print_validation(task, cfg) -> None:
    """What ``--validate`` prints. Deliberately the same shape for every task."""
    where = cfg.config_path or "<defaults and flags only>"
    print(f"config OK: {where}")
    print(f"  model: {cfg.model_path or '<unset>'}")
    print(f"  labels: {cfg.labels_path}")
    print(f"  family={cfg.family} -> BoxDecodeType.{FAMILY_DECODE_TOKENS[cfg.family]}")
    print(f"  source: type={cfg.source_type} uri={cfg.source_uri or '<default camera>'}")
    print(
        f"  decode: conf={cfg.score_threshold} iou={cfg.nms_iou} "
        f"max_det={cfg.max_detections}"
    )
    print(f"  {describe_preprocess(cfg, cfg.source_width, cfg.source_height)}")
    for line in task.describe(cfg):
        print(f"  {line}")
    outputs = []
    if cfg.video_enable:
        outputs.append(f"video={cfg.video_path}")
    if cfg.save_enable:
        outputs.append(f"stills={cfg.save_dir}/ every={cfg.save_every}")
    if cfg.insight_enable:
        outputs.append(f"insight={cfg.insight_host}:{cfg.video_port_base}")
    print(f"  output: {' '.join(outputs)}")


def load_drawing_dependencies() -> None:
    """Bind numpy and OpenCV without requiring pyneat.

    ``load_runtime_dependencies`` needs all three, because a real run does. A
    preview draws and composites but never touches the MLA, so it needs only
    two -- which is what lets it work on a laptop.
    """
    from . import runtime as rt

    if rt.cv2 is not None:
        return
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            f"preview needs numpy and OpenCV, and {exc.name} is missing.\n"
            "  pip install 'sima-vision[preview]'\n"
            "\nOn the DevKit they come from the board's system packages instead, "
            "so nothing needs installing there."
        ) from None
    rt.cv2, rt.np, rt.FONT = cv2, np, cv2.FONT_HERSHEY_SIMPLEX


def run_preview(args) -> int:
    """Draw one frame the way a real run would, and write it to a PNG."""
    from . import preview as preview_module
    from .config import discover_config, read_config_file
    from .sinks import load_labels

    load_drawing_dependencies()
    task = TASKS[args.task]()
    use_file = not args.no_config

    # A preview runs no model and opens no source, but the config it previews
    # still has to pass the ordinary validation. Fill in only what is missing,
    # so a real config's values are never shadowed by a placeholder.
    found = discover_config(args.config) if use_file else None
    raw = read_config_file(found)
    overrides = {}
    if not (raw.get("model") or {}).get("path"):
        overrides["model.path"] = "<preview: no model is run>"
    if not (raw.get("source") or {}).get("uri"):
        overrides["source.uri"] = "<preview>"

    cfg = task.load(args.config, overrides, use_file=use_file)
    labels = load_labels(cfg.labels_path)

    frame = preview_module.read_first_frame(args.source) if args.source else None
    if frame is not None:
        height, width = frame.shape[:2]
        _, subjects = preview_module.build_scene(width, height)
        origin = args.source
    else:
        if args.source:
            print(
                f"[warn] OpenCV could not read {args.source}; using the synthetic "
                f"scene instead.\n       Raw .h264 is expected to fail here -- it "
                f"has no container for OpenCV to parse.",
                file=sys.stderr,
            )
        width, height = parse_size(args.size)
        frame, subjects = preview_module.build_scene(width, height)
        origin = f"synthetic scene {width}x{height}"

    annotated = preview_module.render(task, cfg, frame, subjects, labels)

    out = args.out
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    if not runtime.cv2.imwrite(str(out), annotated):
        raise SystemExit(f"could not write {out}")

    where = cfg.config_path or "<defaults only>"
    print(f"preview: {task.name} overlay from {where}")
    print(f"  frame:  {origin}")
    print(f"  drew:   {len(subjects)} synthetic detections -- NO MODEL WAS RUN")
    for line in task.describe(cfg):
        print(f"  {line}")
    print(f"\nwrote {out.resolve()}  ({annotated.shape[1]}x{annotated.shape[0]})")
    print("Open it, edit the config, run this again. No board needed.")
    return 0


def mark_for(ok: bool) -> str:
    return "yes" if ok else "no "


def run_doctor() -> int:
    """Report what is installed, and what each piece unlocks."""
    import glob
    import importlib.util
    import shutil

    print(f"sima-vision {__version__}")
    print(f"python      {sys.version.split()[0]}  ({sys.executable})\n")

    def probe(module: str):
        try:
            found = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            return None
        return found

    rows = [
        ("yaml", "read config files", "required", "pip install pyyaml"),
        ("numpy", "preview, and every run", "preview", "pip install 'sima-vision[preview]'"),
        ("cv2", "preview, and every run", "preview", "pip install 'sima-vision[preview]'"),
        ("pyneat", "run inference on the MLA", "DevKit", "ships with the Palette SDK"),
    ]
    have = {}
    width = len("board packages")
    for module, what, needed_for, fix in rows:
        ok = probe(module) is not None
        have[module] = ok
        print(f"  {mark_for(ok)}  {module:<{width}}  {what}")
        if not ok:
            print(f"  {'':<5}{'':<{width}}  needed for: {needed_for} -- {fix}")

    dist = glob.glob("/usr/lib/python3*/dist-packages")
    board = dist[0] if dist else "(not a DevKit, which is fine)"
    print(f"\n  {mark_for(bool(dist))}  {'board packages':<{width}}  {board}")

    ffprobe = shutil.which("ffprobe")
    where = ffprobe or "optional; raw .h264 is read directly"
    print(f"  {mark_for(bool(ffprobe))}  {'ffprobe':<{width}}  {where}")

    drawing = have["numpy"] and have["cv2"]
    print("\nWhat you can do right now:")
    for ok, command, what in (
        (have["yaml"], "sima-vision <task> --validate", "check a config"),
        (drawing, "sima-vision preview", "see your overlay and blur"),
        (drawing and have["pyneat"], "sima-vision <task>", "run inference on the MLA"),
    ):
        print(f"  {mark_for(ok)}  {command:<30}  {what}")

    if not have["pyneat"]:
        print(
            "\npyneat is an aarch64 wheel that ships with the Palette SDK and is not on\n"
            "PyPI, so inference only runs on the DevKit. Everything else works here."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "task", None):
        parser.print_help()
        return 2

    command = getattr(args, "_command", None)
    if command is not None:
        try:
            if command == "doctor":
                return run_doctor()
            if command == "init":
                return run_init(args.task, args.out, args.force)
            if command == "fetch":
                return run_fetch(args.task, args.into)
            return run_preview(args)
        except KeyboardInterrupt:
            return 130
        except SystemExit as exc:
            print(f"[ERR] {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"[ERR] {exc}", file=sys.stderr)
            return 1

    task = args._task()
    try:
        cfg = task.load(
            args.config, collect_overrides(args), use_file=not args.no_config
        )
        cfg = task.post_process(cfg, args)

        if args.validate:
            print_validation(task, cfg)
            return 0

        load_runtime_dependencies()
        if cfg.profile:
            os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
            os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
        if cfg.save_enable:
            Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        task.run(cfg, Stopper())
        return 0
    except KeyboardInterrupt:
        return 130
    except ImportError as exc:
        # pyneat is aarch64-only and cv2 comes from the board's system packages,
        # so this is what running on the wrong machine looks like.
        print(
            f"[ERR] {exc}\n"
            "This app runs on the DevKit, not in the SDK container or on your "
            "laptop:\n  pyneat is compiled for aarch64 and OpenCV comes from the "
            "board's system\n  packages. Use --validate to check a config from here.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
