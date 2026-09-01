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

from . import __version__
from .runtime import FAMILY_DECODE_TOKENS, load_runtime_dependencies
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
    return parser


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
    from .neat import describe_preprocess

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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "task", None):
        parser.print_help()
        return 2

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

        from .runloop import Stopper

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
