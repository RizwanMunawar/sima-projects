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

Config is optional, and so are the flags. The dataclass defaults are a
complete configuration down to a model and a clip -- see
:mod:`sima_vision.assets` -- so ``sima-vision detect`` runs with no YAML and no
arguments at all; a file adds to that, and flags win over both.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .devkit import (
    DEVKIT_ENV,
    VIDEO_PORT,
    run_pull,
    run_push,
    run_remote,
    run_watch,
)
from .neat import describe_preprocess
from .netsetup import run_setup_network
from .runloop import Stopper
from .runtime import (
    FAMILY_DECODE_TOKENS,
    find_pyneat_env,
    load_runtime_dependencies,
    missing_pyneat_message,
)
from .setup_commands import run_fetch, run_init
from .tasks import TASKS

EPILOG = """\
examples:
  sima-vision detect                       the sample clip and model, fetched
  sima-vision detect  --source clip.h264 --model yolo26m-det.tar.gz
  sima-vision detect  --source https://example.com/clip.h264
  sima-vision segment --source clip.h264 --model yolo26m-seg.tar.gz --blur
  sima-vision segment --source clip.h264 --anonymise --keep-classes person
  sima-vision fall    --source rtsp://cam/live --alert-to ops@example.com

without a board:
  sima-vision init segment                    write a documented config.yaml
  sima-vision segment --validate              check it
  sima-vision doctor                          what is installed, and what it allows

first time, if the board has no internet:
  sima-vision setup network                   check the sharing from PC to board
  sima-vision setup network --apply           and set it up (Windows, as admin)

driving the board from your PC:
  sima-vision push clip.h264                  copy files over
  sima-vision watch  -- detect                run it there, live video back here
  sima-vision remote -- detect --frames 200   run it there, output in the terminal
  sima-vision pull                            bring the results back

config.yaml in the working directory is picked up automatically, and so is
./assets -- a clip or model that is missing there is downloaded on the first
run. Running a task needs the DevKit; everything under "without a board" does
not, and none of it touches the network.
"""


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags every task understands. The dest is the config key it writes."""
    source = parser.add_argument_group("source")
    source.add_argument(
        "--source", "-s", dest="source.uri", metavar="URI",
        help="Video file, https URL, RTSP URL, or empty for this task's sample "
             "clip. An https URL is downloaded into assets/videos/ once and "
             "reused. Raw .h264 only for files; see the README on converting.",
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
        help="Compiled model archive (.tar.gz), or an https URL to one. Empty "
             "uses this task's default in assets/models/, fetched with sima-cli "
             "on the first run.",
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
        "--video-path", dest="output.video.path", metavar="PATH",
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


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Which config file to read, or none at all."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config", "-c", type=Path, metavar="PATH",
        help="Config file. Defaults to ./config.yaml in the working directory.",
    )
    group.add_argument(
        "--no-config", action="store_true",
        help="Ignore any config file and use the built-in defaults plus these "
             "flags, even when a config.yaml is sitting right there.",
    )


def add_task_arguments(parser: argparse.ArgumentParser, task) -> None:
    """Everything one task understands: the shared flags plus its own."""
    add_shared_arguments(parser)
    add_config_arguments(parser)
    parser.add_argument(
        "--validate", action="store_true",
        help="Parse and check the config, print what it resolved to, and exit. "
             "Needs neither pyneat nor the board, so it works on a laptop.",
    )
    task.add_arguments(parser.add_argument_group(f"{task.name} options"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sima-vision",
        description="Live YOLO computer vision on a SiMa Modalix DevKit 3.0.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"sima-vision {__version__}")
    # dest="command", not "task": the `init` and `fetch` positionals are
    # both called task, and would overwrite it.
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    for name, task_cls in TASKS.items():
        task = task_cls()
        sub = subparsers.add_parser(
            name,
            help=task.help,
            description=task.help + ".",
            epilog=EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        add_task_arguments(sub, task)
        sub.set_defaults(_task=task_cls)

    add_init_parser(subparsers)
    add_fetch_parser(subparsers)
    add_doctor_parser(subparsers)
    add_setup_parser(subparsers)
    add_push_parser(subparsers)
    add_pull_parser(subparsers)
    add_watch_parser(subparsers)
    add_remote_parser(subparsers)
    return parser


def add_host_argument(parser: argparse.ArgumentParser) -> None:
    """Which board. The same flag on all three transfer commands."""
    parser.add_argument(
        "--host", "-H", metavar="USER@ADDR",
        help=f"The DevKit, as ssh takes it. Defaults to ${DEVKIT_ENV} so you "
             f"only say it once.",
    )


def add_push_parser(subparsers) -> None:
    """``push`` -- copy files to the board."""
    sub = subparsers.add_parser(
        "push",
        help="Copy files or folders to the DevKit",
        description=(
            "Copy local files to the DevKit's home directory with scp. Folders "
            "are copied whole. On Windows this is also the way to avoid scp "
            "reading a drive letter as a hostname."
        ),
        epilog=(
            "examples:\n"
            "  sima-vision push config.yaml\n"
            "  sima-vision push my-clip.h264 my-model.tar.gz\n"
            "  sima-vision push assets --dest '~/'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub.add_argument("paths", nargs="+", type=Path, metavar="PATH",
                     help="Files or folders to copy.")
    sub.add_argument("--dest", default="~/", metavar="DIR",
                     help="Where to put them on the board. Default ~/.")
    add_host_argument(sub)
    sub.set_defaults()


def add_pull_parser(subparsers) -> None:
    """``pull`` -- copy results back."""
    sub = subparsers.add_parser(
        "pull",
        help="Copy results back from the DevKit",
        description=(
            "Copy a run's output back to this machine. With no names it asks "
            "for everything any task could have written -- the annotated video, "
            "frames/, alerts/ and config.yaml -- and takes whatever is there, "
            "so it does not need to be told which task ran."
        ),
        epilog=(
            "examples:\n"
            "  sima-vision pull\n"
            "  sima-vision pull detections.mp4\n"
            "  sima-vision pull --into results/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub.add_argument("names", nargs="*", metavar="NAME",
                     help="Names on the board, relative to its home directory.")
    sub.add_argument("--into", type=Path, default=Path("."), metavar="DIR",
                     help="Where to put them here. Default the current directory.")
    add_host_argument(sub)
    sub.set_defaults()


def add_setup_parser(subparsers) -> None:
    """``setup network`` -- share this PC's internet with the board."""
    sub = subparsers.add_parser(
        "setup",
        help="One-time setup steps, starting with the network",
        description="One-time things you do once per machine.",
    )
    inner = sub.add_subparsers(dest="topic", metavar="TOPIC", required=True)
    network = inner.add_parser(
        "network",
        help="Share this PC's internet connection with the DevKit",
        description=(
            "The DevKit has no internet of its own: it is cabled to this PC, so "
            "this PC has to pass its connection along. This works out which of "
            "your adapters has the internet and which one the board is on, says "
            "whether sharing is actually set up, and with --apply sets it up.\n\n"
            "It changes nothing unless you pass --apply."
        ),
        epilog=(
            "examples:\n"
            "  sima-vision setup network\n"
            "  sima-vision setup network --apply\n"
            "  sima-vision setup network --host sima@192.168.137.50\n"
            "\nWith --host it also runs the checks on the board itself, which is\n"
            "the only answer that cannot be wrong.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    network.add_argument(
        "--apply", action="store_true",
        help="Actually turn sharing on. Windows only, and needs an "
             "Administrator terminal; without one it prints the command to run.",
    )
    add_host_argument(network)
    sub.set_defaults()


def add_watch_parser(subparsers) -> None:
    """``watch`` -- run on the board with the live video sent back here."""
    sub = subparsers.add_parser(
        "watch",
        help="Run a task on the DevKit and watch its live video here",
        description=(
            "Run a task on the board with its video feed aimed at this machine. "
            "These are the real annotated frames the board is producing, not a "
            "simulation: the same overlay that goes into the recording, encoded "
            "as H.264 and sent over RTP while the run happens.\n\n"
            "Nothing is decoded here. An SDP file is written and the exact "
            "ffplay, GStreamer or VLC command printed, because those already do "
            "that job properly."
        ),
        epilog=(
            "examples:\n"
            "  sima-vision watch -- detect\n"
            "  sima-vision watch -- segment --blur-strength 81\n"
            "  sima-vision watch -- fall --frames 500\n"
            "\nEverything after -- is passed to sima-vision on the board, with\n"
            "--insight and --insight-host added for you.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub.add_argument("argv", nargs=argparse.REMAINDER, metavar="-- ARGS",
                     help="The task to run there, after a literal --.")
    sub.add_argument(
        "--to", metavar="ADDR",
        help="The address the board should send video to. Default: whichever "
             "of this machine's addresses is on the board's own network, which "
             "is not the same as the one that reaches the internet.",
    )
    sub.add_argument(
        "--port", type=int, default=VIDEO_PORT, metavar="N",
        help=f"UDP port to receive video on. Default {VIDEO_PORT}.",
    )
    sub.add_argument(
        "--sdp", type=Path, metavar="PATH",
        help="Where to write the SDP the player needs. Default ./sima-vision.sdp.",
    )
    add_host_argument(sub)
    sub.set_defaults()


def add_remote_parser(subparsers) -> None:
    """``remote`` -- run a task on the board from here."""
    sub = subparsers.add_parser(
        "remote",
        help="Run a sima-vision command on the DevKit over SSH",
        description=(
            "Run `sima-vision <args>` on the board and watch it here. It always "
            "asks for a pty, so Ctrl-C actually reaches the task and it releases "
            "the MLA -- without that the next launch fails with a busy device."
        ),
        epilog=(
            "examples:\n"
            "  sima-vision remote -- detect --frames 200\n"
            "  sima-vision remote -- segment --blur-strength 81\n"
            "  sima-vision remote -- doctor\n"
            "\nEverything after -- is passed to sima-vision on the board.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub.add_argument("argv", nargs=argparse.REMAINDER, metavar="-- ARGS",
                     help="The command to run there, after a literal --.")
    add_host_argument(sub)
    sub.set_defaults()


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
    sub.set_defaults()


def add_fetch_parser(subparsers) -> None:
    """``fetch`` -- download the sample clips and print the model command."""
    sub = subparsers.add_parser(
        "fetch",
        help="Download the sample clips, and say how to get the model",
        description=(
            "Download the two sample videos into ./assets/videos/. They are on a "
            "public GitHub release, so they need no login. The model packs do "
            "need a community.sima.ai login, so that command is printed for you "
            "to run rather than attempted here. A run fetches both on its own "
            "when they are missing, so this is only for getting the 13 MB clip "
            "out of the way first."
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
    sub.set_defaults()



def add_doctor_parser(subparsers) -> None:
    """``doctor`` -- say what is installed and what each part enables."""
    sub = subparsers.add_parser(
        "doctor",
        help="Check what is installed and what you can do with it",
        description=(
            "Report which pieces are present. Nothing here is fatal on its own: "
            "checking a config needs almost nothing, and running inference "
            "needs the DevKit."
        ),
    )
    sub.set_defaults()



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
        ("numpy", "draw the overlay on every frame", "a run", "the board provides it"),
        ("cv2", "draw the overlay on every frame", "a run", "the board provides it"),
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

    # pyneat lives in a venv of its own, so "not importable here" and "not on
    # this machine" are different answers and the fix differs completely.
    site, note = find_pyneat_env()
    reachable = have["pyneat"] or site is not None
    if not have["pyneat"]:
        print(f"  {mark_for(reachable)}  {'pyneat venv':<{width}}  {note}")
        have["pyneat"] = reachable

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
        (True, "sima-vision watch -- <task>", "run it on the board, watch it here"),
        (drawing and have["pyneat"], "sima-vision <task>", "run inference on the MLA"),
    ):
        print(f"  {mark_for(ok)}  {command:<30}  {what}")

    if not have["pyneat"]:
        print("\n" + missing_pyneat_message(note))
    elif site is not None:
        # site is <root>/lib/pythonX.Y/site-packages, so the venv root is three
        # up. Two up is <root>/lib, and `<root>/lib/bin/pip` helps nobody.
        root = site.parents[2]
        print(
            f"\npyneat is not in this interpreter, but {root} has it and will be\n"
            f"used automatically. To skip the search, install into that venv instead:\n"
            f"  {root}/bin/pip install sima-vision"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 2

    if args.command not in TASKS:
        try:
            if args.command == "doctor":
                return run_doctor()
            if args.command == "init":
                return run_init(args.task, args.out, args.force)
            if args.command == "fetch":
                return run_fetch(args.task, args.into)
            if args.command == "setup":
                return run_setup_network(args.host, args.apply)
            if args.command == "push":
                return run_push(args.paths, args.host, args.dest)
            if args.command == "pull":
                return run_pull(args.names, args.host, args.into)
            if args.command in ("remote", "watch"):
                # argparse.REMAINDER keeps the literal `--`; ssh does not want it.
                argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
                if args.command == "watch":
                    return run_watch(argv, args.host, args.to, args.port, args.sdp)
                return run_remote(argv, args.host)
        except KeyboardInterrupt:
            return 130
        except SystemExit as exc:
            # These carry a message, not a status: `raise SystemExit("...")` is
            # how the setup commands refuse. An int code is argparse's, and is
            # already the answer.
            if isinstance(exc.code, int):
                return exc.code
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

        early = task.early_exit(cfg, args)
        if early is not None:
            return early

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
        # load_runtime_dependencies already worked out which of the two this is
        # -- wrong machine, or right machine and the wrong interpreter -- and
        # said so. Anything else that fails to import gets the generic half.
        message = str(exc)
        if "pyneat" not in message:
            message = (
                f"{exc}\n"
                "A run needs numpy and OpenCV as well. On the DevKit both come "
                "from the board's\nsystem packages; anywhere else, use "
                "`sima-vision <task> --validate` instead."
            )
        print(f"[ERR] {message}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
