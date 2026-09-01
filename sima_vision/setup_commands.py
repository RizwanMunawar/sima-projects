"""``sima-vision init`` and ``sima-vision fetch``.

Between them these replace the part of the old workflow that was manual: copying
a config out of the repo by hand, and curl-ing sample clips into the right
directory. Neither needs the repo to be cloned.
"""

from __future__ import annotations

import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .config import packaged_config

#: Sample clips, already raw H.264 so they skip the Neat 0.3.0 demuxer bug.
SAMPLE_RELEASE = "https://github.com/RizwanMunawar/sima-projects/releases/download/0.0.1"
SAMPLE_VIDEOS = {
    "people-walking-outside-mall.h264": "1920x1080 @ 24 fps, 13 MB. The usual default",
    "people-walking-inside-mall.h264": "1920x1080 @ 30 fps, 1.2 MB. Quicker smoke test",
}

#: Where the SDK publishes compiled model packs, by task.
MODEL_BASE = "https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix"
MODELS = {
    "detect": ("yolo26-detection", "yolo26m-det-bf16-mla_tess-b1.tar.gz"),
    "segment": ("yolo26-segmentation", "yolo26m-seg-bf16-mla_tess-b1.tar.gz"),
    "fall": ("yolo26-detection", "yolo26m-det-bf16-mla_tess-b1.tar.gz"),
}


def model_command(task: str) -> str:
    """The one line that downloads the right model pack for a task.

    ``sima-cli download`` needs a community.sima.ai login and writes into the
    working directory, which is why this is printed rather than run: it is not
    ours to authenticate, and getting the directory wrong is the single most
    common way to end up with a pack the config cannot see.
    """
    directory, name = MODELS[task]
    return (
        f"mkdir -p assets/models && cd assets/models && \\\n"
        f"  sima-cli download {MODEL_BASE}/{directory}/{name} && cd ../.."
    )


def run_init(task: str, out: Path, force: bool) -> int:
    """Write the commented starter config for a task into the working directory."""
    source = packaged_config(task)
    if not source.is_file():  # pragma: no cover - guards a broken install
        raise SystemExit(f"no packaged config for {task!r}")
    if out.exists() and not force:
        raise SystemExit(
            f"{out} already exists. Pass --force to overwrite it, or -o to write "
            f"somewhere else."
        )
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, out)

    lines = source.read_text(encoding="utf-8").count("\n") + 1
    print(f"wrote {out.resolve()}  ({lines} lines, every setting documented)")
    print("\nEdit it, then:")
    print(f"  sima-vision preview --task {task}     # see the overlay, no board needed")
    print(f"  sima-vision {task}                    # run it, on the DevKit")
    return 0


def download(url: str, out: Path) -> bool:
    """Fetch one file, reporting progress. Returns False on any HTTP failure."""
    if out.exists():
        print(f"  have  {out}  ({out.stat().st_size / 1e6:.1f} MB)")
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    # A carriage-return progress line is only readable on a terminal. Piped to a
    # file or a CI log it just repeats the whole line hundreds of times.
    live = sys.stdout.isatty()
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with part.open("wb") as handle:
                while chunk := response.read(1 << 16):
                    handle.write(chunk)
                    done += len(chunk)
                    if live and total:
                        print(
                            f"\r  ...   {out.name}  {done / 1e6:5.1f} / {total / 1e6:.1f} MB",
                            end="", flush=True,
                        )
        print(f"{chr(13) if live else ''}  got   {out}  ({done / 1e6:.1f} MB)          ")
        part.replace(out)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        part.unlink(missing_ok=True)
        print(f"\r  FAIL  {out.name}: {exc}", file=sys.stderr)
        return False


def run_fetch(task: str, into: Path) -> int:
    """Download the sample clips, then say how to get the model.

    The clips are on a public GitHub release, so they can just be fetched. The
    model packs are behind a community.sima.ai login, so the command is printed
    for you to run rather than attempted here.
    """
    print(f"sample clips -> {(into / 'videos').resolve()}")
    ok = True
    for name, what in SAMPLE_VIDEOS.items():
        ok &= download(f"{SAMPLE_RELEASE}/{name}", into / "videos" / name)
        print(f"        {what}")

    print("\nNow the model. It needs a community.sima.ai login, so run this yourself:\n")
    print("  sima-cli login")
    print(f"  {model_command(task)}")
    print("\nThen:\n")
    default = next(iter(SAMPLE_VIDEOS))
    _, model_name = MODELS[task]
    print(f"  sima-vision {task} \\")
    print(f"    --source {into.as_posix()}/videos/{default} \\")
    print(f"    --model {into.as_posix()}/models/{model_name}")
    return 0 if ok else 1
