"""``sima-vision init`` and ``sima-vision fetch``.

Between them these replace the part of the old workflow that was manual: copying
a config out of the repo by hand, and curl-ing sample clips into the right
directory. Neither needs the repo to be cloned.

``fetch`` is now the eager version of what a run does on its own -- everything
here is also reachable through :func:`sima_vision.assets.ensure_assets`, which
fetches the same files lazily on the first run. It stays because getting the
13 MB clip out of the way while you still have good wifi is worth a command, and
because it is the natural place to print the model line.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .assets import CATALOGUE, SAMPLE_RELEASE, SAMPLE_VIDEOS, download, model_command
from .config import packaged_config


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
    print(f"  sima-vision {task} --validate         # check it, no board needed")
    print(f"  sima-vision watch -- {task}           # run it on the board, watch here")
    return 0


def run_fetch(task: str, into: Path) -> int:
    """Download the sample clips, then say how to get the model.

    The clips are on a public GitHub release, so they can just be fetched. The
    model packs are behind a community.sima.ai login, so the command is printed
    for you to run rather than attempted here -- a run will try it through
    ``sima-cli`` on its own, and this way you can do it first and watch it work.
    """
    print(f"sample clips -> {(into / 'videos').resolve()}")
    ok = True
    for name, what in SAMPLE_VIDEOS.items():
        ok &= download(f"{SAMPLE_RELEASE}/{name}", into / "videos" / name)
        print(f"        {what}")

    print("\nNow the model. It needs a community.sima.ai login, so run this yourself:\n")
    print("  sima-cli login")
    print(f"  {model_command(task, into)}")
    print("\nThen, from here:\n")
    print(f"  sima-vision {task}")
    print("\nThat picks both of them up on its own. To use something else:\n")
    entry = CATALOGUE[task]
    print(f"  sima-vision {task} \\")
    print(f"    --source {(into / 'videos' / entry.clip).as_posix()} \\")
    print(f"    --model {(into / 'models' / entry.model_file).as_posix()}")
    return 0 if ok else 1
