"""Where the sample clips and model archives live, and how they get there.

There is one ``assets/`` directory for the whole project rather than one inside
each task folder. A clip is a clip: the same 13 MB of people walking through a
mall feeds ``detect``, ``segment`` and ``fall``, and the detect archive is
shared by ``detect`` and ``fall`` outright. Three copies of it said nothing that
one copy does not.

``--source`` and ``--model`` therefore take one of three things:

1. a local path -- used as given
2. an ``http(s)`` URL -- downloaded into ``assets/`` once, then reused
3. nothing at all -- the task's default, downloaded on first run

Case 3 is what makes ``sima-vision detect`` work on its own. The clips are on a
public GitHub release so they are simply fetched. The model packs are behind a
`community.sima.ai <https://community.sima.ai>`_ login -- the download URL
answers a plain GET with a 302 to ``auth.sima.ai`` -- so those go through
``sima-cli``, which already holds that login, and fall back to printing the
command when it is not installed.

Nothing here runs at config time. ``--validate`` resolves the same paths and
never touches the network; only :meth:`Task.run
<sima_vision.tasks.base.Task.run>` calls :func:`ensure_assets`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

from .console import console, human_bytes

#: Overrides where downloads land. Default ``./assets`` in the working directory.
ASSETS_ENV = "SIMA_VISION_ASSETS"

#: Sample clips, already raw H.264 so they skip the Neat 0.3.0 demuxer bug.
SAMPLE_RELEASE = "https://github.com/RizwanMunawar/sima-projects/releases/download/0.0.1"
SAMPLE_VIDEOS = {
    "people-walking-outside-mall.h264": "1920x1080 @ 24 fps, 13 MB. The usual default",
    "people-walking-inside-mall.h264": "1920x1080 @ 30 fps, 1.2 MB. Quicker smoke test",
}

#: Where the SDK publishes compiled model packs.
MODEL_BASE = "https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix"


@dataclass(frozen=True)
class TaskAssets:
    """What one task runs on when it is given nothing.

    Attributes:
        model_dir: Model pack directory under :data:`MODEL_BASE`.
        model_file: Archive name, which is also its name inside ``assets/models``.
        clip: Sample clip name, a key of :data:`SAMPLE_VIDEOS`.
    """

    model_dir: str
    model_file: str
    clip: str


#: Task name -> its default model and clip. ``detect`` and ``fall`` share a head.
CATALOGUE: dict[str, TaskAssets] = {
    "detect": TaskAssets(
        "yolo26-detection",
        "yolo26m-det-bf16-mla_tess-b1.tar.gz",
        "people-walking-outside-mall.h264",
    ),
    "segment": TaskAssets(
        "yolo26-segmentation",
        "yolo26m-seg-bf16-mla_tess-b1.tar.gz",
        "people-walking-outside-mall.h264",
    ),
    "fall": TaskAssets(
        "yolo26-detection",
        "yolo26m-det-bf16-mla_tess-b1.tar.gz",
        "people-walking-inside-mall.h264",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Locations
# ─────────────────────────────────────────────────────────────────────────────


def assets_root() -> Path:
    """The ``assets/`` directory, honouring ``$SIMA_VISION_ASSETS``.

    Read on every call rather than cached at import, so setting the variable
    from a test -- or between two calls in one process -- takes effect.
    """
    return Path(os.environ.get(ASSETS_ENV) or "assets")


def videos_dir() -> Path:
    return assets_root() / "videos"


def models_dir() -> Path:
    return assets_root() / "models"


def default_model_path(task: str) -> str:
    """Where this task's model archive is expected, as a string for the config."""
    return (models_dir() / CATALOGUE[task].model_file).as_posix()


def default_source_uri(task: str) -> str:
    """Where this task's sample clip is expected, as a string for the config."""
    return (videos_dir() / CATALOGUE[task].clip).as_posix()


def model_url(task: str) -> str:
    entry = CATALOGUE[task]
    return f"{MODEL_BASE}/{entry.model_dir}/{entry.model_file}"


def model_command(task: str) -> str:
    """The one line that downloads the right model pack for a task.

    ``sima-cli download`` needs a community.sima.ai login and writes into the
    working directory, which is why this exists as a printable string as well as
    something :func:`ensure_model` runs: getting the directory wrong is the
    single most common way to end up with a pack the config cannot see.
    """
    models = models_dir().as_posix()
    # A subshell rather than `cd there && ... && cd back`: the working directory
    # you started in is where the rest of the commands expect to be, and one
    # failed step in the middle of that chain would strand you in assets/models.
    return f"mkdir -p {models} && (cd {models} && sima-cli download {model_url(task)})"


def is_url(value: str) -> bool:
    """True for something to download. ``rtsp://`` is a stream, not a file."""
    return str(value).startswith(("http://", "https://"))


# ─────────────────────────────────────────────────────────────────────────────
# Downloading
# ─────────────────────────────────────────────────────────────────────────────


def say(step, text: str) -> None:
    """One line of progress, under the step that asked for it if there is one."""
    if step is not None:
        step.detail(text)
    else:
        console.info(text)


def download(url: str, out: Path, step=None) -> bool:
    """Fetch one file, reporting progress. Returns False on any HTTP failure."""
    if out.exists():
        say(step, f"have  {out}  ({human_bytes(out.stat().st_size)})")
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    say(step, f"get   {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with part.open("wb") as handle:
                while chunk := response.read(1 << 16):
                    handle.write(chunk)
                    done += len(chunk)
                    console.progress(out.name, done, total)
        console.progress_done()
        # A server that closes early, or a proxy that truncates, ends the read
        # loop exactly like a finished transfer does. Without this the partial
        # file is renamed into place and every later run reuses it, because the
        # first thing this function does is trust a file that already exists.
        if total and done != total:
            part.unlink(missing_ok=True)
            console.error(
                f"{out.name}: got {done} of {total} bytes, the transfer was cut short"
            )
            return False
        part.replace(out)
        say(step, f"got   {out}  ({human_bytes(done)})")
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        console.progress_done()
        part.unlink(missing_ok=True)
        console.error(f"{out.name}: {exc}")
        return False


def cache_name(url: str) -> str:
    """A local filename for a URL that cannot collide with another URL's.

    ``--source https://a.example/clip.h264`` and the same name on another host
    are different videos. Keying the cache on the last path segment alone meant
    the second one silently ran the first one's footage, and the download was
    skipped because the file was already there.

    The digest goes before the extension rather than after, so the suffix still
    says what the file is and `.tar.gz` survives intact::

        clip.h264            ->  clip-1a2b3c4d.h264
        yolo26m-det.tar.gz   ->  yolo26m-det-1a2b3c4d.tar.gz
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    name = Path(url.split("?")[0]).name or "download"
    head, dot, tail = name.partition(".")
    return f"{head}-{digest}{dot}{tail}"


def fetch(url: str, out: Path, what: str, step=None) -> Path:
    """Download to ``out``, or raise. The insisting version of :func:`download`."""
    if not download(url, out, step):
        raise RuntimeError(
            f"could not download the {what} from {url}\n"
            f"  wanted: {out}\n"
            "Fetch it by hand and pass the local path instead."
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────────


def ensure_source(uri: str, source_type: str = "video", step=None) -> str:
    """Make ``source.uri`` name a file that exists, downloading if it has to.

    Args:
        uri: The resolved ``source.uri``: a path, an ``http(s)`` URL, or one of
            the sample clip paths the defaults fill in.
        source_type: Only ``video`` reads a file. An RTSP URL or a camera is
            handed back untouched.
        step: The console step to report under, if there is one.

    Returns:
        A local path, or ``uri`` unchanged when there is nothing to fetch. A
        path that is simply missing is *also* handed back unchanged, so the
        error comes from :func:`sima_vision.media.check_source_file`, which
        knows how to describe it.
    """
    if source_type != "video" or not uri:
        return uri
    if is_url(uri):
        return str(fetch(uri, videos_dir() / cache_name(uri), "source video", step))
    path = Path(uri)
    if path.exists():
        say(step, f"have  {uri}  ({human_bytes(path.stat().st_size)})")
        return uri
    # A default, or a path the user wrote that happens to name a sample clip.
    if path.name in SAMPLE_VIDEOS:
        fetch(f"{SAMPLE_RELEASE}/{path.name}", path, "sample clip", step)
    return uri


def ensure_model(path: str, task: str, step=None) -> str:
    """Make ``model.path`` name an archive that exists, downloading if it has to.

    A URL is fetched directly. Anything already on disk is used as it stands.
    The remaining case is the default -- the task's own archive, not yet
    downloaded -- and that one goes through ``sima-cli``, because a plain GET on
    the pack URL answers with a login redirect rather than a tarball.

    Raises:
        RuntimeError: When the archive is missing and cannot be fetched, with
            the command to run by hand.
    """
    if is_url(path):
        return str(fetch(path, models_dir() / cache_name(path), "model archive", step))
    if not path:
        return path
    target = Path(path)
    if target.exists():
        say(step, f"have  {path}  ({human_bytes(target.stat().st_size)})")
        return path

    entry = CATALOGUE.get(task)
    if entry is None or target.name != entry.model_file:
        # Not something this task knows how to fetch: a name we have no URL for.
        raise RuntimeError(
            f"model archive not found: {path}\n"
            f"  launched from: {Path.cwd()}\n"
            "Pass --model with a path or an https URL, or leave it off to use "
            f"the default for {task}."
        )

    url = model_url(task)
    if shutil.which("sima-cli") is None:
        raise RuntimeError(
            f"model archive not found: {path}\n"
            "The model packs need a community.sima.ai login, and `sima-cli` is "
            "not on PATH here,\nso it cannot be fetched for you. Run:\n\n"
            f"  sima-cli login\n  {model_command(task)}\n\n"
            "Or pass --model with an https URL you can reach."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    say(step, f"get   {url}")
    say(step, "      via sima-cli, which holds your community.sima.ai login")
    result = subprocess.run(  # noqa: S603
        ["sima-cli", "download", url],
        cwd=target.parent,
        check=False,
        # Left alone, sima-cli opens with "a newer version is available, update
        # now? [Y/n]" and waits. Nothing is watching that prompt in the middle
        # of a run, and answering no aborts the download with it. Its own
        # message names this variable as the way off.
        env={**os.environ, "SIMA_CLI_CHECK_FOR_UPDATE": "0"},
    )
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(
            f"`sima-cli download` did not produce {target}\n"
            "It needs a community.sima.ai login, and the board needs a route to the "
            "internet\nthrough the PC it is cabled to. Run `sima-cli login` and try "
            "again, or download\nthe pack on your PC and `sima-vision push` it over."
        )
    say(step, f"got   {path}  ({human_bytes(target.stat().st_size)})")
    return path


def ensure_assets(cfg, task: str, step=None):
    """Resolve ``model.path`` and ``source.uri`` to files that exist.

    Called once, from :meth:`Task.run <sima_vision.tasks.base.Task.run>`, so
    that everything which does not run inference -- ``--validate`` and the
    Python ``validate()`` -- stays offline.

    Returns:
        The config, or a copy of it with the two paths replaced.
    """
    source = ensure_source(cfg.source_uri, cfg.source_type, step)
    model = ensure_model(cfg.model_path, task, step)
    if (source, model) == (cfg.source_uri, cfg.model_path):
        return cfg
    return replace(cfg, source_uri=source, model_path=model)
