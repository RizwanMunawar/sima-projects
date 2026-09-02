"""The Python API: the same three verbs the command line has.

Every keyword these accept is derived from the CLI's own flags, so
``--blur-strength 81`` and ``blur_strength=81`` cannot drift apart -- there is
one table and it is built from the parser at import time.

    from sima_vision import run, preview, validate

    preview("segment", out="blur.png", blur_strength=81)
    validate("detect", conf=0.5)
    run("detect", source="clip.h264", model="yolo26m-det.tar.gz", conf=0.5)

``preview`` and ``validate`` need no board. ``run`` needs the DevKit, because
that is where the MLA is.
"""

from __future__ import annotations

import argparse
from functools import cache
from pathlib import Path

from .tasks import TASKS


def _task(name: str):
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}. Choose one of: {', '.join(TASKS)}")
    return TASKS[name]()


def _alias_table(task) -> tuple[dict[str, str], set[str]]:
    """Cached by task class -- the flags cannot change between calls."""
    return _build_alias_table(type(task))


@cache
def _build_alias_table(task_cls) -> tuple[dict[str, str], set[str]]:
    """Map Python keyword -> config path, straight off this task's CLI flags.

    Returns the table and the set of keywords whose boolean has to be flipped:
    ``--send`` turns ``alerts.dry_run`` *off*, so ``send=True`` must write
    False. ``--no-save`` is registered as ``save`` for the same reason, in the
    other direction -- nobody wants to write ``no_save=True``.
    """
    from .cli import add_shared_arguments

    task = task_cls()
    parser = argparse.ArgumentParser(add_help=False)
    add_shared_arguments(parser)
    task.add_arguments(parser.add_argument_group("task"))

    aliases: dict[str, str] = {}
    inverted: set[str] = set()
    for action in parser._actions:
        if "." not in action.dest:
            continue
        turns_off = getattr(action, "const", None) is False
        for option in action.option_strings:
            negative = option.startswith("--no-")
            name = option.lstrip("-").replace("-", "_")
            if negative:
                name = name[3:]          # no_save -> save
            # Two flags claiming one keyword would silently hide a setting --
            # `--video PATH` and `--no-video` both wanted `video` once, and the
            # path became unreachable from Python.
            claimed = aliases.get(name)
            if claimed is not None and claimed != action.dest:
                raise RuntimeError(
                    f"{task.name}: {option} wants the keyword {name!r}, which "
                    f"already means {claimed!r}. Rename one of the flags."
                )
            aliases[name] = action.dest
            # `--send` reads as "do send", but it clears a dry_run flag.
            if turns_off and not negative:
                inverted.add(name)
    return aliases, inverted


def settings_to_overrides(task, settings: dict) -> dict:
    """Translate Python keywords into the dotted config paths the loader takes.

    Dotted keys are passed through untouched, so anything the aliases do not
    cover is still reachable::

        run("detect", **{"runtime.output_buffers": 2})

    Raises:
        TypeError: On a keyword that is neither an alias nor a dotted path,
            listing the near misses.
    """
    aliases, inverted = _alias_table(task)
    overrides: dict = {}
    for key, value in settings.items():
        if "." in key:
            overrides[key] = value
            continue
        path = aliases.get(key)
        if path is None:
            near = sorted(name for name in aliases if key in name or name in key)
            hint = f" Did you mean: {', '.join(near[:5])}?" if near else ""
            raise TypeError(f"{task.name}() got an unexpected setting {key!r}.{hint}")
        overrides[path] = not value if key in inverted else value
    return overrides


def load(task: str, config: str | Path | None = None, use_config_file: bool = True,
         **settings):
    """Resolve a configuration the way the CLI does, and validate it.

    Args:
        task: ``detect``, ``segment`` or ``fall``.
        config: Path to a config file, or None to look for ``./config.yaml``.
        use_config_file: False ignores any file, like ``--no-config``.
        **settings: Anything the CLI takes, as a keyword.

    Returns:
        A validated config for the task.
    """
    handle = _task(task)
    return handle.load(
        Path(config) if config else None,
        settings_to_overrides(handle, settings),
        use_file=use_config_file,
    )


def validate(task: str, config: str | Path | None = None, **settings):
    """Check a configuration without a board. Raises ValueError if it is wrong.

    Returns:
        The resolved config, so it can be inspected.
    """
    return load(task, config, **settings)


def run(task: str, config: str | Path | None = None, **settings) -> int:
    """Run one task to completion. **Needs the DevKit.**

    A clip or model archive that is missing is downloaded into ``assets/``
    first; see :mod:`sima_vision.assets`. ``validate`` and ``preview`` resolve
    the same paths and never fetch anything.

    Args:
        task: ``detect``, ``segment`` or ``fall``.
        config: Path to a config file, or None to look for ``./config.yaml``.
        **settings: Anything the CLI takes, as a keyword.

    Returns:
        The number of frames processed.
    """
    import os

    from .runloop import Stopper
    from .runtime import load_runtime_dependencies

    handle = _task(task)
    cfg = load(task, config, **settings)
    load_runtime_dependencies()
    # The same two side effects the command line has, so `run(...)` and
    # `sima-vision <task>` really are the same run.
    if cfg.profile:
        os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
        os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
    if cfg.save_enable:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    return handle.run(cfg, Stopper())


def preview(task: str = "detect", config: str | Path | None = None,
            out: str | Path = "preview.png", source: str | None = None,
            size: tuple[int, int] = (1280, 720), use_config_file: bool = True,
            **settings) -> Path:
    """Draw the overlay a config produces, and write it to a PNG. **No board.**

    Runs no model: the detections are synthetic and exist only to give the
    drawing code something to draw.

    Args:
        task: ``detect``, ``segment`` or ``fall``.
        config: Path to a config file, or None to look for ``./config.yaml``.
        out: Where to write the PNG.
        source: An image or video to draw on. None paints a synthetic scene.
        size: Synthetic scene size, ignored when ``source`` is readable.
        use_config_file: False ignores any file, like ``--no-config``.
        **settings: Anything the CLI takes, as a keyword.

    Returns:
        The path written.
    """
    from . import runtime
    from .cli import load_drawing_dependencies
    from .scene import build_frame, render
    from .sinks import load_labels

    load_drawing_dependencies()
    handle = _task(task)
    config_path = Path(config) if config else None
    cfg = handle.load(
        config_path, settings_to_overrides(handle, settings), use_file=use_config_file
    )

    frame, subjects, _origin, _ = build_frame(source, size)
    annotated = render(handle, cfg, frame, subjects, load_labels(cfg.labels_path))
    out = Path(out)
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    if not runtime.cv2.imwrite(str(out), annotated):
        raise RuntimeError(f"could not write {out}")
    return out
