"""Fall detection with SMTP alerts on the SiMa Modalix DevKit -- compatibility entry point.

The application itself now lives in the ``sima_vision`` package, so all three
apps share one pipeline instead of three copies of it. This file stays so the
commands in this app's README keep working unchanged::

    python3 src/app.py --config config.yaml
    python3 src/app.py --config config.yaml --validate-config

The same run, through the installed CLI::

    sima-vision fall --config config.yaml
    sima-vision fall --config config.yaml --validate

``--validate-config`` is accepted here as an alias for ``--validate``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly means there is no installed package to import, so
# put the repo's src/ on the path first. A real ``pip install`` shadows this,
# and an editable install makes it a no-op.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sima_vision.cli import main as _main  # noqa: E402

TASK = "fall"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # The old flag name, kept working rather than silently ignored.
    argv = ["--validate" if a == "--validate-config" else a for a in argv]
    return _main([TASK, *argv])


if __name__ == "__main__":
    raise SystemExit(main())
