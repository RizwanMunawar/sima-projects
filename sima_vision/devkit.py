"""Moving files between your PC and the DevKit, and running a task over SSH.

Three commands wrap `ssh` and `scp` so the awkward parts stop being yours:

  sima-vision push clip.h264      your PC  ->  the board
  sima-vision pull                the board  ->  your PC
  sima-vision remote -- detect    run it there, watch it here

The awkward parts, in order of how much time each one costs:

1. On Windows, `scp D:\\clips\\a.h264 sima@ip:~` fails with "could not resolve
   hostname d:", because scp reads everything before the first colon as a host.
   So a push never passes a full local path: it groups the files by folder and
   runs scp from inside each one, with bare filenames.
2. `ssh host cmd` without a pty means Ctrl-C never reaches the program. The run
   keeps the MLA and the next launch fails with a busy device. `remote` always
   passes -tt.
3. A pull of "whatever the run produced" cannot be written as one scp, because
   scp fails the whole transfer on a name that is not there and the outputs
   depend on which task ran. So the names are listed over ssh first, and only
   what exists is fetched.

The host is `user@address`, from --host or $SIMA_VISION_DEVKIT. Authentication
is ssh's own business: an agent, a key, or it asks. Nothing here handles
passwords, and nothing here stores one.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

#: Saves retyping `--host` on every command.
DEVKIT_ENV = "SIMA_VISION_DEVKIT"

#: What a run leaves in its working directory, across all three tasks. `pull`
#: with no arguments asks for these and takes whatever is actually there, so it
#: does not need to be told which task ran.
OUTPUTS = (
    "detections.mp4", "segmentation.mp4", "falls.mp4",
    "detections.avi", "segmentation.avi", "falls.avi",
    "frames", "alerts", "config.yaml",
)


def resolve_host(host: str | None) -> str:
    """The board to talk to, from the flag or the environment."""
    found = host or os.environ.get(DEVKIT_ENV, "")
    if not found:
        raise SystemExit(
            "no DevKit address. Pass --host, or set it once for the session:\n\n"
            "  export SIMA_VISION_DEVKIT=sima@192.168.137.50      # macOS, Linux\n"
            '  $env:SIMA_VISION_DEVKIT = "sima@192.168.137.50"    # PowerShell\n\n'
            "The board takes a DHCP address that changes between reboots. The\n"
            "serial console prints it, and so does `arp -a` on the PC it is\n"
            "cabled to."
        )
    return found.strip()


def require(tool: str) -> str:
    """Find `ssh` or `scp`, or explain how to get them on this platform."""
    found = shutil.which(tool)
    if found:
        return found
    raise SystemExit(
        f"`{tool}` is not on PATH, and this command is a wrapper around it.\n"
        "  Windows 10/11: Settings > Apps > Optional features > OpenSSH Client\n"
        "  macOS: it ships with the system\n"
        "  Debian/Ubuntu: sudo apt install openssh-client"
    )


def report(command: list[str], cwd: Path | None = None) -> None:
    where = f"  (from {cwd})" if cwd is not None else ""
    print(f"$ {' '.join(command)}{where}", flush=True)


def remote_paths(host: str, names) -> list[str]:
    """Which of `names` exist in the board's home directory.

    One `ls` rather than one probe per name, and a missing name is not an error
    here: `pull` is asking a vague question on purpose.
    """
    listing = " ".join(shlex.quote(str(name)) for name in names)
    command = [require("ssh"), host, f"ls -d -- {listing} 2>/dev/null"]
    result = subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    if result.returncode not in (0, 1, 2):
        # 1 and 2 are `ls` saying "some of those are not there", which is fine.
        # Anything else is ssh itself failing, and its message is the useful one.
        raise SystemExit(f"could not reach {host}:\n{result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def run_push(paths: list[Path], host: str | None, dest: str) -> int:
    """Copy local files or folders to the board.

    Grouped by parent folder and run from inside it, which is what keeps a
    Windows drive letter from being read as a hostname. See the module docstring.
    """
    host = resolve_host(host)
    scp = require("scp")

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            "not found: " + ", ".join(str(p) for p in missing)
        )

    groups: dict[Path, list[str]] = defaultdict(list)
    for path in paths:
        resolved = path.resolve()
        groups[resolved.parent].append(resolved.name)

    target = f"{host}:{dest}"
    for parent, names in groups.items():
        command = [scp, "-r", *names, target]
        report(command, parent)
        result = subprocess.run(command, cwd=parent, check=False)  # noqa: S603
        if result.returncode != 0:
            return result.returncode

    print(f"\ncopied {sum(len(n) for n in groups.values())} item(s) to {target}")
    return 0


def run_pull(names: list[str], host: str | None, into: Path) -> int:
    """Copy results back from the board.

    With no names, asks for everything a run of any task could have left and
    takes what is there.
    """
    host = resolve_host(host)
    scp = require("scp")
    wanted = names or list(OUTPUTS)

    found = remote_paths(host, wanted)
    if not found:
        listed = ", ".join(wanted[:4]) + ("..." if len(wanted) > 4 else "")
        print(
            f"nothing to pull: {host} has none of {listed}\n"
            "Run a task there first, or name the file you want.",
            file=sys.stderr,
        )
        return 1

    into.mkdir(parents=True, exist_ok=True)
    sources = [f"{host}:{shlex.quote(name)}" for name in found]
    # cwd=into with "." as the destination, for the same reason push runs from
    # the file's own folder: a Windows destination path would carry a colon.
    command = [scp, "-r", *sources, "."]
    report(command, into)
    result = subprocess.run(command, cwd=into, check=False)  # noqa: S603
    if result.returncode != 0:
        return result.returncode

    print(f"\npulled into {into.resolve()}:")
    for name in found:
        local = into / name
        size = local.stat().st_size / 1e6 if local.is_file() else 0
        print(f"  {name}" + (f"  ({size:.1f} MB)" if size else ""))
    return 0


def run_remote(argv: list[str], host: str | None) -> int:
    """Run `sima-vision <argv>` on the board, with a pty so Ctrl-C works.

    Without -tt, Ctrl-C stays on your PC. The task keeps running on the board,
    keeps the MLA, and the next launch fails with a busy device.
    """
    host = resolve_host(host)
    if not argv:
        raise SystemExit(
            "nothing to run. Put the task after --:\n"
            "  sima-vision remote -- detect --frames 200"
        )
    inner = " ".join(shlex.quote(token) for token in ["sima-vision", *argv])
    command = [require("ssh"), "-tt", host, inner]
    report(command)
    return subprocess.run(command, check=False).returncode  # noqa: S603
