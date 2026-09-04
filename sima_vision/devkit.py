"""Moving files between your PC and the DevKit, and running a task over SSH.

Four commands wrap `ssh` and `scp` so the awkward parts stop being yours:

  sima-vision push clip.h264      your PC  ->  the board
  sima-vision pull                the board  ->  your PC
  sima-vision remote -- detect    run it there, output in your terminal
  sima-vision watch  -- detect    run it there, live video on your screen

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

import ipaddress
import os
import shlex
import shutil
import socket
import subprocess
from collections import defaultdict
from pathlib import Path

from . import __version__
from .console import console, human_bytes

#: Saves retyping `--host` on every command.
DEVKIT_ENV = "SIMA_VISION_DEVKIT"

#: Where the board sends the live feed, matching the defaults in
#: :class:`~sima_vision.config.BaseConfig`. Video is H.264 in RTP; the metadata
#: alongside it is JSON, one datagram per frame.
VIDEO_PORT = 9000
METADATA_PORT = 9100

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
    """Echo the ssh or scp being run. These wrap other programs, and hiding
    which one would make their failures unattributable."""
    where = f"  (from {cwd})" if cwd is not None else ""
    style = console.style
    console.write(f"  {style.paint('$', style.dim)} {' '.join(command)}{where}")


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
    console.banner(f"sima-vision {__version__}", f"push -> {host}")
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

    console.write()
    console.success(f"copied {sum(len(n) for n in groups.values())} item(s) to {target}")
    return 0


def run_pull(names: list[str], host: str | None, into: Path) -> int:
    """Copy results back from the board.

    With no names, asks for everything a run of any task could have left and
    takes what is there.
    """
    host = resolve_host(host)
    console.banner(f"sima-vision {__version__}", f"pull <- {host}")
    scp = require("scp")
    wanted = names or list(OUTPUTS)

    found = remote_paths(host, wanted)
    if not found:
        listed = ", ".join(wanted[:4]) + ("..." if len(wanted) > 4 else "")
        console.error(
            f"nothing to pull: {host} has none of {listed}\n"
            "Run a task there first, or name the file you want."
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

    console.write()
    console.success(f"pulled into {into.resolve()}:")
    for name in found:
        local = into / name
        size = human_bytes(local.stat().st_size) if local.is_file() else ""
        console.info(f"  {name}" + (f"  ({size})" if size else ""))
    return 0


def address_of(host: str) -> str:
    """The bare address out of `user@address`."""
    return host.rpartition("@")[2]


def address_the_board_sees(host: str) -> str | None:
    """Ask the board where our SSH connection comes from.

    This is the whole question -- "which of this machine's addresses can the
    board send video to" -- answered by the only party that knows, instead of
    inferred. `$SSH_CONNECTION` is `<client ip> <client port> <server ip>
    <server port>`, so the first field is us, as seen from there.

    Returns None if it cannot be asked, leaving :func:`local_ip_seen_by` to
    guess. A wrong answer here is silent: the run starts, the board streams
    into a void, and no video ever appears.
    """
    result = subprocess.run(  # noqa: S603
        [require("ssh"), host, "echo $SSH_CONNECTION"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False, timeout=30,
    )
    if result.returncode != 0:
        return None
    fields = result.stdout.split()
    if not fields:
        return None
    try:
        return str(ipaddress.ip_address(fields[0]))
    except ValueError:
        return None


def local_ip_seen_by(host: str) -> str:
    """A guess at this PC's address on the board's network. The fallback.

    Opening a UDP socket towards the board and asking what the kernel bound
    consults the real routing table, and connect() on UDP sends nothing, so it
    costs no packets.

    It is a guess because it is only as good as the routing table's confidence.
    Watch what happens when the board is *not* answering ARP: the on-link route
    exists, resolution fails, the stack falls back to the default route, and
    this cheerfully returns the Wi-Fi address, which the board cannot reach.
    That is why :func:`address_the_board_sees` is tried first.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((address_of(host), 9))     # discard port, nothing is sent
        return probe.getsockname()[0]
    except OSError as exc:
        raise SystemExit(
            f"could not work out which of this machine's addresses {host} would "
            f"reach:\n  {exc}\n"
            "Pass --to with the address the board should send video to."
        ) from None
    finally:
        probe.close()


def player_commands(port: int, sdp: Path) -> list[tuple[str, str]]:
    """How to open the stream, best first. Each is (tool, command).

    The board sends H.264 in RTP, which needs to be told what it is: RTP carries
    no container and a dynamic payload type means nothing on its own. ffplay
    learns it from an SDP file, GStreamer from caps on the command line. Neither
    is shipped here on purpose -- decoding video is not this package's job, and
    both of these are better at it than anything that would fit in it.
    """
    return [
        ("ffplay", f'ffplay -hide_banner -fflags nobuffer -flags low_delay '
                   f'-protocol_whitelist file,rtp,udp -i "{sdp}"'),
        ("gst-launch-1.0",
         f"gst-launch-1.0 udpsrc port={port} "
         f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" '
         f"! rtpjitterbuffer latency=50 ! rtph264depay ! avdec_h264 ! "
         f"videoconvert ! autovideosink sync=false"),
        ("vlc", f'vlc --network-caching=50 "{sdp}"'),
    ]


def write_sdp(path: Path, port: int) -> Path:
    """The three lines ffplay and VLC need to make sense of the RTP stream.

    Payload type 96 and the 90 kHz clock are what `h264_rtp_udp_from_raw` on
    the board sends; they are fixed, not guesses.
    """
    path.write_text(
        "v=0\n"
        "o=- 0 0 IN IP4 0.0.0.0\n"
        "s=sima-vision\n"
        "c=IN IP4 0.0.0.0\n"
        "t=0 0\n"
        f"m=video {port} RTP/AVP 96\n"
        "a=rtpmap:96 H264/90000\n",
        encoding="utf-8",
    )
    return path


def run_watch(argv: list[str], host: str | None, to: str | None, port: int,
              sdp_path: Path | None) -> int:
    """Run a task on the board with its live video pointed at this machine.

    The board already knows how to stream what it is drawing: that is the
    Insight feed, real frames with the real overlay, and it is off by default
    only because it is pointed at the board's own localhost where nothing is
    listening. All this does is aim it here and start the run.

    Nothing decodes video in this process. The SDP is written and the exact
    player command printed, because ffplay and GStreamer already do that job
    properly and a half-hearted decoder in here would be worse at it.
    """
    host = resolve_host(host)
    if not argv:
        raise SystemExit(
            "nothing to run. Put the task after --:\n"
            "  sima-vision watch -- detect"
        )
    console.banner(f"sima-vision {__version__}", f"watch {' '.join(argv)} on {host}")

    # Asked, then guessed. The board's own view is right by construction; the
    # routing-table guess is only right while the board is answering ARP, and
    # getting it wrong produces a run that streams into a void.
    if to:
        target, how = to, "given with --to"
    else:
        asked = address_the_board_sees(host)
        target = asked or local_ip_seen_by(host)
        how = "as the board sees us" if asked else "guessed from the routing table"

    # Absolute: the player is opened in another terminal, which will not
    # necessarily be in this directory.
    sdp = write_sdp(sdp_path or Path("sima-vision.sdp"), port).resolve()

    console.info(f"live video: {host} -> {target}:{port}   ({how})")
    console.info(f"            metadata on {METADATA_PORT}, same address")
    console.info(f"wrote {sdp}")
    console.write()

    players = player_commands(port, sdp)
    installed = [(tool, command) for tool, command in players if shutil.which(tool)]

    console.info("Open this in a second terminal, then come back:")
    if installed:
        first, *rest = installed
        console.write()
        console.info(f"  {first[1]}")
        for tool, command in rest:
            console.write()
            console.info(f"  or with {tool}:")
            console.info(f"  {command}")
    else:
        console.write()
        console.info(f"  {players[0][1]}")
        console.write()
        console.info(f"  ...once one of these is installed: "
                     f"{', '.join(tool for tool, _ in players)}")
        console.info("  Windows: winget install Gyan.FFmpeg")
        console.info("  macOS:   brew install ffmpeg")
        console.info("  Ubuntu:  sudo apt install ffmpeg")
    console.write()

    # --insight is off by default because its encoder shares the codec daemon
    # with the decoder and can stall a file run. Watching is the case where you
    # have decided that is worth it, so it is turned on here rather than being
    # something else to remember.
    inner = ["sima-vision", *argv, "--insight", "--insight-host", target]
    command = [require("ssh"), "-tt", host, " ".join(shlex.quote(t) for t in inner)]
    report(command)
    return subprocess.run(command, check=False).returncode  # noqa: S603


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
    console.banner(f"sima-vision {__version__}", f"remote {' '.join(argv)} on {host}")
    inner = " ".join(shlex.quote(token) for token in ["sima-vision", *argv])
    command = [require("ssh"), "-tt", host, inner]
    report(command)
    return subprocess.run(command, check=False).returncode  # noqa: S603
