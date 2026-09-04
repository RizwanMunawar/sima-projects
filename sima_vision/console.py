"""What the user actually sees.

Every command prints the same shape, because every command is doing the same
kind of thing: a numbered list of steps, each one saying what it is about to do
*before* it does it, and what came of it afterwards. Nothing waits in silence.

    sima-vision 0.1.0  detect

      [1/7] environment  checking this machine
            -> Modalix DevKit  aarch64  python 3.11.2
      [2/7] pyneat       locating the Neat runtime
            -> 0.3.0  using pyneat from /home/sima/pyneat
      ...
      [7/7] pipeline     building the Neat graph
            -> yolo_detector ready (3.1s)

Four channels, and which one a line goes to is a decision about the reader:

* :meth:`Console.step` and :meth:`Console.write` are *progress*. ``--quiet``
  drops them, because someone who passed ``--quiet`` is not reading along.
* :meth:`Console.report` is a *result*. It survives ``--quiet``: a run that
  said nothing at all would be a broken one.
* :meth:`Console.warn` is something to act on, so it also survives, and stays
  on stdout in step order -- a warning read out of sequence is half a message.
* :meth:`Console.error` is the only thing on stderr.

Everything here is ASCII. Python picks an encoder for stdout the moment it is
redirected, so one smart quote turns `sima-vision detect > log.txt` on a Windows
console into a UnicodeEncodeError, and that is a crash in the logging taking a
run with it. Colour is the one exception, and it only ever reaches a terminal
that asked for it.
"""

from __future__ import annotations

import os
import sys
import time

#: Set to 0/false to strip colour, 1/true to force it on. Unset means "decide".
COLOR_ENV = "SIMA_VISION_COLOR"

#: Set to anything non-empty to drop everything below WARN, as `--quiet` does.
QUIET_ENV = "SIMA_VISION_QUIET"

#: Width the step labels are padded to. Long enough for "environment".
LABEL_WIDTH = 12


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _enable_windows_ansi(stream) -> bool:
    """Turn on VT processing for a Windows console, reporting whether it took.

    Windows 10 and 11 render ANSI perfectly well, but only once the console mode
    says so. Without this the escapes are printed literally, which is worse than
    no colour at all.
    """
    try:
        import ctypes
        from ctypes import wintypes

        handle = ctypes.windll.kernel32.GetStdHandle(-12 if stream is sys.stderr else -11)
        mode = wintypes.DWORD()
        if not ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        # 0x0004 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:  # pragma: no cover - any failure here just means no colour
        return False


def want_color(stream) -> bool:
    """Whether to emit escapes at all.

    ``$NO_COLOR`` wins over everything, as its convention requires.
    ``$SIMA_VISION_COLOR`` then forces the answer either way, which is how a CI
    log gets colour on purpose and a pipe gets it never.
    """
    override = os.environ.get(COLOR_ENV)
    if override is not None and override.strip():
        return _truthy(override)
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if os.name == "nt":
        return _enable_windows_ansi(stream)
    return True


class Style:
    """The handful of colours this uses, or empty strings when it uses none."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.reset = "\033[0m" if enabled else ""
        self.dim = "\033[2m" if enabled else ""
        self.bold = "\033[1m" if enabled else ""
        self.green = "\033[32m" if enabled else ""
        self.yellow = "\033[33m" if enabled else ""
        self.red = "\033[31m" if enabled else ""
        self.cyan = "\033[36m" if enabled else ""

    def paint(self, text: str, colour: str) -> str:
        return f"{colour}{text}{self.reset}" if self.enabled and colour else text


def human_bytes(size: float) -> str:
    """`118.4 MB`. Decimal units, because that is what download pages quote."""
    for unit in ("B", "KB", "MB"):
        if size < 1000:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000.0
    return f"{size:.1f} GB"


def human_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


class Step:
    """One numbered line, plus anything it wants to say underneath.

    Entered as a context manager so the label appears *before* the slow part
    runs and the outcome lands after it. A step that raises is marked failed on
    the way out, so an exception never leaves an unfinished line as the last
    thing on screen.
    """

    def __init__(self, console: Console, index: int, total: int | None, label: str) -> None:
        self.console = console
        self.index = index
        self.total = total
        self.label = label
        self.started = time.perf_counter()
        self._closed = False

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def _prefix(self) -> str:
        counter = f"[{self.index}/{self.total}]" if self.total else f"[{self.index}]"
        style = self.console.style
        return f"  {style.paint(counter, style.dim)} {self.label:<{LABEL_WIDTH}}"

    def begin(self, detail: str = "") -> Step:
        """Print the label now, so a slow step is never a silent one."""
        self.console.write(f"{self._prefix()} {detail}".rstrip())
        return self

    def detail(self, text: str) -> None:
        """A continuation line, indented under the step."""
        for line in str(text).splitlines() or [""]:
            self.console.write(f"        {line}")

    def note(self, text: str) -> None:
        style = self.console.style
        for line in str(text).splitlines() or [""]:
            self.console.write(f"        {style.paint(line, style.dim)}")

    def done(self, summary: str = "", timed: bool = False) -> None:
        """Say what actually happened, on a line of its own.

        A new line rather than a rewrite of the first one: by the time a step
        finishes there may be a subprocess's output printed under it, and moving
        the cursor back over that would eat it.
        """
        if self._closed:
            return
        self._closed = True
        if self.console.active_step is self:
            self.console.active_step = None
        if not summary:
            return
        style = self.console.style
        when = f" {style.paint('(' + human_time(self.elapsed) + ')', style.dim)}" if timed else ""
        self.console.write(f"        {style.paint('->', style.dim)} {summary}{when}")

    def __enter__(self) -> Step:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.console.active_step is self:
            self.console.active_step = None
        if exc_type is not None and not self._closed:
            self._closed = True
            style = self.console.style
            # Not forced: it closes off the step line visually, and under
            # --quiet there is no step line for it to close. The error itself
            # is printed either way, and that is the part that matters.
            self.console.write(f"        {style.paint('failed', style.red)}")
        return False


class Console:
    """The single place anything user-facing is printed.

    Held as a module-level singleton (:data:`console`) rather than threaded
    through every call: the alternative is one more argument on thirty
    functions that all want the same object.
    """

    #: Indent for a line printed from inside a step, matching Step.detail.
    STEP_BODY = " " * 8

    def __init__(self, stream=None, quiet: bool | None = None) -> None:
        self._stream = stream
        self.quiet = bool(os.environ.get(QUIET_ENV)) if quiet is None else quiet
        self._styled = (None, Style(False))
        self._total: int | None = None
        self._index = 0
        #: The step currently open, so a note from deep in the call stack lands
        #: under it instead of at the left margin. Set by `step`, cleared when
        #: that step is closed.
        self.active_step: Step | None = None

    @property
    def stream(self):
        """Whatever stdout is *now*.

        Resolved per access rather than captured in ``__init__``, because this
        object is a module-level singleton built at import: anything that
        replaces ``sys.stdout`` afterwards -- a shell redirect, a pytest
        capture, ``contextlib.redirect_stdout`` -- would otherwise be writing to
        a stream nothing reads.
        """
        return self._stream if self._stream is not None else sys.stdout

    @property
    def style(self) -> Style:
        """Colours for the current stream, decided once per stream.

        ``want_color`` asks the OS about console modes, so it is worth not
        repeating per line; keying the cache on the stream means a redirect
        still gets its own honest answer.
        """
        stream = self.stream
        cached_for, style = self._styled
        if cached_for is not stream:
            style = Style(want_color(stream))
            self._styled = (stream, style)
        return style

    # -- configuration ------------------------------------------------------

    def configure(self, quiet: bool | None = None, stream=None) -> None:
        """Settle the output policy once argv has been parsed."""
        if stream is not None:
            self._stream = stream
            self._styled = (None, Style(False))
        if quiet is not None:
            self.quiet = quiet

    def plan(self, total: int | None) -> None:
        """How many steps are coming, so they can be numbered `[2/6]`.

        None is the honest answer when the count depends on what is already
        installed. The counter then reads `[2]` and nobody is promised a finish
        line that may not arrive.
        """
        self._total = total
        self._index = 0

    # -- primitives ---------------------------------------------------------

    def write(self, text: str = "", force: bool = False) -> None:
        if self.quiet and not force:
            return
        print(text, file=self.stream, flush=True)

    def banner(self, title: str, subtitle: str = "") -> None:
        style = self.style
        line = style.paint(title, style.bold)
        if subtitle:
            line += f"  {style.paint(subtitle, style.cyan)}"
        self.write()
        self.write(line)
        self.write()

    def step(self, label: str, detail: str = "") -> Step:
        self._index += 1
        self.active_step = Step(self, self._index, self._total, label).begin(detail)
        return self.active_step

    def info(self, text: str) -> None:
        for line in str(text).splitlines() or [""]:
            self.write(f"  {line}")

    def note(self, text: str) -> None:
        """An aside, indented under the open step if there is one.

        Callers four levels down from a step -- building the source graph, say --
        have no step to hand and should not be given one just to say a sentence.
        """
        indent = self.STEP_BODY if self.active_step is not None else "  "
        for line in str(text).splitlines() or [""]:
            self.write(f"{indent}{self.style.paint(line, self.style.dim)}")

    def success(self, text: str) -> None:
        self.write(f"  {self.style.paint('ok', self.style.green)}  {text}")

    def report(self, text: str) -> None:
        """A result, not progress. Survives ``--quiet``.

        The line between this and :meth:`info` is what someone asked for versus
        how it was arrived at. ``--quiet`` exists to keep the first and drop the
        second, so a run that says nothing at all would be a broken one.
        """
        for line in str(text).splitlines() or [""]:
            self.write(f"  {line}", force=True)

    def warn(self, text: str) -> None:
        """Warnings survive --quiet: something to act on is not noise."""
        for n, line in enumerate(str(text).splitlines() or [""]):
            head = self.style.paint("warn", self.style.yellow) if n == 0 else "    "
            self.write(f"  {head}  {line}", force=True)

    def error(self, text: str) -> None:
        """Errors go to stderr, always, whatever --quiet says."""
        style = Style(want_color(sys.stderr))
        for n, line in enumerate(str(text).splitlines() or [""]):
            head = style.paint("ERROR", style.red) if n == 0 else "     "
            print(f"  {head}  {line}", file=sys.stderr, flush=True)

    # -- progress -----------------------------------------------------------

    def progress(self, name: str, done: int, total: int) -> None:
        """A one-line download meter, redrawn in place on a terminal only.

        Piped to a file the carriage return does nothing useful and the same
        line lands three hundred times, so a non-terminal gets silence here and
        the single completion line from the caller instead.
        """
        if self.quiet or not self.style.enabled:
            return
        if total:
            filled = int(24 * done / total)
            bar = "=" * filled + "." * (24 - filled)
            text = f"{bar}  {human_bytes(done)} / {human_bytes(total)}  {name}"
        else:
            text = f"{human_bytes(done)}  {name}"
        print(f"\r      {text}", end="", file=self.stream, flush=True)

    def progress_done(self) -> None:
        if not self.quiet and self.style.enabled:
            print("\r" + " " * 78 + "\r", end="", file=self.stream, flush=True)


#: The one console. Commands call :meth:`Console.configure` on it at startup.
console = Console()
