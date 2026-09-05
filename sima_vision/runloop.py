"""The pull loop, shared by all three tasks.

Everything expensive happens on the :class:`~sima_vision.sinks.SinkWorker`, so
the only work between two ``pull`` calls is whatever the task's ``decode`` hook
does: copying the frame out, parsing boxes and -- for segmentation -- rebuilding
masks. That is what keeps the decoder's pool turning over.

A task plugs into this by implementing :class:`TaskRuntime`.
"""

from __future__ import annotations

import signal

from .console import console
from .runtime import time_ms
from .samples import FrameStamp
from .sinks import Pipeline, SinkJob, SinkWorker

HEARTBEAT_EVERY = 50


class Stopper:
    """Cooperative stop flag driven by SIGINT, SIGTERM and SIGHUP.

    ``dk`` and ``devkit-run`` invoke over SSH without a pty, so a terminal
    Ctrl-C is never forwarded and the app can be orphaned on the DevKit while
    still holding the MLA. The next run then fails with a busy device. Launch
    interactive runs with ``ssh -tt`` and let these handlers close the Run.

    Attributes:
        stop: Set to True once a signal has been received.
    """

    #: Looked up by name, not by attribute. ``signal.SIGHUP`` does not exist on
    #: Windows, and naming it directly raised AttributeError while building the
    #: tuple -- before the try block that was meant to tolerate exactly that.
    SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP")

    def __init__(self) -> None:
        self.stop = False
        for name in self.SIGNALS:
            number = getattr(signal, name, None)
            if number is None:
                continue
            try:
                signal.signal(number, self._handle)
            except (ValueError, OSError):
                # ValueError: not the main thread, which is fine -- the run
                # loop still checks `stop`, it just cannot be set by a signal.
                pass

    def _handle(self, signum, _frame) -> None:
        if not self.stop:
            console.report(f"[signal {signum}] stopping, closing Run...")
        self.stop = True


class ProfileWindow:
    """Rolling per-stage timing accumulator.

    Sums pull, decode, an optional task stage and sink latencies over a fixed
    number of frames, then prints one averaged line and resets. Averaging avoids
    the noise of per-frame timings without needing to keep every sample.

    Attributes:
        enabled: Whether profiling output is on at all.
        interval: Frames per window before a flush.
        stage: Name of the task-specific stage, such as ``masks``. Empty when
            the task has no third stage, and then it is left out of the line.
        unit: What the counter counts, such as ``detections``.
    """

    def __init__(self, enabled: bool, interval: int, stage: str = "",
                 unit: str = "objects") -> None:
        self.enabled = enabled
        self.interval = interval
        self.stage = stage
        self.unit = unit
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.objects = 0
        self.start_ms = 0.0
        self.pull_ms = 0.0
        self.decode_ms = 0.0
        self.stage_ms = 0.0
        self.sink_ms = 0.0

    def add(self, pull_ms: float, decode_ms: float, stage_ms: float, sink_ms: float,
            count: int) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.objects += count
        self.pull_ms += pull_ms
        self.decode_ms += decode_ms
        self.stage_ms += stage_ms
        self.sink_ms += sink_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = time_ms() - self.start_ms
        fps = self.frames * 1000.0 / elapsed if elapsed > 0 else 0.0
        stage = (
            f"{self.stage}={self.stage_ms / self.frames:.1f}ms " if self.stage else ""
        )
        console.write(
            f"  profile: frames={self.frames} fps={fps:.1f} "
            f"pull={self.pull_ms / self.frames:.1f}ms "
            f"decode={self.decode_ms / self.frames:.1f}ms "
            f"{stage}"
            f"sinks={self.sink_ms / self.frames:.1f}ms "
            f"{self.unit}={self.objects / self.frames:.1f}"
        )
        self.reset()


class TaskRuntime:
    """What the pull loop needs from a task.

    Attributes:
        output_label: Public output the loop pulls, such as ``detector_output``.
        stream: Insight stream name, such as ``object-detection``.
        unit: Plural noun for the heartbeat and the profile line.
        stage: Name of the task's own profiling stage, or "" for none.
    """

    output_label = "detector_output"
    stream = "objects"
    unit = "detections"
    stage = ""

    def decode(self, pipeline: Pipeline, cfg, sample, index: int):
        """Turn one pulled sample into a frame and this task's results.

        Owns the sample's whole lifetime. It must drop every reference to it
        before returning, because the buffer it holds belongs to the hardware
        decoder's small pool -- see
        :class:`~sima_vision.samples.FrameStamp` for what happens otherwise.

        Args:
            pipeline: Live pipeline.
            cfg: Application configuration.
            sample: The joined sample from ``pull``.
            index: 1-based frame number.

        Returns:
            A ``(frame, results, stage_ms)`` triple.
        """
        raise NotImplementedError

    def render(self, cfg, pipeline: Pipeline, frame, results, fps: float):
        """Draw one frame's overlay. Runs on the sink thread."""
        raise NotImplementedError

    def metadata(self, pipeline: Pipeline, results) -> list[dict]:
        """The Insight JSON payload for one frame's results."""
        raise NotImplementedError

    def summarise(self, pipeline: Pipeline, processed: int) -> list[str]:
        """Extra lines printed once the run is over."""
        return []


def pull_frame(pipeline: Pipeline, cfg, sinks: SinkWorker, label: str, processed: int):
    """Pull one joined sample, flushing our own backlog before giving up.

    A starved decoder and a finished clip look identical from here: both are
    silence. So on the first timeout, hand back everything the app is still
    holding -- the sink queue is several decoded frames deep -- and ask again. A
    pool that refills answers straight away; a clip that ended stays quiet.

    Args:
        pipeline: Live pipeline.
        cfg: Application configuration, for ``pull_timeout_ms``.
        sinks: Sink worker to drain before the retry.
        label: Public output to pull.
        processed: Frames processed so far, for the message only.

    Returns:
        A ``(sample, timed_out, recovered)`` triple. ``sample`` is None only
        when both attempts came back empty.
    """
    sample = pipeline.run.pull(label, cfg.pull_timeout_ms)
    if sample is not None:
        return sample, False, False

    console.warn(
        f"timed out waiting for results after {processed} frames; "
        "flushing the sink queue and retrying once"
    )
    sinks.drain()
    sample = pipeline.run.pull(label, cfg.pull_timeout_ms)
    if sample is None:
        return None, True, False

    console.warn(
        "the source recovered once the backlog was flushed. That was "
        "back-pressure from this app rather than the end of the clip; lower "
        "runtime.sink_queue_depth if it keeps happening."
    )
    return sample, True, True


def source_stopped_message(cfg, pipeline: Pipeline, processed: int,
                           sink_ms: float = 0.0) -> str:
    """Explain a source that went quiet, ruling out what the frame count rules out.

    The clip's length is known before the run starts, so "it just ended" is
    either the whole answer or not on the list at all. Saying which turns a
    short recording from something to be interpreted into something decided.
    """
    total = pipeline.source_frames
    head = (
        f"source produced nothing for {cfg.pull_timeout_ms} ms twice in a row after "
        f"{processed} frames"
    )
    if total and processed >= total:
        return (
            f"{head}, which is the whole clip ({total} frames). Nothing is wrong: "
            "the run is complete."
        )

    if total:
        head += (
            f", {processed / total:.0%} of the way through a {total} frame clip. "
            "The source stalled; it did not end."
        )
    else:
        head += ". If that is far short of the clip, the source stalled rather than ended."

    causes = stall_causes(cfg, pipeline, sink_ms)
    listed = "\n".join(
        f"  {n}. {cause}" for n, cause in enumerate(causes, 1)
    )
    return (
        f"{head}\nIn order of likelihood:\n{listed}\n"
        "Run again with --no-save --no-video to tell the graph apart from how much\n"
        "work this app does per frame: the same stall means the graph, a complete\n"
        "run means the sinks."
    )


def stall_causes(cfg, pipeline: Pipeline, sink_ms: float) -> list[str]:
    """Why the source stopped, most likely first.

    Ordered by what this run actually measured rather than by what is usually
    true. The sink cost is known -- ``SinkWorker`` times every ``submit`` that
    had to wait -- and when it dwarfs the frame interval it is not a hypothesis,
    it is the answer, so it goes first and the generic advice goes below it.

    Causes that cannot apply are left out entirely. Telling someone their
    Insight feed may have wedged the codec daemon when they never turned Insight
    on is one more thing to rule out by hand.
    """
    interval = 1000.0 / (pipeline.fps or 25)
    causes: list[str] = []

    if sink_ms > interval:
        causes.append(
            f"the sinks cannot keep up. They held the pull loop {sink_ms:.0f} ms per\n"
            f"     frame against a {interval:.0f} ms frame interval, so the loop was not\n"
            "     asking for frames and decoded ones piled up between the decoder and\n"
            "     this app. Drawing and encoding 1080p in software on the board's CPU\n"
            "     is the usual reason. --no-video is the cheapest thing to try, and\n"
            "     --sink-queue-depth buys the loop room to keep draining the decoder.\n"
            "     Not --queue-depth: that one deepens the runtime's own queues, which\n"
            "     parks more decoded frames and makes this worse."
        )

    # Only worth suggesting when there is somewhere to lower it to. The floor is
    # 1 and so is the default, so on a config nobody has touched this used to
    # read as "turn down the thing that is already all the way down".
    room = (
        f"     Then lower runtime.output_buffers, currently {cfg.output_buffers}, "
        f"which costs\n     two buffers for every one you take off it."
        if cfg.output_buffers > 1
        else "     runtime.output_buffers is already 1, its minimum, so the slack has\n"
             "     to come from somewhere else."
    )
    causes.append(
        "the hardware decoder ran out of buffers. Its pool is small (the boot log\n"
        "     prints BufferNum), and every element between it and the source appsink\n"
        "     can park one. Count the queues in the first pipeline printed above:\n"
        "     their max-buffers plus the appsink's must stay under BufferNum.\n"
        f"{room}"
    )

    if cfg.insight_enable:
        causes.append(
            "output.insight.enable is on and its encoder shares the codec daemon\n"
            "     with the decoder, so it can wedge it. Try again without --insight."
        )
    return causes


def consume_frames(pipeline: Pipeline, cfg, stopper: Stopper, sinks: SinkWorker,
                   profile: ProfileWindow, task: TaskRuntime) -> tuple[int, int, int]:
    """The pull loop. Returns ``(processed, timeouts, recovered)``."""
    processed = 0
    timeouts = 0
    recovered = 0
    heartbeat_start = time_ms()
    heartbeat_count = 0
    live_fps = float(pipeline.fps or 25)   # HUD value, refreshed each heartbeat

    while not stopper.stop and (cfg.frames <= 0 or processed < cfg.frames):
        pull_start = time_ms()
        sample, timed_out, came_back = pull_frame(
            pipeline, cfg, sinks, task.output_label, processed
        )
        pull_end = time_ms()
        timeouts += int(timed_out)
        recovered += int(came_back)

        if sample is None:
            if cfg.source_type == "video":
                console.report(
                    source_stopped_message(
                        cfg, pipeline, processed,
                        sinks.blocked_ms / processed if processed else 0.0,
                    )
                )
                break
            continue

        stamp = FrameStamp.of(sample)
        frame, results, stage_ms = task.decode(pipeline, cfg, sample, processed + 1)
        sample = None
        decode_end = time_ms()

        processed += 1
        sinks.submit(SinkJob(processed, stamp, frame, results, live_fps))
        sink_end = time_ms()

        count = len(results)
        profile.add(
            pull_end - pull_start,
            (decode_end - pull_end) - stage_ms,
            stage_ms,
            sink_end - decode_end,
            count,
        )

        # Heartbeat, so a healthy run does not look identical to a stalled one.
        heartbeat_count += count
        if processed % HEARTBEAT_EVERY == 0:
            elapsed = time_ms() - heartbeat_start
            rate = HEARTBEAT_EVERY * 1000.0 / elapsed if elapsed > 0 else 0.0
            live_fps = rate or live_fps
            console.write(
                f"  {processed:>6}  {rate:.1f} fps, "
                f"{heartbeat_count / HEARTBEAT_EVERY:.1f} {task.unit}/frame avg"
            )
            heartbeat_start = time_ms()
            heartbeat_count = 0

    return processed, timeouts, recovered


def report_recording(cfg, pipeline: Pipeline, timeouts: int) -> None:
    """Say whether the recording is complete, and rank the reasons when it is not.

    An incomplete recording is the most commonly reported symptom, and it has
    several quite different causes. "Incomplete" is measured against the clip's
    own length where that is known, not against an arbitrary few seconds: a 15
    second clip cut off at 3 seconds used to pass this check in silence.
    """
    if pipeline.writer is None or not pipeline.writer_frames:
        return

    total = pipeline.source_frames
    out_fps = cfg.video_fps or pipeline.fps or 25
    seconds = pipeline.writer_frames / out_fps if out_fps else 0.0
    short = pipeline.writer_frames < total if total else seconds < 2.0

    if not short:
        if total and pipeline.writer_frames >= total:
            console.report(f"video: complete, all {total} frames of the clip.")
        return

    causes = []
    if cfg.frames:
        causes.append(f"runtime.frames is {cfg.frames}, which capped the run.")
    if cfg.insight_enable:
        causes.append(
            "output.insight.enable is true. Its H.264 encoder shares the codec "
            "daemon with the decoder feeding the source, so a failing encoder "
            "stalls the run. Set it to false; the recording does not need it."
        )
    if timeouts:
        causes.append(
            f"the source stopped producing frames ({timeouts} timeout(s)), so "
            "the run ended before the clip did."
        )
    if not causes:
        causes.append(
            "frames were dropped rather than blocked on. Check that "
            "runtime.overflow_policy resolved to block, as printed at startup."
        )
    listed = "\n".join(f"       {i}. {c}" for i, c in enumerate(causes, 1))
    missing = (
        f"{pipeline.writer_frames} of {total} frames, {seconds:.1f}s of "
        f"{total / out_fps:.1f}s" if total
        else f"only {seconds:.1f}s ({pipeline.writer_frames} frames at {out_fps} fps)"
    )
    console.warn(f"the recording is incomplete: {missing}.\n{listed}")


def run_pipeline(pipeline: Pipeline, cfg, stopper: Stopper, task: TaskRuntime) -> int:
    """Run one task to completion and print the closing report."""
    profile = ProfileWindow(cfg.profile, cfg.profile_interval, task.stage, task.unit)
    sinks = SinkWorker(
        cfg, pipeline, cfg.sink_queue_depth, task.render, task.stream, task.metadata
    )
    try:
        processed, timeouts, recovered = consume_frames(
            pipeline, cfg, stopper, sinks, profile, task
        )
    finally:
        # Ordered before anything that reads writer_frames: frames may still be
        # queued, and they belong in the recording. close() re-raises whatever
        # the worker hit, so a failing sink is not swallowed.
        sinks.close()

    profile.flush()
    total = pipeline.source_frames
    summary = " ".join(task.summarise(pipeline, processed))
    console.write()
    console.report(
        f"processed={processed}{f' of {total}' if total else ''} timeouts={timeouts} "
        f"recovered={recovered}{f' {summary}' if summary else ''}"
    )
    if sinks.blocked_ms > 1000.0 and processed:
        console.report(
            f"sinks: the pull loop waited {sinks.blocked_ms / 1000.0:.1f}s in total for "
            f"the sink thread ({sinks.blocked_ms / processed:.0f} ms/frame). "
            "Cheaper settings are in the README under \"It runs slower than the detector\"."
        )

    report_recording(cfg, pipeline, timeouts)

    if pipeline.metadata_sender is not None:
        stats = pipeline.metadata_sender.stats()
        console.report(
            f"metadata: sent={stats.datagrams_sent} failures={stats.send_failures} "
            f"would_block={stats.would_block}"
        )
    if pipeline.video_dropped:
        console.report(
            f"insight: dropped {pipeline.video_dropped} preview frames because the "
            f"feed was busy. The recording is unaffected."
        )
    return processed
