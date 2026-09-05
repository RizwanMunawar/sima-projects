"""The pull loop and the sink thread.

This is the part that only ran on the board, and the part most likely to be
wrong: it is threaded, it owns buffer lifetime, and its failure mode is a
deadlock rather than an exception. Faking ``pipeline.run`` is enough to drive
all of it off the board -- ``pull`` returning objects is the entire contract
the loop depends on.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from sima_vision.runloop import (
    Stopper,
    TaskRuntime,
    run_pipeline,
    source_stopped_message,
    stall_causes,
)
from sima_vision.sinks import Pipeline
from sima_vision.tasks import TASKS


class FakeSample:
    """What ``pull`` hands back. Only the timing fields are ever read."""

    def __init__(self, index: int) -> None:
        self.pts_ns = index * 40_000_000       # 25 fps
        self.dts_ns = self.pts_ns
        self.duration_ns = 40_000_000
        self.frame_id = index
        self.stream_id = 0


class FakeRun:
    """A ``Run`` that yields ``frames`` samples, then times out forever.

    ``timeouts_before`` inserts a run of empty pulls at a given frame, which is
    how a starved decoder looks from here.
    """

    def __init__(self, frames: int, timeouts_at: dict[int, int] | None = None) -> None:
        self.frames = frames
        self.timeouts_at = dict(timeouts_at or {})
        self.pulled = 0
        self.closed = False
        self.labels: list[str] = []

    def pull(self, label: str, timeout_ms: int):
        self.labels.append(label)
        pending = self.timeouts_at.get(self.pulled, 0)
        if pending:
            self.timeouts_at[self.pulled] = pending - 1
            return None
        if self.pulled >= self.frames:
            return None
        self.pulled += 1
        return FakeSample(self.pulled)

    def close(self) -> None:
        self.closed = True


class FakeWriter:
    """An OpenCV VideoWriter stand-in that records what it was given."""

    def __init__(self) -> None:
        self.frames: list = []
        self.released = False

    def write(self, frame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True


class CountingRuntime(TaskRuntime):
    """A task that records the order it saw frames in, on both threads."""

    output_label = "detector_output"
    stream = "test"
    unit = "things"

    def __init__(self, per_frame: int = 2, fail_render_on: int | None = None) -> None:
        self.per_frame = per_frame
        self.fail_render_on = fail_render_on
        self.decoded: list[int] = []
        self.rendered: list[int] = []
        self.decode_thread: set[str] = set()
        self.render_thread: set[str] = set()

    def decode(self, pipeline, cfg, sample, index: int):
        self.decoded.append(index)
        self.decode_thread.add(threading.current_thread().name)
        frame = np.full((16, 24, 3), index % 256, np.uint8)
        return frame, [{"i": index}] * self.per_frame, 0.0

    def render(self, cfg, pipeline, frame, results, fps: float):
        index = int(frame[0, 0, 0])
        self.rendered.append(index)
        self.render_thread.add(threading.current_thread().name)
        if self.fail_render_on is not None and index == self.fail_render_on:
            raise RuntimeError("sink exploded")
        return frame.copy()

    def metadata(self, pipeline, results) -> list[dict]:
        return [{"id": str(i)} for i, _ in enumerate(results)]


def make(frames: int = 5, source_frames: int = 0, writer: bool = True, **settings):
    """A config and a Pipeline wired to a FakeRun."""
    cfg = TASKS["detect"]().load(
        None,
        {"model.path": "m", "source.uri": "c.h264", "output.save.enable": False,
         **settings},
        use_file=False,
    )
    pipeline = Pipeline(labels=["thing"], frame_w=24, frame_h=16, fps=25)
    pipeline.run = FakeRun(frames)
    pipeline.source_frames = source_frames
    if writer:
        pipeline.writer = FakeWriter()
        pipeline.writer_path = "out.mp4"
    return cfg, pipeline


# ── the loop ──


def test_every_frame_is_processed_and_written():
    cfg, pipeline = make(frames=7)
    task = CountingRuntime()
    processed = run_pipeline(pipeline, cfg, Stopper(), task)
    assert processed == 7
    assert task.decoded == list(range(1, 8))
    assert pipeline.writer_frames == 7


def test_the_sinks_keep_source_order():
    """One worker draining a FIFO is what guarantees the recording is in order."""
    cfg, pipeline = make(frames=25)
    task = CountingRuntime()
    run_pipeline(pipeline, cfg, Stopper(), task)
    assert task.rendered == list(range(1, 26))


def test_rendering_happens_off_the_pull_thread():
    """If drawing ran on the pull loop it would hold decoder buffers."""
    cfg, pipeline = make(frames=6)
    task = CountingRuntime()
    run_pipeline(pipeline, cfg, Stopper(), task)
    assert task.decode_thread and task.render_thread
    assert task.decode_thread.isdisjoint(task.render_thread)


def test_frames_caps_the_run():
    cfg, pipeline = make(frames=50, **{"runtime.frames": 4})
    processed = run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    assert processed == 4


def test_the_label_pulled_is_the_tasks_own():
    cfg, pipeline = make(frames=2)
    task = CountingRuntime()
    task.output_label = "segmenter_output"
    run_pipeline(pipeline, cfg, Stopper(), task)
    assert set(pipeline.run.labels) == {"segmenter_output"}


def test_a_stopper_ends_the_run():
    cfg, pipeline = make(frames=1000)
    stopper = Stopper()
    stopper.stop = True
    assert run_pipeline(pipeline, cfg, stopper, CountingRuntime()) == 0


# ── timeouts ──


def test_a_single_timeout_is_retried_not_fatal():
    """A starved pool answers on the retry; a finished clip does not."""
    cfg, pipeline = make(frames=6)
    pipeline.run.timeouts_at = {3: 1}          # one empty pull after 3 frames
    processed = run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    assert processed == 6, "the retry should have recovered the run"


def test_a_finished_clip_is_not_retried_or_warned_about(capsys):
    """A healthy run used to end with a warning and a `pull_timeout_ms` wait.

    Every frame of the clip had arrived, and the loop still drained, waited the
    full timeout again and printed "timed out waiting for results" before
    stopping. Twenty idle seconds and a scare, on the successful path.
    """
    cfg, pipeline = make(frames=3, source_frames=3)
    task = CountingRuntime()
    processed = run_pipeline(pipeline, cfg, Stopper(), task)

    assert processed == 3
    out = capsys.readouterr().out
    assert "the run is complete" in out, "a complete run must not read as a stall"
    assert "timed out" not in out
    assert "timeouts=0" in out, "the end of a file is not a timeout"
    # Four pulls: three frames and the one empty pull that ends the clip. A
    # fifth would be the retry this test exists to prevent.
    assert pipeline.run.pulled == 3
    assert len(pipeline.run.labels) == 4


def test_a_clip_with_frames_left_is_fought_for_not_abandoned(capsys):
    """The bug behind a 28-of-379 frame recording.

    One retry, then the run ended -- on a clip whose length the app had already
    counted and printed. Draining the sink queue is what releases the stall, so
    the frames the app knows are still coming are worth more than one attempt.
    """
    from sima_vision.runloop import STALL_RETRIES

    cfg, pipeline = make(frames=4, source_frames=100)
    # Silent for two whole attempts, then the backlog clears and it comes back.
    pipeline.run.timeouts_at = {4: STALL_RETRIES - 1}
    pipeline.run.frames = 9

    processed = run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())

    assert processed == 9, "the run gave up on a clip that still had frames"
    assert pipeline.writer_frames == 9
    out = capsys.readouterr().out
    assert "recovered once the backlog was flushed" in out


def test_the_recovery_advice_points_the_knob_the_right_way(capsys):
    """It said *lower* sink_queue_depth, which causes the stall it follows.

    A shallower sink queue means `submit` blocks sooner, and a blocked pull loop
    is exactly what lets decoded frames pile up against the decoder's pool. The
    stall advice in `stall_causes` and the `--sink-queue-depth` help both say
    raise it; this message was the odd one out, and it is the one a stalling run
    actually prints.
    """
    cfg, pipeline = make(frames=2, source_frames=100)
    pipeline.run.timeouts_at = {1: 1}
    run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())

    out = capsys.readouterr().out
    assert "raise runtime.sink_queue_depth" in out
    assert "lower runtime.sink_queue_depth" not in out


def test_how_hard_a_silent_source_is_fought_depends_on_the_clip():
    """Three different questions wearing the same silence."""
    from sima_vision.runloop import STALL_RETRIES, stall_attempts

    # Every frame arrived: this is the end of the file, not a stall.
    assert stall_attempts(stalled_pipeline(total=379), 379) == 0
    assert stall_attempts(stalled_pipeline(total=379), 400) == 0
    # Frames demonstrably left: the source stalled, so fight for them.
    assert stall_attempts(stalled_pipeline(total=379), 28) == STALL_RETRIES
    # Length unknown: one retry tells a starved pool from a finished clip.
    assert stall_attempts(stalled_pipeline(total=0), 28) == 1


def test_a_short_run_against_a_known_clip_length_is_called_out(capsys):
    cfg, pipeline = make(frames=4, source_frames=100)
    run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    # Warnings share stdout with the steps, deliberately: a warning about the
    # recording only makes sense read in place, next to the run it belongs to.
    # Only errors go to stderr.
    combined = capsys.readouterr()
    assert "stalled" in combined.out
    assert "incomplete" in combined.out
    assert combined.err == ""


# ── failures ──


def test_a_failing_sink_is_raised_not_swallowed():
    cfg, pipeline = make(frames=10)
    task = CountingRuntime(fail_render_on=3)
    with pytest.raises(RuntimeError, match="sink exploded"):
        run_pipeline(pipeline, cfg, Stopper(), task)


def test_a_failing_sink_does_not_deadlock_the_pull_loop():
    """The worker must keep draining after an error, or submit() blocks forever."""
    cfg, pipeline = make(frames=200, **{"runtime.queue_depth": 1})
    task = CountingRuntime(fail_render_on=2)
    done = threading.Event()

    def go():
        try:
            run_pipeline(pipeline, cfg, Stopper(), task)
        except RuntimeError:
            pass
        finally:
            done.set()

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    assert done.wait(timeout=30), "the run loop deadlocked after a sink failure"


def test_a_failing_decode_still_closes_the_sinks():
    cfg, pipeline = make(frames=10)

    class Exploding(CountingRuntime):
        def decode(self, pipeline, cfg, sample, index):
            if index == 3:
                raise RuntimeError("decode exploded")
            return super().decode(pipeline, cfg, sample, index)

    task = Exploding()
    with pytest.raises(RuntimeError, match="decode exploded"):
        run_pipeline(pipeline, cfg, Stopper(), task)
    # Two frames made it through and were written before the failure.
    assert pipeline.writer_frames == 2


# ── reporting ──


def test_the_summary_reports_what_the_task_adds(capsys):
    class Summarising(CountingRuntime):
        def summarise(self, pipeline, processed):
            return ["masks=packed"]

    cfg, pipeline = make(frames=3)
    run_pipeline(pipeline, cfg, Stopper(), Summarising())
    assert "masks=packed" in capsys.readouterr().out


def test_profiling_prints_a_line(capsys):
    cfg, pipeline = make(
        frames=4, **{"runtime.profile": True, "runtime.profile_interval": 2}
    )
    run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    assert "profile: frames=" in capsys.readouterr().out


def test_the_heartbeat_counts_what_the_task_returns(capsys):
    from sima_vision import runloop

    cfg, pipeline = make(frames=4)
    monkey = runloop.HEARTBEAT_EVERY
    runloop.HEARTBEAT_EVERY = 2
    try:
        run_pipeline(pipeline, cfg, Stopper(), CountingRuntime(per_frame=3))
    finally:
        runloop.HEARTBEAT_EVERY = monkey
    out = capsys.readouterr().out
    assert "3.0 things/frame" in out


# -- what the app says when the source stalls --


def stall_config(**settings):
    base = {"model.path": "m.tar.gz", "source.uri": "c.h264"}
    return TASKS["detect"]().load(None, {**base, **settings}, use_file=False)


def stalled_pipeline(fps: int = 24, total: int = 379) -> Pipeline:
    pipeline = Pipeline(labels=["person"])
    pipeline.fps = fps
    pipeline.source_frames = total
    return pipeline


def test_the_advice_only_names_flags_that_exist():
    """It told a `detect` user to run `--minimal`, which only `segment` has.

    Advice that does not parse is worse than none: it costs a run to find out.
    Every flag this message mentions has to be accepted by every task, since
    one message is shared by all of them.
    """
    import re

    from sima_vision.cli import build_parser

    message = source_stopped_message(stall_config(), stalled_pipeline(), 23, sink_ms=183.0)
    mentioned = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", message))
    assert mentioned, "the message should suggest something"

    # Compared against the option strings rather than parsed: some of these take
    # a value, and this is asking whether the flag exists, not how to call it.
    subparsers = [
        action.choices
        for action in build_parser()._actions
        if isinstance(getattr(action, "choices", None), dict)
    ][0]
    for task in TASKS:
        accepted = {
            option
            for action in subparsers[task]._actions
            for option in action.option_strings
        }
        unknown = mentioned - accepted
        assert not unknown, f"the stall advice tells `{task}` to use {sorted(unknown)}"


def test_the_measured_cause_is_ranked_first():
    """The run timed the sinks. A measurement outranks a hypothesis."""
    causes = stall_causes(stall_config(), stalled_pipeline(), sink_ms=183.0)
    assert "sinks cannot keep up" in causes[0]
    assert "183 ms" in causes[0] and "42 ms" in causes[0]


def test_sinks_that_keep_up_are_not_blamed():
    causes = stall_causes(stall_config(), stalled_pipeline(), sink_ms=2.0)
    assert not any("cannot keep up" in cause for cause in causes)
    assert "decoder ran out of buffers" in causes[0]


def test_the_two_queue_depths_are_separate_knobs():
    """One setting drove both, and they pull opposite ways.

    `RunOptions.queue_depth` parks decoded frames from the hardware decoder's
    eight-buffer pool. The sink queue holds numpy copies in host memory and no
    decoder buffer at all, so depth there is what lets the pull loop keep
    draining. Raising the single old knob for slack deepened the runtime queues
    too, making the buffer exhaustion it was meant to relieve slightly worse.
    """
    cfg = stall_config(**{"runtime.queue_depth": 1, "runtime.sink_queue_depth": 8})
    assert cfg.queue_depth == 1
    assert cfg.sink_queue_depth == 8


def test_the_sink_queue_is_the_one_the_worker_gets():
    """A regression here is invisible: the run works, just with less slack."""
    import sima_vision.runloop as loop

    seen = {}
    real = loop.SinkWorker

    class Spy(real):
        def __init__(self, cfg, pipeline, depth, *args):
            seen["depth"] = depth
            super().__init__(cfg, pipeline, depth, *args)

    cfg, pipeline = make(frames=2, **{"runtime.sink_queue_depth": 6})
    loop.SinkWorker = Spy
    try:
        run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    finally:
        loop.SinkWorker = real
    assert seen["depth"] == 6, "the sink worker must get the sink depth"


def test_the_advice_does_not_point_at_the_knob_that_makes_it_worse():
    causes = stall_causes(stall_config(), stalled_pipeline(), sink_ms=183.0)
    assert "--sink-queue-depth" in causes[0]
    assert "Not --queue-depth" in causes[0]


def test_a_setting_already_at_its_floor_is_not_suggested():
    """`output_buffers` bottoms out at 1, which is also the default.

    The advice read "lower runtime.output_buffers" on a config nobody had
    touched, which is telling someone to turn down a dial already at zero.
    """
    at_floor = stall_causes(stall_config(), stalled_pipeline(), sink_ms=0.0)
    decoder = next(c for c in at_floor if "decoder ran out" in c)
    assert "already 1" in decoder
    assert "Then lower" not in decoder

    raised = stall_causes(
        stall_config(**{"runtime.output_buffers": 4}), stalled_pipeline(), sink_ms=0.0
    )
    decoder = next(c for c in raised if "decoder ran out" in c)
    assert "currently 4" in decoder
    assert "already 1" not in decoder


def test_insight_is_only_blamed_when_it_is_on():
    """It was listed unconditionally, so every user had one more thing to rule out."""
    off = stall_causes(stall_config(), stalled_pipeline(), sink_ms=0.0)
    assert not any("insight" in cause for cause in off)

    on = stall_causes(
        stall_config(**{"output.insight.enable": True}), stalled_pipeline(), sink_ms=0.0
    )
    assert any("insight" in cause for cause in on)


def test_a_complete_run_is_not_called_a_stall():
    message = source_stopped_message(stall_config(), stalled_pipeline(total=23), 23)
    assert "the run is complete" in message
    assert "In order of likelihood" not in message
