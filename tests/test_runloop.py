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

from sima_vision.runloop import Stopper, TaskRuntime, run_pipeline
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


def test_two_timeouts_in_a_row_end_a_file_run(capsys):
    cfg, pipeline = make(frames=3, source_frames=3)
    processed = run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    assert processed == 3
    out = capsys.readouterr().out
    assert "the whole clip" in out, "a complete run must not read as a stall"


def test_a_short_run_against_a_known_clip_length_is_called_out(capsys):
    cfg, pipeline = make(frames=4, source_frames=100)
    run_pipeline(pipeline, cfg, Stopper(), CountingRuntime())
    combined = capsys.readouterr()
    assert "stalled" in combined.out
    assert "incomplete" in combined.err


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
    assert "[profile]" in capsys.readouterr().out


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
