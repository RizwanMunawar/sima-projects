"""The Task protocol and the startup sequence every task shares."""

from __future__ import annotations

from pathlib import Path

from ..assets import ensure_assets
from ..config import (
    BaseConfig,
    TaskDefaults,
    apply_overrides,
    discover_config,
    load_base_config,
    read_config_file,
    validate_base,
)
from ..console import console, human_bytes
from ..media import (
    check_source_file,
    check_source_support,
    decoder_budget_warning,
    decoder_buffers_for,
    ensure_annex_b,
    is_elementary_h264,
    resolve_source_geometry,
    source_frame_count,
)
from ..neat import (
    build_task_graph,
    describe_preprocess,
    make_model,
    make_run_options,
    resolve_flow_control,
)
from ..runloop import Stopper, TaskRuntime, run_pipeline, sink_depth_for
from ..runtime import FAMILY_DECODE_TOKENS
from ..sinks import Pipeline, load_labels, open_video_writer, start_insight


class Task:
    """One application: detect, segment or fall.

    A task supplies its config type, its CLI flags, its graph labels and a
    :class:`~sima_vision.runloop.TaskRuntime`. Everything else -- probing the
    source, loading the model, building the graph, bringing up Insight, opening
    the writer, the pull loop and the closing report -- is shared.

    Attributes:
        name: Subcommand name, such as ``detect``. Also the key it is
            registered under, and the key into the asset catalogue.
        help: One-line description for ``sima-vision --help``.
        graph_name: Name of the Neat graph, such as ``yolo_detector``.
        result_label: Public output carrying the model results.
        output_label: Public output the run loop pulls.
        defaults: Per-task overrides of the shared config defaults.
    """

    name = ""
    help = ""
    graph_name = "yolo_detector"
    result_label = "detections"
    output_label = "detector_output"
    defaults = TaskDefaults()

    #: The config dataclass this task builds. Tasks that add sections point
    #: this at their own subclass of :class:`~sima_vision.config.BaseConfig`.
    config_class: type = BaseConfig

    #: Set by :meth:`build` as soon as it exists, so a failure part-way through
    #: still has something to close.
    pipeline: Pipeline | None = None

    # ── configuration ──

    def add_arguments(self, parser) -> None:
        """Add task-specific flags. Shared flags are added by the CLI."""

    def load(self, path: Path | None, overrides: dict, use_file: bool = True) -> BaseConfig:
        """Build this task's config from a file plus CLI overrides.

        Args:
            path: ``--config``, or None to discover one.
            overrides: Dotted config paths from the CLI, as built by
                :func:`sima_vision.cli.collect_overrides`.
            use_file: False for ``--no-config``: skip discovery entirely and
                build from the dataclass defaults plus ``overrides``.

        Returns:
            A validated config.
        """
        resolved = discover_config(path) if use_file else None
        raw = apply_overrides(read_config_file(resolved), self.link_overrides(overrides))
        cfg = self.build_config(raw, resolved)
        self.validate(cfg)
        return cfg

    def link_overrides(self, overrides: dict) -> dict:
        """Let one flag imply another before the overrides are applied.

        Override to express things like "naming a recipient means you want
        alerts on". Returns the dict to apply; mutating and returning the
        argument is fine.
        """
        return overrides

    def early_exit(self, cfg, args) -> int | None:
        """Handle a flag that answers a question instead of running.

        Runs after the config is resolved and before pyneat is loaded, so it
        works off the board. Return an exit code to stop there, or None to go
        on and run the pipeline.
        """
        return None

    def post_process(self, cfg, args):
        """Adjust the finished config from flags that are not config keys.

        Override for switches like ``--minimal`` that turn several unrelated
        settings off at once.
        """
        return cfg

    def extra_sections(self, raw: dict) -> dict:
        """This task's own config sections, as ``config_class`` keywords.

        Override to read a section the base config knows nothing about. Return
        nothing and the task simply uses :class:`BaseConfig` as it stands.
        """
        return {}

    def build_config(self, raw: dict, path: Path | None) -> BaseConfig:
        """Read the shared sections, then let the task add its own."""
        base = load_base_config(raw, path, self.defaults)
        extra = self.extra_sections(raw)
        if not extra:
            return base
        return self.config_class(
            **{name: getattr(base, name) for name in BaseConfig.__dataclass_fields__},
            **extra,
        )

    def validate(self, cfg) -> None:
        """Check the config. Override to add rules, and call ``super()`` first."""
        validate_base(cfg)

    def describe(self, cfg) -> list[str]:
        """Lines printed by ``--validate``, beyond the shared ones."""
        return []

    # ── running ──

    def make_pipeline(self, cfg, labels: list[str]) -> Pipeline:
        """Build this task's (possibly subclassed) empty Pipeline."""
        return Pipeline(labels=labels)

    def prepare(self, cfg, pipeline: Pipeline, step) -> None:
        """Hook run after the model is loaded and before the graph is built."""

    def runtime(self, cfg, pipeline: Pipeline) -> TaskRuntime:
        """The pull-loop implementation for this task."""
        raise NotImplementedError

    # -- the three build steps --

    def open_source(self, cfg, step) -> tuple[int, int, int]:
        """Step: prove the source is readable and get its real geometry.

        What is being read is named before it is probed, because probing is
        where the warnings come from and a warning about a file you have not
        been told the name of is half a message.
        """
        size = check_source_file(cfg)
        check_source_support(cfg)
        where = cfg.source_uri or "<default camera>"
        step.detail(f"{where}  ({cfg.source_type}{f', {human_bytes(size)}' if size else ''})")
        width, height, fps = resolve_source_geometry(cfg)
        step.note(describe_preprocess(cfg, width, height))
        # Said here rather than after the stall it predicts. The SPS is already
        # open and the arithmetic is settled before a frame moves, so there is
        # no reason to spend a model load and half a clip finding out.
        if cfg.source_type == "video" and is_elementary_h264(cfg.source_uri):
            # What the decoder is actually asked for, which is what the budget
            # has to be measured against. Left to itself the daemon picks 8 for
            # 1080p no matter what the stream keeps, and that is the stall.
            asked = decoder_buffers_for(cfg, width, height)
            pool = asked or cfg.decoder_pool
            if asked:
                step.detail(f"decoder: asking for {asked} buffers (pyneat picks 8 alone)")
            budget = decoder_budget_warning(cfg.source_uri, width, height, pool)
            if budget:
                console.warn(budget)
        step.done(f"{width}x{height} @ {fps} fps")
        return width, height, fps

    def load_model(self, cfg, width: int, height: int, step) -> Pipeline:
        """Step: unpack the archive onto the MLA and build the empty Pipeline."""
        step.note("the first load unpacks the archive, which can take a minute")
        model = make_model(cfg, width, height)
        labels = load_labels(cfg.labels_path)
        # Published before the graph exists so run()'s finally can close a
        # pipeline that failed part-way through building.
        pipeline = self.make_pipeline(cfg, labels)
        self.pipeline = pipeline
        pipeline.model = model
        step.done(
            f"{cfg.family} -> {FAMILY_DECODE_TOKENS[cfg.family]}, {len(labels)} classes",
            timed=True,
        )
        return pipeline

    def build_pipeline(self, cfg, pipeline: Pipeline, geometry, step) -> None:
        """Step: flow control, the Neat graph, Insight and the output sinks."""
        width, height, fps = geometry
        pipeline.frame_w, pipeline.frame_h, pipeline.fps = width, height, fps
        pipeline.source_frames = source_frame_count(cfg)
        if pipeline.source_frames:
            step.detail(f"{pipeline.source_frames} coded pictures in the clip")

        self.prepare(cfg, pipeline, step)

        preset, policy = resolve_flow_control(cfg)
        # The effective depth, not the configured floor. On a file source the
        # backlog is sized to the clip, and that number is the difference
        # between a complete recording and a stall, so it belongs on screen.
        depth = sink_depth_for(cfg, pipeline)
        held_mb = depth * width * height * 3 / (1 << 20)
        step.detail(
            f"flow: preset={preset} overflow={policy} queue_depth={cfg.queue_depth} "
            f"output_buffers={cfg.output_buffers} sinks={depth} (up to {held_mb:.0f} MB)"
        )
        if policy == "block":
            step.note(
                "block keeps every frame, so the run takes longer than the clip and "
                "the output length matches the input. That is right for a file."
            )
        elif cfg.source_type == "video":
            console.warn(
                "overflow_policy drops frames, so the recording will be shorter\n"
                "than the input and will play fast. Use auto for a file source."
            )

        graph = build_task_graph(
            cfg, pipeline.model, width, height, fps,
            self.graph_name, self.result_label, self.output_label,
        )
        if cfg.profile:
            step.detail(f"backend:\n{graph.describe_backend()}")
        # Kept on the pipeline, not just used here: the Run below outlives this
        # scope and goes on using what the Graph owns. See Pipeline.graph.
        pipeline.graph = graph
        pipeline.run = graph.build(make_run_options(cfg))

        if cfg.insight_enable:
            start_insight(cfg, pipeline, width, height, fps, step)
        if cfg.save_enable:
            step.detail(
                f"stills: {cfg.save_dir}/ every {cfg.save_every} frames "
                f"overlay={cfg.save_overlay}"
            )
        if cfg.video_enable:
            pipeline.writer, pipeline.writer_path = open_video_writer(cfg, width, height, fps)
            step.detail(
                f"video: {pipeline.writer_path} codec={cfg.video_codec} "
                f"fps={cfg.video_fps or fps} hud={cfg.video_hud}"
            )
        if not (cfg.save_enable or cfg.video_enable or cfg.insight_enable):
            # Not an error. `stall_causes` tells a stalled run to come back with
            # `--no-save --no-video`, and for a while the app answered that
            # advice with "enable at least one of output.save, output.video or
            # output.insight" -- refusing the one run that separates a slow app
            # from a stalled graph. Saying what the run does is enough.
            step.note(
                "no outputs are enabled, so this run writes nothing and measures "
                "the graph alone. That is what tells a slow app apart from a "
                "stalled source."
            )
        step.done(f"{self.graph_name} ready", timed=True)

    def build(self, cfg) -> Pipeline:
        """The startup sequence, as three numbered steps. Shared by every task."""
        with console.step("source", "probing the stream") as step:
            geometry = self.open_source(cfg, step)
        with console.step("model", f"loading {Path(cfg.model_path).name}") as step:
            pipeline = self.load_model(cfg, geometry[0], geometry[1], step)
        with console.step("pipeline", "building the Neat graph") as step:
            self.build_pipeline(cfg, pipeline, geometry, step)
        return pipeline

    def run(self, cfg, stopper: Stopper) -> int:
        """Fetch what is missing, build, and run, closing the pipeline whatever happens.

        This is the only place that fetches anything. A missing clip or model is
        downloaded into ``assets/`` here, so ``--validate``, which resolves
        exactly the same paths, stays offline.

        The ``finally`` covers a failure part-way through :meth:`build` as well
        as one during the run, which is why :attr:`pipeline` is published early:
        by the time the graph is built the Run holds the MLA, and leaving it
        held makes the *next* launch fail with a busy device.
        """
        with console.step("assets", "model archive and video source") as step:
            cfg = ensure_assets(cfg, self.name, step)
            # After the fetch, because the file has to exist to be reframed,
            # and before build(), because everything downstream -- the SPS
            # probe, the picture count, the source graph -- is written against
            # a raw stream.
            cfg = ensure_annex_b(cfg, step)
            step.done("ready")
        try:
            pipeline = self.build(cfg)
            console.banner("running", "press Ctrl-C to stop")
            return run_pipeline(pipeline, cfg, stopper, self.runtime(cfg, pipeline))
        finally:
            if self.pipeline is not None:
                self.pipeline.close()
