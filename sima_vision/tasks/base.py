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
from ..media import check_source_file, resolve_source_geometry, source_frame_count
from ..neat import (
    build_task_graph,
    describe_preprocess,
    make_model,
    make_run_options,
    resolve_flow_control,
)
from ..runloop import Stopper, TaskRuntime, run_pipeline
from ..runtime import FAMILY_DECODE_TOKENS
from ..sinks import Pipeline, load_labels, open_video_writer, start_insight


class Task:
    """One application: detect, segment or fall.

    A task supplies its config type, its CLI flags, its graph labels and a
    :class:`~sima_vision.runloop.TaskRuntime`. Everything else -- probing the
    source, loading the model, building the graph, bringing up Insight, opening
    the writer, the pull loop and the closing report -- is shared.

    Attributes:
        name: Subcommand name, such as ``detect``. Also names the packaged
            starter config, ``sima_vision/configs/<name>.yaml``.
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

    def sample_results(self, cfg, pipeline: Pipeline, frame, boxes: list[dict]):
        """Synthetic results for ``sima-vision preview``.

        Turns plain boxes into whatever this task's ``render`` expects, so a
        preview exercises the real drawing code rather than a stand-in. Only
        ever called by :mod:`sima_vision.preview`; no model is involved.
        """
        return boxes

    def build(self, cfg) -> Pipeline:
        """The startup sequence. Identical for all three tasks."""
        step = lambda msg: print(msg, flush=True)  # noqa: E731

        check_source_file(cfg)
        width, height, fps = resolve_source_geometry(cfg)
        step(f"source: type={cfg.source_type} uri={cfg.source_uri or '<default camera>'} "
             f"stream={width}x{height}@{fps}")
        step(describe_preprocess(cfg, width, height))

        step("loading model (first load unpacks the archive, this can take a minute)...")
        model = make_model(cfg, width, height)
        labels = load_labels(cfg.labels_path)
        step(
            f"model: {cfg.model_path} family={cfg.family} "
            f"decode_type={FAMILY_DECODE_TOKENS[cfg.family]} labels={len(labels)}"
        )

        # Published before the graph exists so run()'s finally can close a
        # pipeline that failed part-way through building.
        pipeline = self.make_pipeline(cfg, labels)
        self.pipeline = pipeline
        pipeline.model = model
        pipeline.frame_w, pipeline.frame_h, pipeline.fps = width, height, fps
        pipeline.source_frames = source_frame_count(cfg)
        if pipeline.source_frames:
            step(f"source: {pipeline.source_frames} coded pictures in the clip")

        self.prepare(cfg, pipeline, step)

        preset, policy = resolve_flow_control(cfg)
        step(f"runtime: preset={preset} overflow={policy} queue_depth={cfg.queue_depth} "
             f"output_buffers={cfg.output_buffers} (2 outputs -> "
             f"{2 * cfg.output_buffers} decoder buffers in flight)")
        if policy == "block":
            step(
                "       block keeps every frame, so the run takes longer than the clip.\n"
                "       Output length matches the input. This is the right mode for a file."
            )
        elif cfg.source_type == "video":
            step(
                "[warn] overflow_policy drops frames, so the recording will be shorter\n"
                "       than the input and will play fast. Use auto for a file source."
            )

        step("building graph...")
        graph = build_task_graph(
            cfg, model, width, height, fps,
            self.graph_name, self.result_label, self.output_label,
        )
        if cfg.profile:
            step(f"Backend:\n{graph.describe_backend()}")
        pipeline.graph = graph
        pipeline.run = graph.build(make_run_options(cfg))
        step("graph built")

        if cfg.insight_enable:
            start_insight(cfg, pipeline, width, height, fps, step)
        if cfg.save_enable:
            step(f"save: dir={cfg.save_dir} every={cfg.save_every} overlay={cfg.save_overlay}")
        if cfg.video_enable:
            pipeline.writer, pipeline.writer_path = open_video_writer(cfg, width, height, fps)
            step(
                f"video: {pipeline.writer_path} codec={cfg.video_codec} "
                f"fps={cfg.video_fps or fps} hud={cfg.video_hud}"
            )
        step("running. press Ctrl-C to stop.")
        return pipeline

    def run(self, cfg, stopper: Stopper) -> int:
        """Build and run, closing the pipeline whatever happens.

        This is the only place that fetches anything. A missing clip or model
        is downloaded into ``assets/`` here, so ``--validate`` and ``preview``
        -- which resolve exactly the same paths -- stay offline.

        The ``finally`` covers a failure part-way through :meth:`build` as well
        as one during the run, which is why :attr:`pipeline` is published early:
        by the time the graph is built the Run holds the MLA, and leaving it
        held makes the *next* launch fail with a busy device.
        """
        cfg = ensure_assets(cfg, self.name)
        try:
            pipeline = self.build(cfg)
            return run_pipeline(pipeline, cfg, stopper, self.runtime(cfg, pipeline))
        finally:
            if self.pipeline is not None:
                self.pipeline.close()
