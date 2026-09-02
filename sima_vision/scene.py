"""The synthetic scene behind ``sima-vision preview``.

Renders what a config will look like, without a board and without a model.

The overlay, the masks and the background blur are plain numpy and OpenCV. Only
inference needs the MLA. So the appearance of a run can be produced anywhere,
which turns tuning ``visualization`` and ``blur`` from an edit-scp-ssh-run-scp
loop into an edit-and-look one.

**No model is run.** The detections are synthetic, placed over a synthetic
scene, and they exist only so the drawing code has something to draw. What you
are looking at is your config's *styling*, not its accuracy.
"""

from __future__ import annotations

from pathlib import Path

from . import runtime

#: Class ids used for the synthetic subjects, chosen so the COCO labels read
#: sensibly: person, person, person, car.
SUBJECT_CLASSES = (0, 0, 0, 2)


def build_scene(width: int, height: int):
    """Paint a synthetic frame with enough structure to judge a blur against.

    A flat colour would make any background treatment look identical. This has a
    gradient sky, a textured floor and hard edges, so ``kernel``, ``downscale``
    and ``pixelate`` all show their differences.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        A ``(frame, subjects)`` pair. ``subjects`` is a list of
        ``(x1, y1, x2, y2, class_id, score)`` in frame pixels.
    """
    cv2, np = runtime.cv2, runtime.np
    frame = np.zeros((height, width, 3), np.uint8)

    # Sky: a vertical gradient, so a blur has something smooth to work on.
    top, bottom = np.array([150, 110, 70], float), np.array([220, 200, 180], float)
    ramp = np.linspace(0.0, 1.0, height)[:, None]
    frame[:] = (top * (1 - ramp[..., None]) + bottom * ramp[..., None]).astype(np.uint8)

    # Floor, with a grid: hard edges are what make a blur obvious.
    horizon = int(height * 0.62)
    frame[horizon:] = (70, 78, 86)
    step = max(24, width // 24)
    for x in range(0, width, step):
        cv2.line(frame, (x, horizon), (int(x * 1.35 - width * 0.17), height), (92, 100, 110), 2)
    for i in range(1, 9):
        y = horizon + int((height - horizon) * (i / 9.0) ** 1.7)
        cv2.line(frame, (0, y), (width, y), (92, 100, 110), 2)

    # A couple of buildings, for depth behind the subjects.
    for cx, w, h, shade in ((0.12, 0.14, 0.30, 96), (0.78, 0.18, 0.38, 84)):
        x1 = int(width * cx)
        cv2.rectangle(
            frame, (x1, horizon - int(height * h)), (x1 + int(width * w), horizon),
            (shade, shade + 6, shade + 12), -1,
        )

    subjects = []
    # Three people at different depths, and a car. Sizes are fractions of the
    # frame so the preview looks the same at any --size.
    people = ((0.22, 0.38), (0.45, 0.30), (0.63, 0.22))
    # SUBJECT_CLASSES has a fourth entry for the car, so zip stops at `people`.
    for (cx, ph), class_id in zip(people, SUBJECT_CLASSES, strict=False):
        person_h = int(height * ph)
        person_w = int(person_h * 0.38)
        x1 = int(width * cx) - person_w // 2
        y2 = horizon + int((height - horizon) * (0.25 + (0.40 - ph)))
        y1 = y2 - person_h
        head_r = person_w // 3
        cv2.rectangle(frame, (x1, y1 + head_r), (x1 + person_w, y2), (58, 74, 190), -1)
        cv2.circle(frame, (x1 + person_w // 2, y1 + head_r), head_r, (140, 170, 220), -1)
        subjects.append((x1, y1, x1 + person_w, y2, class_id, 0.94 - 0.11 * len(subjects)))

    car_w, car_h = int(width * 0.20), int(height * 0.11)
    cx1, cy1 = int(width * 0.80) - car_w // 2, horizon + int((height - horizon) * 0.34)
    cv2.rectangle(frame, (cx1, cy1), (cx1 + car_w, cy1 + car_h), (52, 160, 200), -1)
    cv2.rectangle(
        frame, (cx1 + car_w // 5, cy1 - car_h // 2),
        (cx1 + 4 * car_w // 5, cy1), (70, 180, 215), -1,
    )
    subjects.append((cx1, cy1 - car_h // 2, cx1 + car_w, cy1 + car_h, SUBJECT_CLASSES[3], 0.71))

    return frame, subjects


def read_first_frame(source: str):
    """First frame of an image or video, or None when it cannot be read.

    Raw ``.h264`` elementary streams are exactly what OpenCV cannot open, and
    those are the files this repo ships, so failing here is expected rather
    than exceptional. The caller falls back to the synthetic scene.
    """
    cv2 = runtime.cv2
    path = Path(source)
    if not path.is_file():
        return None
    frame = cv2.imread(str(path))
    if frame is not None:
        return frame
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def boxes_from(subjects) -> list[dict]:
    """Turn scene subjects into the detection dicts the tasks consume."""
    return [
        {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
         "score": float(score), "class_id": int(class_id)}
        for x1, y1, x2, y2, class_id, score in subjects
    ]


def render(task, cfg, frame, subjects, labels: list[str]):
    """Draw one frame exactly the way a real run would.

    Goes through the task's own :meth:`TaskRuntime.render`, so what comes back
    is produced by the same code that draws the recording -- not a lookalike.

    Args:
        task: The :class:`~sima_vision.tasks.base.Task` being previewed.
        cfg: Its resolved configuration.
        frame: The BGR frame to draw on.
        subjects: Scene subjects from :func:`build_scene`.
        labels: Class names indexed by class id.

    Returns:
        A new annotated BGR frame.
    """
    pipeline = task.make_pipeline(cfg, labels)
    pipeline.frame_w, pipeline.frame_h = frame.shape[1], frame.shape[0]
    pipeline.fps = cfg.source_fps or 25
    results = task.sample_results(cfg, pipeline, frame, boxes_from(subjects))
    return task.runtime(cfg, pipeline).render(cfg, pipeline, frame, results, 24.0)


def build_frame(source: str | None, size: tuple[int, int]):
    """Get a frame to draw on, and the subjects to draw over it.

    Args:
        source: An image or video to use, or None for the synthetic scene.
        size: Synthetic scene size, used when ``source`` cannot be read.

    Returns:
        A ``(frame, subjects, origin, unreadable)`` tuple. ``origin`` describes
        where the frame came from; ``unreadable`` is True when a source was
        given but could not be opened, which the CLI turns into a warning.
    """
    frame = read_first_frame(source) if source else None
    if frame is not None:
        _, subjects = build_scene(frame.shape[1], frame.shape[0])
        return frame, subjects, str(source), False
    frame, subjects = build_scene(*size)
    return frame, subjects, f"synthetic scene {size[0]}x{size[1]}", bool(source)
