"""The task registry.

Adding a fourth application means writing one module here and adding it to
:data:`TASKS`. Everything the CLI needs -- the subcommand, its flags, its
config sections, its ``--validate`` output -- comes off the
:class:`~sima_vision.tasks.base.Task`.
"""

from __future__ import annotations

from .base import Task
from .detect import DetectTask
from .fall import FallTask
from .segment import SegmentTask

#: Subcommand name -> task class, in the order they appear in ``--help``.
TASKS: dict[str, type[Task]] = {
    DetectTask.name: DetectTask,
    SegmentTask.name: SegmentTask,
    FallTask.name: FallTask,
}

__all__ = ["TASKS", "Task", "DetectTask", "SegmentTask", "FallTask"]
