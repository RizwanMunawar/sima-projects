"""Live YOLO computer vision on a SiMa Modalix DevKit 3.0.

Three applications share one pipeline: object detection, instance segmentation
with an optional background blur, and fall detection with SMTP alerts. They
differ only in what they do with a frame once the MLA has finished with it, so
everything up to that point -- config loading, source geometry, the Neat graph,
sample decoding, drawing and the sinks -- lives in this package and is written
once.

Run them from the command line::

    sima-vision detect  --source clip.h264 --model yolo26m-det.tar.gz
    sima-vision segment --source clip.h264 --model yolo26m-seg.tar.gz --blur
    sima-vision fall    --source rtsp://camera/live --alert-to ops@example.com

or from Python, with the same settings under the same names::

    from sima_vision import run, validate

    validate("detect", conf=0.5)                           # no board needed
    run("detect", source="clip.h264", model="yolo26m-det.tar.gz", conf=0.5)

Everything runs **on the DevKit**, not in the x86 SDK container: ``pyneat`` is
compiled for aarch64. The imports that need it are deferred, so ``--validate``
and ``--help`` work anywhere.
"""

from __future__ import annotations

__version__ = "0.0.2"

from .api import load, run, validate

__all__ = ["__version__", "load", "run", "validate"]
