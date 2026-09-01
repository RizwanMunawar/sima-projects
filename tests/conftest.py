"""Make the numpy/cv2 halves of the package testable off the DevKit.

``pyneat`` is an aarch64 wheel that only exists on the board, but almost
everything that runs per frame -- box parsing, mask decoding, compositing and
the whole overlay -- is plain numpy and OpenCV. Binding those two here lets that
code be tested anywhere, and leaves only the graph construction untested off
the board.
"""

from __future__ import annotations

import pytest

import sima_vision.runtime as rt


@pytest.fixture(scope="session", autouse=True)
def bind_cv2_and_numpy():
    """Do what load_runtime_dependencies() does, minus pyneat."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    rt.cv2 = cv2
    rt.np = np
    rt.FONT = cv2.FONT_HERSHEY_SIMPLEX
    yield
