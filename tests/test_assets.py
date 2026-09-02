"""Resolving `--source` and `--model` down to files that exist.

Nothing here reaches the network: `urlopen` and `sima-cli` are both replaced.
What is being tested is the decision -- use this, fetch that, refuse the other
-- not the download, which is the same `download()` `fetch` already used.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

from sima_vision import assets
from sima_vision.tasks import TASKS

CLIP = "people-walking-outside-mall.h264"


@pytest.fixture
def here(tmp_path, monkeypatch):
    """Run in an empty directory, so `assets/` is this test's own."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def offline(monkeypatch):
    """Serve four bytes for any URL, and return the list of URLs asked for."""
    asked: list[str] = []

    class Response:
        headers = {"Content-Length": "4"}

        def __init__(self):
            self.left = [b"data"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size):
            return self.left.pop() if self.left else b""

    def urlopen(url, timeout=0):
        asked.append(url)
        return Response()

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    return asked


# ── where things live ──


def test_the_default_assets_directory_is_beside_you(here):
    assert assets.assets_root() == Path("assets")
    assert assets.default_model_path("detect").startswith("assets/models/")
    assert assets.default_source_uri("fall").endswith("people-walking-inside-mall.h264")


def test_the_assets_directory_can_be_moved(here, monkeypatch):
    monkeypatch.setenv(assets.ASSETS_ENV, str(here / "shared"))
    assert assets.models_dir() == here / "shared" / "models"
    assert assets.default_model_path("segment").startswith((here / "shared").as_posix())


def test_every_task_has_a_model_and_a_clip():
    for name in TASKS:
        entry = assets.CATALOGUE[name]
        assert entry.clip in assets.SAMPLE_VIDEOS
        assert entry.model_file.endswith(".tar.gz")
        assert assets.model_url(name).startswith(assets.MODEL_BASE)


def test_only_http_urls_count_as_downloadable():
    assert assets.is_url("https://example.com/clip.h264")
    assert assets.is_url("http://example.com/clip.h264")
    # An RTSP stream is opened, never fetched.
    assert not assets.is_url("rtsp://cam/live")
    assert not assets.is_url("assets/videos/clip.h264")


# ── sources ──


def test_a_source_url_is_downloaded_once(here, offline):
    first = assets.ensure_source("https://example.com/clip.h264")
    assert Path(first) == Path("assets/videos/clip.h264")
    assert Path(first).is_file()

    second = assets.ensure_source("https://example.com/clip.h264")
    assert second == first
    assert len(offline) == 1, "the second call must reuse the file on disk"


def test_a_missing_sample_clip_is_fetched_from_the_release(here, offline):
    uri = assets.ensure_source(f"assets/videos/{CLIP}")
    assert uri == f"assets/videos/{CLIP}"
    assert Path(uri).is_file()
    assert offline == [f"{assets.SAMPLE_RELEASE}/{CLIP}"]


def test_an_existing_file_is_left_alone(here, offline):
    clip = here / "mine.h264"
    clip.write_bytes(b"\x00\x00\x00\x01")
    assert assets.ensure_source("mine.h264") == "mine.h264"
    assert not offline


def test_an_unknown_missing_file_is_left_for_the_real_error(here, offline):
    """check_source_file describes a missing path far better than we could."""
    assert assets.ensure_source("nowhere/mine.h264") == "nowhere/mine.h264"
    assert not offline


def test_a_stream_source_is_never_touched(here, offline):
    assert assets.ensure_source("rtsp://cam/live", "rtsp") == "rtsp://cam/live"
    assert assets.ensure_source("", "usb") == ""
    assert not offline


def test_a_failed_source_download_says_so(here, monkeypatch):
    def boom(url, timeout=0):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(assets.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="could not download"):
        assets.ensure_source("https://example.com/clip.h264")
    assert not list(Path("assets").rglob("*.part"))


# ── models ──


def test_a_model_url_is_downloaded(here, offline):
    path = assets.ensure_model("https://example.com/det.tar.gz", "detect")
    assert Path(path) == Path("assets/models/det.tar.gz")
    assert Path(path).is_file()


def test_an_existing_model_is_used_as_it_stands(here, monkeypatch):
    monkeypatch.setattr(assets.shutil, "which", lambda _name: "/usr/bin/sima-cli")
    archive = here / "det.tar.gz"
    archive.write_bytes(b"tar")
    assert assets.ensure_model("det.tar.gz", "detect") == "det.tar.gz"


def test_a_missing_model_without_sima_cli_prints_the_command(here, monkeypatch):
    monkeypatch.setattr(assets.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError) as caught:
        assets.ensure_model(assets.default_model_path("detect"), "detect")
    message = str(caught.value)
    assert "sima-cli login" in message
    assert assets.model_url("detect") in message


def test_a_missing_model_is_fetched_with_sima_cli(here, monkeypatch):
    monkeypatch.setattr(assets.shutil, "which", lambda _name: "/usr/bin/sima-cli")
    seen = {}

    class Result:
        returncode = 0

    def fake_run(command, cwd=None, check=False, env=None):
        seen["command"] = command
        seen["cwd"] = Path(cwd)
        seen["env"] = env
        (Path(cwd) / Path(command[-1]).name).write_bytes(b"tar")
        return Result()

    monkeypatch.setattr(assets.subprocess, "run", fake_run)
    path = assets.default_model_path("segment")
    assert assets.ensure_model(path, "segment") == path
    assert seen["command"] == ["sima-cli", "download", assets.model_url("segment")]
    assert seen["cwd"] == Path("assets/models")
    assert Path(path).is_file()
    # Without this sima-cli opens with "update now? [Y/n]" and waits for an
    # answer nobody is there to give, then aborts the download with it.
    assert seen["env"]["SIMA_CLI_CHECK_FOR_UPDATE"] == "0"
    assert "PATH" in seen["env"], "the rest of the environment must survive"


def test_sima_cli_failing_is_reported(here, monkeypatch):
    monkeypatch.setattr(assets.shutil, "which", lambda _name: "/usr/bin/sima-cli")

    class Result:
        returncode = 1

    monkeypatch.setattr(
        assets.subprocess, "run", lambda *a, **k: Result()
    )
    with pytest.raises(RuntimeError, match="sima-cli login"):
        assets.ensure_model(assets.default_model_path("fall"), "fall")


def test_a_model_we_have_no_url_for_is_not_guessed_at(here, monkeypatch):
    monkeypatch.setattr(assets.shutil, "which", lambda _name: "/usr/bin/sima-cli")
    with pytest.raises(RuntimeError, match="model archive not found"):
        assets.ensure_model("some/other-model.tar.gz", "detect")


# ── the two together ──


def test_ensure_assets_leaves_a_resolved_config_alone(here, offline):
    clip = here / "mine.h264"
    clip.write_bytes(b"\x00\x00\x00\x01")
    archive = here / "det.tar.gz"
    archive.write_bytes(b"tar")

    cfg = TASKS["detect"]().load(
        None, {"source.uri": "mine.h264", "model.path": "det.tar.gz"}, use_file=False
    )
    assert assets.ensure_assets(cfg, "detect") is cfg
    assert not offline


def test_ensure_assets_fills_in_both_defaults(here, offline, monkeypatch):
    monkeypatch.setattr(assets.shutil, "which", lambda _name: "/usr/bin/sima-cli")

    class Result:
        returncode = 0

    monkeypatch.setattr(
        assets.subprocess,
        "run",
        lambda command, cwd=None, check=False, env=None: (
            (Path(cwd) / Path(command[-1]).name).write_bytes(b"tar"), Result()
        )[1],
    )

    cfg = TASKS["detect"]().load(None, {}, use_file=False)
    resolved = assets.ensure_assets(cfg, "detect")
    assert Path(resolved.source_uri).is_file()
    assert Path(resolved.model_path).is_file()
    assert offline == [f"{assets.SAMPLE_RELEASE}/{CLIP}"]
