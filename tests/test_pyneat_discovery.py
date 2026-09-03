"""Finding pyneat, which does not live where pip put sima-vision.

`sima-cli sdk setup` installs pyneat into a virtualenv of its own at ~/pyneat.
Nothing puts that on the default path, so `pip install sima-vision` followed by
`sima-vision detect` fails with ModuleNotFoundError for every install that is
not inside that venv, which is most of them.

These build the board's directory layout under tmp_path and check the search
picks the right one, refuses the wrong one, and explains itself either way.
The version check is the part that matters most: pyneat is a compiled
extension, so putting a 3.10 venv on a 3.12 path trades a clear import error
for an undefined-symbol crash out of the dynamic linker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sima_vision import runtime

THIS = f"python{sys.version_info.major}.{sys.version_info.minor}"
OTHER = "python3.7" if THIS != "python3.7" else "python3.8"


def make_venv(root: Path, version: str, with_pyneat: bool = True) -> Path:
    """The parts of a virtualenv the search actually looks at."""
    site = root / "lib" / version / "site-packages"
    site.mkdir(parents=True)
    (root / "bin").mkdir(exist_ok=True)
    if with_pyneat:
        (site / "pyneat").mkdir()
        (site / "pyneat" / "__init__.py").write_text("", encoding="utf-8")
    return site


@pytest.fixture(autouse=True)
def no_real_env(monkeypatch, tmp_path):
    """Nothing on the developer's own machine may answer these."""
    monkeypatch.delenv(runtime.PYNEAT_ENV, raising=False)
    monkeypatch.setattr(runtime, "PYNEAT_HOMES", (str(tmp_path / "absent"),))


# -- finding it --


def test_the_env_var_wins(tmp_path, monkeypatch):
    site = make_venv(tmp_path / "custom", THIS)
    monkeypatch.setenv(runtime.PYNEAT_ENV, str(tmp_path / "custom"))
    found, note = runtime.find_pyneat_env()
    assert found == site
    assert "custom" in note


def test_the_usual_locations_are_searched_in_order(tmp_path, monkeypatch):
    first = make_venv(tmp_path / "pyneat", THIS)
    make_venv(tmp_path / "nvme", THIS)
    monkeypatch.setattr(
        runtime, "PYNEAT_HOMES", (str(tmp_path / "pyneat"), str(tmp_path / "nvme"))
    )
    found, _ = runtime.find_pyneat_env()
    assert found == first


def test_a_home_that_is_not_there_is_skipped(tmp_path, monkeypatch):
    site = make_venv(tmp_path / "real", THIS)
    monkeypatch.setattr(
        runtime, "PYNEAT_HOMES", (str(tmp_path / "gone"), str(tmp_path / "real"))
    )
    assert runtime.find_pyneat_env()[0] == site


def test_a_venv_without_pyneat_is_not_taken(tmp_path, monkeypatch):
    """Some other venv at the same path must not be mistaken for this one."""
    make_venv(tmp_path / "pyneat", THIS, with_pyneat=False)
    monkeypatch.setattr(runtime, "PYNEAT_HOMES", (str(tmp_path / "pyneat"),))
    found, note = runtime.find_pyneat_env()
    assert found is None
    assert "no pyneat" in note


# -- refusing it --


def test_a_venv_for_another_python_is_refused_by_name(tmp_path, monkeypatch):
    """The wrong ABI would crash the linker, so it is named, not used."""
    make_venv(tmp_path / "pyneat", OTHER)
    monkeypatch.setattr(runtime, "PYNEAT_HOMES", (str(tmp_path / "pyneat"),))
    found, note = runtime.find_pyneat_env()
    assert found is None
    assert OTHER in note and THIS in note
    assert "bin/python3" in note, "the note has to name the interpreter that works"


def test_the_right_version_wins_when_a_venv_holds_several(tmp_path, monkeypatch):
    make_venv(tmp_path / "pyneat", OTHER)
    site = make_venv(tmp_path / "pyneat", THIS)
    monkeypatch.setattr(runtime, "PYNEAT_HOMES", (str(tmp_path / "pyneat"),))
    assert runtime.find_pyneat_env()[0] == site


def test_a_wrong_env_var_says_it_was_the_env_var(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv(runtime.PYNEAT_ENV, str(tmp_path / "empty"))
    found, note = runtime.find_pyneat_env()
    assert found is None
    assert runtime.PYNEAT_ENV in note


def test_the_env_var_stops_the_search(tmp_path, monkeypatch):
    """An explicit answer is not second-guessed by the defaults."""
    make_venv(tmp_path / "default", THIS)
    (tmp_path / "explicit").mkdir()
    monkeypatch.setattr(runtime, "PYNEAT_HOMES", (str(tmp_path / "default"),))
    monkeypatch.setenv(runtime.PYNEAT_ENV, str(tmp_path / "explicit"))
    assert runtime.find_pyneat_env()[0] is None


# -- what it tells you --


def test_the_message_off_board_points_at_the_board(monkeypatch):
    monkeypatch.setattr(runtime.glob, "glob", lambda _pattern: [])
    message = runtime.missing_pyneat_message("no pyneat virtualenv found")
    assert "does not look like a DevKit" in message
    assert "sima-vision watch" in message, "say how to use the board from here"


def test_the_message_on_board_gives_the_install_command(monkeypatch):
    monkeypatch.setattr(runtime.glob, "glob", lambda _p: ["/usr/lib/python3.11/dist-packages"])
    monkeypatch.setattr(sys, "platform", "linux")
    message = runtime.missing_pyneat_message("no pyneat virtualenv found")
    assert "~/pyneat/bin/pip install sima-vision" in message
    assert runtime.PYNEAT_ENV in message
    assert "sima-cli sdk setup" in message


def test_every_message_is_ascii():
    """These print on a board whose console encoding is nobody's guess."""
    for note in ("no pyneat virtualenv found", "found pyneat, but built for x"):
        text = runtime.missing_pyneat_message(note)
        assert all(ord(ch) < 128 for ch in text), text


# -- loading --


def test_a_working_interpreter_is_never_interfered_with(monkeypatch):
    """If `import pyneat` succeeds, no venv is searched for or put on the path.

    The board's `/usr/lib/python3*/dist-packages` is still added, because that
    is where its OpenCV lives and every run needs it. What must not happen is a
    second environment appearing underneath an interpreter that was already
    working.
    """
    monkeypatch.setattr(runtime, "pyneat", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = list(sys.path)

    def refuse():
        raise AssertionError("must not search when the plain import works")

    monkeypatch.setattr(runtime, "find_pyneat_env", refuse)
    monkeypatch.setitem(sys.modules, "pyneat", object())
    try:
        runtime.load_runtime_dependencies()
    finally:
        runtime.pyneat = None

    added = [p for p in sys.path if p not in before]
    assert all("dist-packages" in p for p in added), added
    assert not any("site-packages" in p for p in added), added


def test_the_venv_goes_ahead_of_the_current_environment(tmp_path, monkeypatch, capsys):
    """It holds the numpy<2 pyneat was built against, which has to win."""
    site = make_venv(tmp_path / "pyneat", THIS)
    monkeypatch.setattr(runtime, "pyneat", None)
    monkeypatch.setattr(runtime, "PYNEAT_HOMES", (str(tmp_path / "pyneat"),))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "pyneat", raising=False)

    runtime.load_runtime_dependencies()
    try:
        assert sys.path[0] == str(site), "ahead of everything, not appended"
        assert runtime.pyneat.__file__.startswith(str(site))
        # It says so, because a path appearing from nowhere is worse than a line.
        assert "[pyneat] using pyneat from" in capsys.readouterr().out
    finally:
        runtime.pyneat = None
        sys.modules.pop("pyneat", None)


def test_a_missing_pyneat_raises_the_explanation(monkeypatch):
    monkeypatch.setattr(runtime, "pyneat", None)
    monkeypatch.delitem(sys.modules, "pyneat", raising=False)
    monkeypatch.setattr(runtime, "find_pyneat_env", lambda: (None, "no pyneat virtualenv found"))
    with pytest.raises(ImportError, match="pyneat"):
        runtime.load_runtime_dependencies()
    runtime.pyneat = None


def test_doctor_names_the_venv_root_not_its_lib_directory(tmp_path, monkeypatch, capsys):
    """site-packages is three levels down, and `<root>/lib/bin/pip` helps nobody."""
    from sima_vision.cli import run_doctor

    root = tmp_path / "pyneat"
    make_venv(root, THIS)
    monkeypatch.setenv(runtime.PYNEAT_ENV, str(root))
    run_doctor()
    # Both sides, or the separators differ on Windows and the test fails for
    # the wrong reason.
    out = capsys.readouterr().out.replace("\\", "/")
    assert f"{str(root).replace(chr(92), '/')}/bin/pip install sima-vision" in out
    assert "/lib/bin/pip" not in out


#: Not a real path anywhere, on purpose. A test that names the board's actual
#: dist-packages passes or fails on whether the host happens to have it and
#: whether it is already on sys.path, neither of which is what is being tested.
FAKE_DIST = "/nowhere/sima-vision-test/dist-packages"


def test_whatever_the_glob_finds_goes_on_the_path(monkeypatch):
    """The board's OpenCV branch, exercised on every platform.

    Three tests in this file have now been written to pass on the author's
    machine for a reason that had nothing to do with the behaviour: twice
    because `/usr/lib/python3*/dist-packages` does not exist on Windows, and
    once because on Linux it is already on `sys.path` so inserting it adds
    nothing. Faking the glob with a path that exists nowhere and is on no
    `sys.path` removes the host from the question entirely.
    """
    monkeypatch.setattr(runtime, "pyneat", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(runtime.glob, "glob", lambda _p: [FAKE_DIST])
    monkeypatch.setitem(sys.modules, "pyneat", type(sys)("pyneat"))
    before = list(sys.path)
    try:
        runtime.load_runtime_dependencies()
    finally:
        runtime.pyneat = None

    assert [p for p in sys.path if p not in before] == [FAKE_DIST]


def test_a_path_already_present_is_not_added_twice(monkeypatch):
    """Repeated runs in one process must not grow sys.path."""
    monkeypatch.setattr(runtime, "pyneat", None)
    monkeypatch.setattr(sys, "path", [FAKE_DIST, *sys.path])
    monkeypatch.setattr(runtime.glob, "glob", lambda _p: [FAKE_DIST])
    monkeypatch.setitem(sys.modules, "pyneat", type(sys)("pyneat"))
    before = list(sys.path)
    try:
        runtime.load_runtime_dependencies()
    finally:
        runtime.pyneat = None

    assert sys.path == before
