"""`sima-vision setup network`: sharing the PC's internet with the board.

The probe itself is platform code, so it is faked here and what gets tested is
the reasoning on top: which adapter is the internet, which is the board, when
to warn, and whether anything is claimed that was not actually established.

That last one is the point. Two things look diagnostic and are not: `Forwarding`
is Disabled on a working ICS setup because ICS does its own NAT, and
192.168.137.1 outlives the sharing that put it there. So a non-elevated run must
say "not known", never "not shared".
"""

from __future__ import annotations

import pytest

from sima_vision import netsetup
from sima_vision.cli import main
from sima_vision.netsetup import Adapter, Report

WIFI = Adapter(name="Wi-Fi", ip="192.168.18.15", origin="Dhcp", has_internet=True)
ETH = Adapter(name="Ethernet", ip="192.168.137.1", origin="Manual")
SPARE = Adapter(name="Ethernet 2", ip="10.9.9.1", origin="Manual")


def windows(adapters, elevated=False, shared=None):
    return Report(
        platform="windows",
        adapters=list(adapters),
        elevated=elevated,
        shared=shared or {},
        sharing_known=elevated,
    )


@pytest.fixture
def seen(monkeypatch):
    """Install a report and capture nothing else."""
    def install(report):
        monkeypatch.setattr(netsetup, "probe", lambda: report)
    return install


# -- working out the topology --


def test_the_internet_adapter_is_the_one_with_a_route():
    assert windows([ETH, WIFI]).internet is WIFI


def test_the_board_is_on_the_adapter_with_no_way_out():
    assert windows([WIFI, ETH]).board_link is ETH


def test_the_ics_address_settles_a_tie():
    """Windows only ever puts 192.168.137.1 on the adapter it shares."""
    report = windows([WIFI, SPARE, ETH])
    assert report.board_link is ETH, "the ICS address wins over the other candidate"


def test_an_adapter_with_no_address_is_not_the_board_link():
    blank = Adapter(name="Bluetooth", ip="")
    assert windows([WIFI, blank]).board_link is None


# -- what it says --


def test_no_internet_at_all_is_a_warning(seen, capsys):
    seen(windows([ETH]))
    assert netsetup.run_setup_network(None, False) == 1
    out = capsys.readouterr().out
    assert "WARNING" in out and "route to the internet" in out


def test_no_second_network_is_the_warning_you_asked_for(seen, capsys):
    """The board is not plugged in, or is not powered on."""
    seen(windows([WIFI]))
    assert netsetup.run_setup_network(None, False) == 1
    out = capsys.readouterr().out
    assert "WARNING  no second network found to share with." in out
    assert "Plug the board into this PC's Ethernet port" in out


def test_it_names_both_ends_when_it_can(seen, capsys):
    seen(windows([WIFI, ETH]))
    netsetup.run_setup_network(None, False)
    out = capsys.readouterr().out
    assert "internet comes in on:  Wi-Fi" in out
    assert "the DevKit is on:      Ethernet" in out


def test_without_admin_it_admits_it_cannot_tell(seen, capsys):
    """The failure mode this whole module exists to avoid: a confident guess."""
    seen(windows([WIFI, ETH], elevated=False))
    netsetup.run_setup_network(None, False)
    out = capsys.readouterr().out
    assert "needs Administrator to read" in out
    assert "sharing is OFF" not in out, "must not claim what it did not check"


def test_with_admin_it_reports_sharing_off(seen, capsys):
    seen(windows([WIFI, ETH], elevated=True, shared={}))
    netsetup.run_setup_network(None, False)
    assert "sharing is OFF" in capsys.readouterr().out


def test_with_admin_it_reports_sharing_on(seen, capsys):
    seen(windows([WIFI, ETH], elevated=True,
                 shared={"Wi-Fi": "public", "Ethernet": "private"}))
    netsetup.run_setup_network(None, False)
    assert "sharing is ON: Wi-Fi -> Ethernet" in capsys.readouterr().out


def test_sharing_the_wrong_adapter_is_called_out(seen, capsys):
    """Shared, but with something that is not where the board is."""
    seen(windows([WIFI, SPARE, ETH], elevated=True,
                 shared={"Wi-Fi": "public", "Ethernet 2": "private"}))
    netsetup.run_setup_network(None, False)
    out = capsys.readouterr().out
    assert "WARNING" in out and "not Ethernet" in out


# -- applying --


def test_apply_without_admin_prints_the_command(seen, capsys):
    seen(windows([WIFI, ETH], elevated=False))
    assert netsetup.run_setup_network(None, True) == 1
    out = capsys.readouterr().out
    assert "needs Administrator" in out
    assert "EnableSharing(0)" in out and "EnableSharing(1)" in out


def test_the_command_names_the_adapters_the_right_way_round():
    command = netsetup.sharing_command("Wi-Fi", "Ethernet")
    # 0 is PUBLIC (shares its internet), 1 is PRIVATE (receives it). Swapping
    # them shares the board's dead link with the internet, which does nothing.
    assert "$p='Wi-Fi'" in command and "$q='Ethernet'" in command
    assert command.index("$p=") < command.index("$q=")
    assert "EnableRebootPersistConnection" in command, "or it dies on reboot"


def test_apply_runs_it_when_elevated(seen, monkeypatch, capsys):
    seen(windows([WIFI, ETH], elevated=True))
    ran = {}

    class Ok:
        returncode = 0

    def fake_run(command, **kwargs):
        ran["command"] = command
        return Ok()

    monkeypatch.setattr(netsetup.subprocess, "run", fake_run)
    netsetup.run_setup_network(None, True)
    assert "-Command" in ran["command"]
    assert "EnableSharing" in ran["command"][-1]
    assert "dhclient" in capsys.readouterr().out, "say what to do on the board next"


# -- asking the board --


def test_the_board_ladder_stops_at_the_first_failure(seen, monkeypatch, capsys):
    seen(windows([WIFI, ETH], elevated=True,
                 shared={"Wi-Fi": "public", "Ethernet": "private"}))
    monkeypatch.setattr(netsetup, "check_board", lambda _h: [
        ("an address on the shared network", True),
        ("a route out", True),
        ("the PC answers", True),
        ("the internet answers", False),
        ("names resolve", False),
    ])
    assert netsetup.run_setup_network("sima@192.168.137.50", False) == 1
    out = capsys.readouterr().out
    assert "First thing that fails: the internet answers" in out
    assert "not\n  forwarding traffic" in out


def test_a_board_that_is_fully_online_is_a_pass(seen, monkeypatch, capsys):
    seen(windows([WIFI, ETH], elevated=True,
                 shared={"Wi-Fi": "public", "Ethernet": "private"}))
    monkeypatch.setattr(netsetup, "check_board", lambda _h: [
        ("an address on the shared network", True),
        ("names resolve", True),
    ])
    assert netsetup.run_setup_network("sima@192.168.137.50", False) == 0
    assert "The board is online" in capsys.readouterr().out


def test_only_dns_failing_says_only_dns(seen, monkeypatch, capsys):
    seen(windows([WIFI, ETH], elevated=True,
                 shared={"Wi-Fi": "public", "Ethernet": "private"}))
    monkeypatch.setattr(netsetup, "check_board", lambda _h: [
        ("the internet answers", True),
        ("names resolve", False),
    ])
    netsetup.run_setup_network("sima@192.168.137.50", False)
    out = capsys.readouterr().out
    assert "only DNS is missing" in out
    assert "resolv.conf" in out


def test_an_unreachable_board_is_explained_not_raised(seen, monkeypatch, capsys):
    seen(windows([WIFI, ETH]))

    def refuse(_host):
        raise ConnectionError("connect to host 192.168.137.50 port 22: timed out")

    monkeypatch.setattr(netsetup, "check_board", refuse)
    assert netsetup.run_setup_network("sima@192.168.137.50", False) == 1
    out = capsys.readouterr().out
    assert "cannot reach it over SSH" in out
    assert "Fix the PC side first" in out


# -- through the CLI --


def test_the_cli_reaches_it(monkeypatch, capsys):
    monkeypatch.setattr(netsetup, "probe", lambda: windows([WIFI, ETH]))
    assert main(["setup", "network"]) == 0
    assert "Internet sharing, PC -> DevKit" in capsys.readouterr().out


def test_setup_with_no_topic_is_an_error(capsys):
    """argparse should refuse, not fall through to a default topic."""
    with pytest.raises(SystemExit) as exit_info:
        main(["setup"])
    assert exit_info.value.code == 2


def test_non_windows_says_so_rather_than_pretending(seen, capsys):
    seen(Report(platform="linux", adapters=[
        Adapter(name="wlan0", ip="192.168.1.5", has_internet=True),
        Adapter(name="eth0", ip="192.168.137.1"),
    ]))
    assert netsetup.run_setup_network(None, False) == 1
    out = capsys.readouterr().out
    assert "Windows-only for now" in out
    assert "MASQUERADE" in out and "eth0" in out and "wlan0" in out
