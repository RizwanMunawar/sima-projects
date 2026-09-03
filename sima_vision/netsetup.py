"""`sima-vision setup network`: sharing your PC's internet with the DevKit.

The DevKit has no internet of its own. It is cabled straight to your PC, so the
PC has to pass its connection along, and the board needs three things from it:
an address, a route out, and something that answers DNS. On Windows that is
Internet Connection Sharing; on Linux and macOS it is NAT plus a DHCP server.

None of that is hard, but all of it is silent when it goes wrong. The board
just sits there. So this command's first job is to *say what it sees* rather
than to change anything:

    sima-vision setup network              look, explain, change nothing
    sima-vision setup network --apply      set it up (Windows, needs admin)

What it can see without admin rights is limited, and it says so rather than
guessing. Two things people assume are diagnostic and are not:

* `Forwarding: Disabled` on the adapters. ICS does its own NAT and does not use
  that per-interface flag, so it reads Disabled on a perfectly working setup.
* `192.168.137.1` being present. ICS assigns that address, but it is left
  behind when sharing is turned off, so it outlives the thing it implies.

The one signal that never lies is whether the board can actually reach
anything, which is why `--host` runs the checks on the board itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

#: The address Windows ICS always gives the adapter it shares to. It is not
#: configurable, which is the one convenient thing about it.
ICS_HOST = "192.168.137.1"
ICS_NETWORK = "192.168.137."

#: One PowerShell round-trip for everything worth knowing, as JSON. Written as
#: a single expression so it can be passed with -Command and needs no file.
WINDOWS_PROBE = r"""
$ErrorActionPreference = 'SilentlyContinue'
$adapters = Get-NetAdapter | Where-Object Status -eq 'Up' | ForEach-Object {
  $alias = $_.Name
  $ip = Get-NetIPAddress -InterfaceAlias $alias -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike '169.254.*' } | Select-Object -First 1
  $gw = Get-NetRoute -InterfaceAlias $alias -DestinationPrefix '0.0.0.0/0' | Select-Object -First 1
  [pscustomobject]@{
    name        = $alias
    ip          = $(if ($ip) { $ip.IPAddress } else { '' })
    origin      = $(if ($ip) { "$($ip.PrefixOrigin)" } else { '' })
    has_internet = [bool]$gw
  }
}
$elevated = ([Security.Principal.WindowsPrincipal] `
  [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
  [Security.Principal.WindowsBuiltInRole]::Administrator)
$shared = @()
if ($elevated) {
  $net = New-Object -ComObject HNetCfg.HNetShare
  foreach ($c in $net.EnumEveryConnection) {
    $props = $net.NetConnectionProps($c)
    $cfg = $net.INetSharingConfigurationForINetConnection($c)
    if ($cfg.SharingEnabled) {
      $shared += [pscustomobject]@{
        name = $props.Name
        role = $(if ($cfg.SharingConnectionType -eq 0) { 'public' } else { 'private' })
      }
    }
  }
}
@{ adapters = @($adapters); elevated = $elevated; shared = @($shared) } |
  ConvertTo-Json -Depth 4 -Compress
"""


@dataclass
class Adapter:
    name: str
    ip: str = ""
    origin: str = ""
    has_internet: bool = False


@dataclass
class Report:
    """What could be established about this machine's networking."""

    platform: str
    adapters: list[Adapter] = field(default_factory=list)
    elevated: bool = False
    #: Adapter name -> "public" / "private", empty when it could not be read.
    shared: dict[str, str] = field(default_factory=dict)
    #: True only when sharing was actually read, not merely not seen.
    sharing_known: bool = False
    error: str = ""

    @property
    def internet(self) -> Adapter | None:
        """The adapter with a default route: the one worth sharing."""
        return next((a for a in self.adapters if a.has_internet), None)

    @property
    def board_link(self) -> Adapter | None:
        """The adapter the DevKit is most likely on.

        An adapter that is up, has no route to the internet, and is not
        link-local. If one of them already carries the ICS address, that is the
        one, because Windows only ever puts it on the shared adapter.
        """
        candidates = [
            a for a in self.adapters if not a.has_internet and a.ip
        ]
        for adapter in candidates:
            if adapter.ip == ICS_HOST:
                return adapter
        return candidates[0] if candidates else None


def probe_windows() -> Report:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:  # pragma: no cover - Windows always has one
        return Report(platform="windows", error="no powershell on PATH")
    result = subprocess.run(  # noqa: S603
        [powershell, "-NoProfile", "-NonInteractive", "-Command", WINDOWS_PROBE],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False, timeout=60,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return Report(platform="windows", error=result.stderr.strip() or "probe failed")
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return Report(platform="windows", error=f"unreadable probe output: {exc}")

    adapters = [
        Adapter(
            name=item.get("name", ""),
            ip=item.get("ip") or "",
            origin=item.get("origin") or "",
            has_internet=bool(item.get("has_internet")),
        )
        for item in raw.get("adapters") or []
    ]
    elevated = bool(raw.get("elevated"))
    shared = {
        item["name"]: item["role"]
        for item in raw.get("shared") or []
        if item.get("name")
    }
    return Report(
        platform="windows",
        adapters=adapters,
        elevated=elevated,
        shared=shared,
        # Only an elevated probe can read sharing at all, so an empty result
        # from a normal shell means "not known", never "not shared".
        sharing_known=elevated,
    )


def probe_posix() -> Report:
    """Enough to name the interfaces on Linux and macOS.

    No sharing state: there is no single thing to read for it the way there is
    on Windows, and guessing from nftables would be worse than saying nothing.
    """
    adapters: list[Adapter] = []
    if shutil.which("ip"):
        routes = subprocess.run(  # noqa: S603
            ["ip", "-json", "route"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False, timeout=30,
        )
        default = ""
        try:
            for entry in json.loads(routes.stdout or "[]"):
                if entry.get("dst") == "default":
                    default = entry.get("dev", "")
                    break
        except json.JSONDecodeError:
            pass
        addresses = subprocess.run(  # noqa: S603
            ["ip", "-json", "-4", "addr"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False, timeout=30,
        )
        try:
            for entry in json.loads(addresses.stdout or "[]"):
                name = entry.get("ifname", "")
                if name == "lo":
                    continue
                info = (entry.get("addr_info") or [{}])[0]
                adapters.append(Adapter(
                    name=name,
                    ip=info.get("local", "") or "",
                    origin="",
                    has_internet=(name == default),
                ))
        except json.JSONDecodeError:
            pass
    return Report(platform=sys.platform, adapters=adapters)


def probe() -> Report:
    return probe_windows() if sys.platform == "win32" else probe_posix()


def sharing_body(public: str, private: str) -> str:
    """The PowerShell that turns ICS on. Elevation is the caller's problem."""
    return (
        f"$p='{public}'; $q='{private}'; "
        "Get-NetIPAddress -InterfaceAlias $q -AddressFamily IPv4 -EA 0 | "
        "? PrefixOrigin -eq Manual | Remove-NetIPAddress -Confirm:$false; "
        "Set-NetIPInterface -InterfaceAlias $q -Dhcp Enabled -EA 0; "
        "$n=New-Object -ComObject HNetCfg.HNetShare; "
        "foreach($c in $n.EnumEveryConnection){"
        "$g=$n.INetSharingConfigurationForINetConnection($c); "
        "if($g.SharingEnabled){$g.DisableSharing()}}; "
        "foreach($c in $n.EnumEveryConnection){"
        "$d=$n.NetConnectionProps($c); "
        "$g=$n.INetSharingConfigurationForINetConnection($c); "
        "if($d.Name -eq $p){$g.EnableSharing(0)}; "
        "if($d.Name -eq $q){$g.EnableSharing(1)}}; "
        "New-ItemProperty -Path "
        "'HKLM:\\SYSTEM\\CurrentControlSet\\Services\\SharedAccess\\Parameters' "
        "-Name EnableRebootPersistConnection -Value 1 -PropertyType DWord -Force | Out-Null; "
        "Set-Service SharedAccess -StartupType Automatic; "
        "Write-Host 'internet sharing is on'"
    )


def sharing_command(public: str, private: str) -> str:
    """The same thing as one line to paste into an Administrator terminal."""
    body = sharing_body(public, private).replace('"', '`"')
    return f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{body}"'


BOARD_CHECKS = (
    ("an address on the shared network", "ip -4 addr show scope global | grep -c 'inet '"),
    ("a route out", "ip route | grep -c '^default'"),
    ("the PC answers", f"ping -c1 -W2 {ICS_HOST} >/dev/null 2>&1 && echo 1 || echo 0"),
    ("the internet answers", "ping -c1 -W2 8.8.8.8 >/dev/null 2>&1 && echo 1 || echo 0"),
    ("names resolve", "ping -c1 -W2 pypi.org >/dev/null 2>&1 && echo 1 || echo 0"),
)


def check_board(host: str) -> list[tuple[str, bool]]:
    """Run the ladder on the board itself. Each step isolates one layer.

    Ordered so the first failure names the layer: no address means DHCP, no
    route means the share is not routing, 8.8.8.8 failing means NAT, and only
    names failing means the DNS proxy.
    """
    from .devkit import require

    script = "; ".join(f"echo -n '{label}:'; {command}" for label, command in BOARD_CHECKS)
    result = subprocess.run(  # noqa: S603
        [require("ssh"), "-o", "ConnectTimeout=8", host, script],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False, timeout=90,
    )
    if result.returncode != 0:
        raise ConnectionError(result.stderr.strip() or "could not reach the board")

    outcomes = []
    for line in result.stdout.splitlines():
        label, _, value = line.partition(":")
        if label.strip():
            outcomes.append((label.strip(), value.strip() not in ("", "0")))
    return outcomes


def run_setup_network(host: str | None, apply: bool) -> int:
    """Report what the sharing setup looks like, and optionally fix it.

    Returns 0 when the board can reach the internet or, with no board to ask,
    when sharing is confirmed on. Anything less is 1, because "probably fine"
    is what wastes the afternoon.
    """
    report = probe()
    print("Internet sharing, PC -> DevKit\n")

    if report.error:
        print(f"  could not inspect this machine: {report.error}")
        return 1
    if not report.adapters:
        print("  no network adapters are up. Plug something in first.")
        return 1

    print("  adapters:")
    for adapter in report.adapters:
        role = "the internet" if adapter.has_internet else "no way out"
        role_now = report.shared.get(adapter.name)
        share = f"  [shared: {role_now}]" if role_now else ""
        print(f"    {adapter.name:<22} {adapter.ip or '(no address)':<16} {role}{share}")

    internet, link = report.internet, report.board_link
    print()

    if internet is None:
        print("  WARNING  no adapter has a route to the internet.")
        print("           Connect this PC to Wi-Fi or a network first; there is")
        print("           nothing to share yet.")
        return 1

    if link is None:
        print("  WARNING  no second network found to share with.")
        print("           Nothing here looks like a cable to a DevKit: every")
        print("           adapter that is up already has its own way out.")
        print("           Plug the board into this PC's Ethernet port, power it")
        print("           on, and run this again.")
        return 1

    print(f"  internet comes in on:  {internet.name}  ({internet.ip})")
    print(f"  the DevKit is on:      {link.name}  ({link.ip or 'no address yet'})")
    print()

    if report.platform != "windows":
        print("  Setting this up automatically is Windows-only for now. On this")
        print("  system, share it with NAT and a DHCP server:")
        print("    sudo sysctl -w net.ipv4.ip_forward=1")
        print(f"    sudo iptables -t nat -A POSTROUTING -o {internet.name} -j MASQUERADE")
        print(f"    sudo iptables -A FORWARD -i {link.name} -o {internet.name} -j ACCEPT")
        print("  macOS: System Settings > General > Sharing > Internet Sharing.")
        return 1

    if report.sharing_known:
        public = [n for n, role in report.shared.items() if role == "public"]
        private = [n for n, role in report.shared.items() if role == "private"]
        if public and private:
            print(f"  sharing is ON: {public[0]} -> {private[0]}")
            if link.name not in private:
                print(f"  WARNING  but it shares with {private[0]}, not {link.name},")
                print("           which is where the board is. Re-run with --apply.")
        else:
            print("  sharing is OFF. The board cannot reach anything through this PC.")
    else:
        print("  sharing state: needs Administrator to read. Re-run an elevated")
        print("  terminal to have this checked properly.")

    if apply:
        print()
        if report.elevated:
            print("  applying...")
            result = subprocess.run(  # noqa: S603
                [shutil.which("powershell") or "powershell", "-NoProfile",
                 "-Command", sharing_body(internet.name, link.name)],
                check=False,
            )
            if result.returncode != 0:
                print("  that did not work. Run this by hand:")
                print(f"\n  {sharing_command(internet.name, link.name)}")
                return 1
            print("  done. On the board: sudo dhclient -v eth0")
        else:
            print("  --apply needs Administrator, and this terminal is not.")
            print("  Right-click Start > Terminal (Admin), then paste:\n")
            print(f"  {sharing_command(internet.name, link.name)}")
            return 1

    if host:
        print(f"\n  asking the board itself ({host}):")
        try:
            outcomes = check_board(host)
        except ConnectionError as exc:
            print(f"    cannot reach it over SSH: {exc}")
            print("    That is expected while sharing is off, because the board")
            print("    has no address yet. Fix the PC side first.")
            return 1
        for label, ok in outcomes:
            print(f"    {'yes' if ok else 'NO ':<4} {label}")
        if all(ok for _, ok in outcomes):
            print("\n  The board is online. Nothing else to do.")
            return 0
        first_bad = next(label for label, ok in outcomes if not ok)
        print(f"\n  First thing that fails: {first_bad}.")
        if first_bad.startswith("an address"):
            print("  The board never got one. On the board: sudo dhclient -v eth0")
        elif first_bad.startswith("the internet"):
            print("  It reaches this PC but goes no further: sharing is not")
            print("  forwarding traffic. Re-run this with --apply.")
        elif first_bad.startswith("names"):
            print("  Everything routes; only DNS is missing. On the board, put")
            print(f"  'nameserver {ICS_HOST}' in /etc/resolv.conf.")
        return 1

    print("\n  Pass --host sima@<devkit-ip> to have the board checked too.")
    return 0
