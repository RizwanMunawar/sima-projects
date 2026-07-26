# WSL and networking

Steps 2 to 4. The last two are the ones that quietly ruin an afternoon if skipped.

## 2. Install WSL2

```powershell title="PowerShell as Administrator"
wsl --install -d Ubuntu
```

Reboot if prompted. On first launch Ubuntu asks you to create a username and password.
Write the password down, `sudo` will want it repeatedly.

```powershell title="PowerShell"
wsl -l -v
wsl --version
```

!!! success "Exit criteria"

    `Ubuntu / Running / 2`, and WSL **2.0.0 or newer**. The next step needs mirrored
    networking, which older builds do not support.

---

## 3. Mirrored networking

!!! danger "This is the step that decides whether the rest works"

By default WSL sits behind NAT on its own private network. It reaches the internet
fine, which makes everything feel healthy, but it has **no route at all** to the
`192.168.137.x` subnet your DevKit lives on.

```
┌─────── BEFORE · default NAT ───────┐   ┌───────── AFTER · mirrored ─────────┐
│                                    │   │                                    │
│   WSL      172.22.41.196           │   │   WSL      192.168.137.1           │
│              │                     │   │              │                     │
│              │   X  no route       │   │              │   =  same subnet    │
│              v                     │   │              v                     │
│   DevKit   <devkit-ip>             │   │   DevKit   <devkit-ip>             │
│                                    │   │                                    │
└────────────────────────────────────┘   └────────────────────────────────────┘
```

The reason this bites so hard is timing. In [step 7](sdk.md#7-install-the-neat-sdk),
`sima-cli sdk setup --devkit` does two jobs: it configures the SDK on your PC **and**
installs the Neat Library onto the board over the network. Without a route, the PC half
succeeds and reports success while the board half quietly does nothing. You discover it
much later, with no obvious connection back to a networking decision made an hour
earlier.

```powershell title="PowerShell"
@"
[wsl2]
networkingMode=mirrored
"@ | Set-Content -Path "$env:USERPROFILE\.wslconfig" -Encoding utf8

wsl --shutdown
```

Wait about 10 seconds, open a WSL terminal so it boots, then verify:

```powershell title="PowerShell"
wsl -- hostname -I                  # must list 192.168.137.1
wsl -- ping -c 2 <devkit-ip>        # must reply
```

!!! success "Exit criteria"

    **Both must pass.** This is the one place worth being stubborn.

!!! tip

    Created the file in Notepad? Check it is not silently saved as `.wslconfig.txt`.
    In the save dialog set **Save as type → All Files**.

---

## 4. Firewall

Mirrored networking has a side effect: it places WSL behind the Hyper-V firewall, which
blocks all inbound traffic by default. Your DevKit pushing video into WSL **is** inbound
traffic.

```
   DevKit                    Hyper-V firewall              Neat Insight
   ──────                    ────────────────              ────────────
   UDP 9000  ──────────────>  [ BLOCKED ]  - - - - - - >   (nothing)
   UDP 9100  ──────────────>  [ BLOCKED ]  - - - - - - >   (nothing)
```

What makes this one nasty is the failure mode. Insight loads perfectly in your browser,
shows a normal interface, and simply displays nothing. No error, no log line, no clue.

```powershell title="PowerShell as Administrator"
New-NetFirewallHyperVRule -Name "NeatVideo" -DisplayName "Neat Insight video" `
  -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
  -Protocol UDP -LocalPorts 9000-9079 -Action Allow

New-NetFirewallHyperVRule -Name "NeatMeta" -DisplayName "Neat Insight metadata" `
  -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
  -Protocol UDP -LocalPorts 9100-9179 -Action Allow

Get-NetFirewallHyperVRule | Where-Object DisplayName -match 'Neat'
```

Two narrow rules rather than flipping the firewall default to Allow. They open only the
port ranges Insight actually uses.

!!! success "Exit criteria"

    Both rules appear in the listing.

!!! tip

    These rules live in Windows and are tied to the WSL VM creator ID, not to the
    distro. They survive `wsl --unregister`, so if you ever rebuild WSL you can skip
    this step.

??? note "Want the LAN address to work as well?"

    Useful for viewing Insight from a phone or another machine on the subnet.

    ```powershell title="PowerShell as Administrator"
    New-NetFirewallHyperVRule -Name "NeatInsightWebUI" -DisplayName "Neat Insight web UI TCP" `
      -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
      -Protocol TCP -LocalPorts 9900,8081,9999,10000,8022,8554 -Action Allow
    ```

    Covers the web UI, video UI, both Code UI ports, web SSH and RTSP. It widens what
    is reachable on your network, so add it deliberately.

---

Next: [sima-cli, Docker and the SDK](sdk.md)
