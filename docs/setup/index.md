# Setup

Eight one-time steps, from a bare Windows PC to a paired DevKit with a model on it.
**Follow them in order.** Several steps fail silently if done too early.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│      WINDOWS PC        │       WSL2 · UBUNTU        │     MODALIX DEVKIT      │
├────────────────────────┼────────────────────────────┼─────────────────────────┤
├─────────────────────────────── ONE-TIME SETUP ────────────────────────────────┤
│                        │                            │                         │
│ 1  cable up  ══════════╪════════════════════════════╪═══►  DHCP address       │
│      USB + Ethernet    │                            │      board powers on    │
│                        │                            │                         │
│ 2  wsl --install ══════╪═══►  Ubuntu ready          │                         │
│                        │                            │                         │
│ 3  .wslconfig  ════════╪═══►  WSL takes .137.1  ════╪═══►  now reachable      │
│      mirrored mode     │        same subnet         │      both directions    │
│                        │                            │                         │
│ 4  firewall rules ═════╪═══►  UDP 9000/9100 open    │                         │
│      Hyper-V inbound   │        ready to receive    │                         │
│                        │                            │                         │
│                        │ 5  git clone + sima-cli    │                         │
│                        │      repo + venv           │                         │
│                        │                            │                         │
│                        │ 6  docker + nfs            │                         │
│                        │      the SDK is a container│                         │
│                        │                            │                         │
│                        │ 7  sima-cli sdk setup ═════╪═►  pyneat + runtime     │
│                        │      12.6 GB image         │      installed on board │
│                        │                            │                         │
│                        │ 8  download model          │                         │
│                        │      yolo26m .tar.gz       │                         │
│                        │                            │                         │
├────────────────────────────── EVERY RUN, REPEAT ──────────────────────────────┤
│                        │                            │                         │
│                        │ 9  scp object-detection/ ══╪═►  ~/object-detection   │
│                        │      edit → copy → run     │      python3 src/app.py │
│                        │                            │                         │
│10  browser  ◄══════════╪════  Insight  ◄════════════╪═══╡ VideoSender         │
│      localhost:9900    │        live, while running │      MetadataSender     │
│                        │                            │                         │
│10  scp  ◄══════════════╪════════════════════════════╪═══╡ VideoWriter         │
│      keeps a copy      │        detections.mp4      │      every frame        │
│                        │                            │                         │
└────────────────────────┴────────────────────────────┴─────────────────────────┘

```

| Step | Runs on | Time |
| :-- | :-- | :-- |
| [1. Connect the DevKit](hardware.md) | PowerShell | 15 min |
| [2. Install WSL2](wsl.md#2-install-wsl2) | PowerShell | 10 min |
| [3. Mirrored networking](wsl.md#3-mirrored-networking) | PowerShell | 5 min |
| [4. Firewall](wsl.md#4-firewall) | PowerShell | 2 min |
| [5. Get the code and install sima-cli](sdk.md#5-get-the-code-and-install-sima-cli) | WSL | 5 min |
| [6. Docker Engine and NFS](sdk.md#6-docker-engine-and-nfs) | WSL | 10 min |
| [7. Install the Neat SDK](sdk.md#7-install-the-neat-sdk) | WSL | 30 to 60 min |
| [8. Download a model](model.md) | SDK container | 5 min |

!!! danger "Steps 3 and 4 are load-bearing"

    Step 7 installs software onto the board **over the network**. Without mirrored
    networking and the firewall rules, the PC half succeeds and the board half
    silently does nothing. You find out an hour later, when
    `source ~/pyneat/bin/activate` reports "not found".

## Requirements

| Resource | Minimum |
| :-- | :-- |
| OS | Windows 11 with WSL2, Ubuntu 22.04/24.04, or macOS 15.5+ Apple Silicon |
| CPU | 4 cores |
| Memory | 16 GB |
| Free disk | 100 GB, of which the SDK image alone is 12.6 GB |
| Account | [community.sima.ai](https://community.sima.ai), needed to download models |
