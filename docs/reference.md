# Reference

## Addresses

| What | Value | Notes |
| :-- | :-- | :-- |
| DevKit | `<devkit-ip>`, user `sima` | DHCP, changes between reboots |
| Your PC, as the board sees it | `192.168.137.1` | Fixed by ICS, use this for `insight.host` |
| Insight web view | `https://localhost:9900` | Not `192.168.137.1:9900` |

## Ports

| Port | Protocol | Purpose |
| :-- | :-- | :-- |
| `9000-9079` | UDP | Video to Insight, `9000 + channel` |
| `9100-9179` | UDP | Detection metadata, `9100 + channel` |
| `9900` | TCP | Insight web UI |
| `8081` | TCP | Insight video UI |
| `8554` | TCP | RTSP |
| `8022` | TCP | Web SSH |
| `9999`, `10000` | TCP | VS Code browser UI |
| `40000-40199` | UDP | WebRTC |

## Paths

| What | Where | Machine |
| :-- | :-- | :-- |
| Repo | `/root/sima-projects` | WSL |
| Repo, from Windows | `\\wsl$\Ubuntu\root\sima-projects` | Windows |
| Shared workspace | `/workspace` | SDK container |
| Shared workspace | `/root/workspace` | WSL |
| SDK container name | `ghcr.io-sima-neat-sdk-latest` | WSL |
| PyNeat venv | `~/pyneat` | DevKit |
| App | `~/object-detection` | DevKit |
| Output video | `~/object-detection/detections.mp4` | DevKit |
| Annotated stills | `~/object-detection/sandbox/frames` | DevKit |
| Board install directory | `/media/nvme/neat` | DevKit |
| Playbooks | `/neat-resources/apps-src/skills/` | SDK container |
| Neat source | `/neat-resources/core-src/` | SDK container |

## Commands worth remembering

```bash
neat                          # SDK: component versions and exposed ports
neat update                   # SDK: update Neat components
dk shell                      # SDK: SSH into the DevKit
dk status                     # SDK: sync method and remote path
sudo docker ps                # WSL: is the SDK container up
sudo systemctl status docker  # WSL: is the daemon healthy
sima-cli device discover      # find DevKits on the network
cat /etc/buildinfo            # DevKit: firmware version
```

## Versions this was built against

| Component | Version |
| :-- | :-- |
| Palette SDK | 2.1.2 |
| Neat core, PyNeat, runtime, GStreamer plugins | 0.3.0 |
| Neat Insight | 0.0.6 |
| DevKit eLxr | 2.1.2 |
| sima-cli | 2.1.15 |
| Model Compiler | 2.1.0 |

## Six rules that prevent most problems

| # | Rule | Because |
| :-- | :-- | :-- |
| 1 | Networking before pairing | Pairing installs over the network. No route means a silent no-op |
| 2 | Docker before the SDK | The SDK **is** a container |
| 3 | `cd` after `sudo su -` | `-` is a login shell, so it drops you in `/root` |
| 4 | `insight.host = 192.168.137.1` | `127.0.0.1` means the board itself |
| 5 | Raw `.h264`, never `.mp4` | Containers hit a demuxer bug in Neat 0.3.0 |
| 6 | Always `ssh -tt` | Ctrl-C needs a pty to reach the app and release the MLA |
