# Troubleshooting

## Setup and sima-cli

| Symptom | Fix |
| :-- | :-- |
| `sima-cli: command not found` | Venv not active. `sudo su -`, `cd sima-projects`, `source sima/bin/activate` |
| Venv landed in `/root/sima` | You ran `cd` before `sudo su -`. Delete it and redo [step 5](../setup/sdk.md#5-get-the-code-and-install-sima-cli) |
| `externally-managed-environment` | Installing into system Python. Create the venv first |
| `python3 -m venv` fails | `sudo apt install -y python3-venv` |
| `Error: No such command 'sdk'` | You ran it on the board. `sdk` is PC-side only |
| WSL cannot ping the DevKit | `.wslconfig` missing, saved as `.txt`, or WSL not restarted. [Step 3](../setup/wsl.md#3-mirrored-networking) |

## Docker

| Symptom | Fix |
| :-- | :-- |
| `Cannot connect to the Docker daemon` | `sudo systemctl start docker`, or `sudo service docker start` without systemd |
| Docker dead after every WSL restart | systemd not enabled in `/etc/wsl.conf`. [Step 6](../setup/sdk.md#enable-systemd-so-docker-survives-a-restart) |
| `sima-cli install ghcr:...` fails instantly | Docker missing or stopped |
| `docker: permission denied` | Run as root, or `sudo usermod -aG docker $USER` then restart WSL |
| SDK container gone after reboot | Expected. `sima-cli sdk neat` |

## DevKit and firmware

| Symptom | Fix |
| :-- | :-- |
| `DevKit/SDK version mismatch` | Board firmware older than the SDK. [Recovery](recovery.md#devkit-firmware-version-mismatch) |
| `ELXR does not support remote update` | Firmware cannot be pushed from the PC. SSH in and run `sima-cli update` there |
| `No ELXR update was applied` | That is what a dry run does. Re-run without `--dryrun` |
| Version unchanged after updating | Either only the dry run ran, or the APT channel lacks that version |
| `[sudo] password for sima: Sorry, try again` | Same password you SSH in with |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | Expected after a reflash. `ssh-keygen -R <devkit-ip>` |
| `Current directory '/media/nvme' is not writable` | `sima-cli` downloads into the cwd. Create a folder you own first |
| `source ~/pyneat/bin/activate` not found | Neat Library never installed on the board. [Recovery](recovery.md#pyneat-missing-on-the-devkit) |
| `neat: command not found` on the board | Same cause |
| `ModuleNotFoundError: pyneat` | Either you are on the PC, or pairing never ran |
| Device busy at startup | Orphaned run. `ssh sima@<devkit-ip> pkill -f src/app.py` |
| DevKit disk full | You installed outside `/media/nvme` |

## Copying files

| Symptom | Fix |
| :-- | :-- |
| `ssh: Could not resolve hostname d:` | Windows path used inside Linux. `scp` read `D:` as a hostname |
| `scp: Connection closed` | Usually follows the above. The source path did not exist |
| `model archive not found` | Launch from `~/object-detection`, and check `find assets -type f` |
| `failed to open source` | Same, for the video |
| Copy hangs | The IP changed. `arp -a \| Select-String "192.168.137"` |

## Running

| Symptom | Fix |
| :-- | :-- |
| `pyneat requires numpy<2` | `pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"` |
| `No src-element named "nN_demux"` | `.mp4` demuxer bug. Convert to `.h264`. [Known issues](known-issues.md) |
| Stuck after `loading model` | First load unpacks the archive. Give it a minute |
| `processed=0` and a 20 s timeout | You set `overflow_policy: block`. Every stage applies backpressure, so forbidding drops deadlocks the graph. Use `auto` |
| Output video far shorter than the input | The run stalled on backpressure. Keep `overflow_policy: auto`, and set `output.insight.enable: false` to rule out the preview feed |

## Detections and display

| Symptom | Fix |
| :-- | :-- |
| Insight loads, no video | **Firewall.** [Step 4](../setup/wsl.md#4-firewall) was skipped. Most common failure by far |
| Insight blank, no errors | `insight.host` is `127.0.0.1`. Use `192.168.137.1` |
| Insight blank but `sent=N failures=0` | The stream already ended. UDP buffers nothing, so open the viewer **first** and use a looped source |
| `192.168.137.1:9900` will not load | Expected. Use `https://localhost:9900` |
| No detections at all | `model.family` mismatch, then lower `decode.score_threshold` |
| Boxes in the wrong place | `resize.mode: letterbox`, `pad_value: 114`. Do not add your own correction maths |
| Scores all near zero | Head mismatch. YOLOX, v6 and v26 use raw logit heads |
| Video never decodes | Not H.264. Check with `ffprobe` |
| Dropped frames on live sources | Raise `runtime.queue_depth`, keep `overflow_policy: auto` |
| Running slowly | `runtime.profile: true` for per-stage timings |
