<div align="center">

<img src="assets/sima-devkit-docs-logo-home.png" alt="sima-vision: live YOLO computer vision on a SiMa Modalix DevKit 3.0" width="640">

<br>

[![SiMa.ai](https://img.shields.io/badge/SiMa.ai-Modalix_DevKit_3.0-E63946?style=for-the-badge)](https://sima.ai)
[![Palette SDK](https://img.shields.io/badge/Palette_SDK-2.1.2-457B9D?style=for-the-badge)](https://docs.sima.ai)
[![Neat](https://img.shields.io/badge/Neat-0.3.0-2A9D8F?style=for-the-badge)](https://docs.sima.ai)

[![CI](https://github.com/RizwanMunawar/sima-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/RizwanMunawar/sima-projects/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/badge/pip_install-sima--vision-3775A9?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/sima-vision/)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-6C757D?style=flat-square)](LICENSE)
[![YOLO26](https://img.shields.io/badge/Ultralytics-YOLO26-FFB703?style=flat-square&labelColor=333)](https://github.com/ultralytics/ultralytics)

<h3>Live YOLO26 on the MLA of a SiMa.ai Modalix DevKit 3.0.<br>Three apps, one pipeline, one command.</h3>

</div>

```bash
pip install sima-vision
sima-vision detect          # sample clip and model fetched for you, then run
```

<div align="center">

| <img src="assets/preview-detect.png" width="270"> | <img src="assets/preview-segment.png" width="270"> | <img src="assets/preview-fall.png" width="270"> |
|:--:|:--:|:--:|
| **`detect`**<br><sub>boxes, labels, confidence</sub> | **`segment`**<br><sub>masks and a background blur</sub> | **`fall`**<br><sub>tracking, falls, email alerts</sub> |

<sub>Drawn on a laptop by <code>sima-vision preview</code>. No board, no model.</sub>

</div>

> **Inference needs the board. Nothing else does.** Checking a config and seeing exactly
> what the overlay will look like both run on your laptop, so you can set the whole thing
> up before you own any hardware.

<br>

## Contents

| | |
|:--|:--|
| [Install](#install) | pip, and what each extra buys you |
| [Quickstart](#quickstart) | Without a board, then with one |
| [The three tasks](#the-three-tasks) | What each does, and its own flags |
| [Commands](#commands) | Every subcommand in one table |
| [Settings](#settings) | Flags, Python keywords, `config.yaml` |
| [Python API](#python-api) | The same three verbs, importable |
| [Driving the board from your PC](#driving-the-board-from-your-pc) | `push`, `remote`, `pull` |
| [Set up a new DevKit](#set-up-a-new-devkit) | One time, about two hours |
| [Troubleshooting](#troubleshooting) | Symptom to fix, in one table |
| [How it works](#how-it-works) | The pipeline, and why it is shaped that way |

<br>

## Install

Python 3.10 or later. No compiler, and the only dependency is PyYAML.

```bash
pip install sima-vision
```

| Where | Install | Why |
|:--|:--|:--|
| **On the DevKit** | `pip install sima-vision` | The board already provides numpy and OpenCV |
| **Your laptop** | `pip install "sima-vision[preview]"` | Adds numpy and OpenCV so `preview` can draw |
| **Contributing** | `pip install -e ".[dev,preview]"` | Also ruff and pytest |

> [!CAUTION]
> **On the board, never let pip pull numpy 2.x.** `pyneat` and every `simaai-*` package
> need `numpy<2`. `sima-vision` depends on neither numpy nor OpenCV precisely so that
> installing it cannot upgrade them. If something already broke it:
> `pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"`

Check what you have:

```bash
sima-vision doctor
```

<br>

## Quickstart

### Without a board

```bash
sima-vision preview --task segment -o blur.png    # draw the overlay your config makes
sima-vision init segment                          # write a documented config.yaml
sima-vision segment --validate                    # check it
```

`preview` runs **no model**. It draws synthetic detections through the real overlay code,
so what you are judging is styling, not accuracy. Nothing here touches the network.

### On the DevKit

```bash
pip install sima-vision
sima-cli login                # once, the model packs need a community.sima.ai account
sima-vision detect            # that is the whole thing
```

No arguments. Each task has a default sample clip and model archive, and the first run
puts both in `./assets/`: the clip comes straight from a public GitHub release, the model
through `sima-cli`, which holds your login. Every run after that reuses them.

Out comes `detections.mp4` and a `frames/` directory. Bring them back with
[`sima-vision pull`](#driving-the-board-from-your-pc).

Your own clip or model, as a path or an `https` URL:

```bash
sima-vision detect --source my-clip.h264 --model my-model.tar.gz
sima-vision detect --source https://example.com/my-clip.h264
```

A URL is downloaded into `assets/` once and reused. There is **one** `assets/` for all
three tasks, not one per task: they share the clips, and `detect` and `fall` share the
model archive outright.

> [!IMPORTANT]
> **Video must be raw H.264, not `.mp4`.** The board decodes H.264 in hardware, and
> containers hit a [demuxer bug](#known-issues) in Neat 0.3.0. Convert once, losslessly:
>
> ```bash
> ffmpeg -i clip.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 clip.h264
> ```
>
> Renaming an `.mp4` does not work and is caught at startup, with that command in the
> error. Leave `source.fps`, `source.width` and `source.height` at `0`: the real geometry
> is read out of the stream's SPS.

> **New board?** Bring-up is a one-time job of about two hours, mostly downloading. That
> is [Set up a new DevKit](#set-up-a-new-devkit). Nothing above needs it.

<br>

## The three tasks

| Task | What it does | Model | Output |
|:--|:--|:--|:--|
| **`detect`** | Boxes, class names and confidence on every frame | YOLO26 detect | `detections.mp4` |
| **`segment`** | Per-pixel masks, and a background blur that keeps the subject sharp | YOLO26 segment | `segmentation.mp4` |
| **`fall`** | Tracks people and emails when one of them goes down | YOLO26 detect | `falls.mp4`, `alerts/` |

All three share one pipeline: the same source handling, Neat graph, sample decoding,
drawing and sinks. A task supplies only what is genuinely its own.

<details>
<summary><b>detect</b> &nbsp;&middot;&nbsp; boxes, labels and confidence</summary>

<br>

<img src="assets/sima-devkit-docs-logo-object-detection.png" alt="Object detection on a Modalix DevKit 3.0" width="560">

The simplest thing that proves the whole chain works. No task-specific flags: everything
it understands is in [Settings](#settings).

```bash
sima-vision detect
sima-vision detect --conf 0.5 --frames 200
sima-vision detect --source rtsp://cam/live --source-type rtsp
```

| Problem | Fix |
|:--|:--|
| No detections at all | Check `model.family` is `yolo26`, then lower `--conf` |
| Scores all near zero | `model.family` does not match the archive, so a raw-logit head is being decoded wrong |
| Boxes in the wrong place | Leave `resize.mode: letterbox` and `pad_value: 114`. Do not add your own maths |

</details>

<details>
<summary><b>segment</b> &nbsp;&middot;&nbsp; masks, and a background blur</summary>

<br>

<img src="assets/sima-devkit-docs-logo-instance-segmentation.png" alt="Instance segmentation on a Modalix DevKit 3.0" width="560">

Needs a `-seg` model pack. A plain detect head carries no mask data, and the run says so
rather than guessing.

```bash
sima-vision segment                                        # masks and the default blur
sima-vision segment --blur-strength 81                     # a stronger blur
sima-vision segment --blur-method pixelate                 # pixelate instead
sima-vision segment --keep-classes person                  # only people stay sharp
sima-vision segment --anonymise --keep-classes person      # the other way round
sima-vision segment --no-blur                              # masks, no background effect
```

| Flag | What it does |
|:--|:--|
| `--blur` / `--no-blur` | Whether the background is treated at all |
| `--blur-method` | `gaussian`, `pixelate` or `none` |
| `--blur-strength PX` | Gaussian kernel width for a 1080p frame. Default 41 |
| `--keep-classes` | Class names or ids that stay sharp. Default: everything detected |
| `--anonymise` | Blur the instances instead of the background |
| `--mask-threshold T` | Mask cut-off as a probability. Default 0.5; lower grows instances |
| `--no-masks` | Blur around plain boxes. Works with a detect head |
| `--minimal` | Pull frames and do nothing else. See below |

`--minimal` is the diagnostic. If a run that stalls part-way through a clip completes
with `--minimal`, the cause is how much work the app does per frame. If it stalls at the
same frame, the cause is the graph.

| Problem | Fix |
|:--|:--|
| `model.family ... is a detect head` | Point `--model` at a `-seg` pack, or pass `--no-masks` |
| Whole frame blurred | Nothing was detected. Lower `--conf` |
| Mask edges jagged | Raise `blur.feather`, then `segmentation.blur_mask` |
| Masks in the wrong place | Pin `segmentation.space` to `net` or `box` instead of `source` |
| Too slow at 1080p | `blur.downscale: 4` is the single biggest win, then `--save-every 30` |

</details>

<details>
<summary><b>fall</b> &nbsp;&middot;&nbsp; tracking, a fall state machine, SMTP alerts</summary>

<br>

<img src="assets/sima-devkit-docs-logo-fall-detection.png" alt="Fall detection on a Modalix DevKit 3.0" width="560">

Tracks people across frames and watches three signals from the plain bounding box: the
box turning wide-and-short, its height collapsing, and how fast its centre is dropping. A
fall has to hold for `--confirm` seconds before an alert fires, which is what keeps
someone crouching from setting it off.

```bash
sima-vision fall                                       # track and judge, no email
sima-vision fall --no-fall                             # track only, to tune tracking first
sima-vision fall --confirm 0.8                         # confirm faster
sima-vision fall --alert-to ops@example.com --send     # actually email
sima-vision fall --test-alert                          # one fake alert now, needs no board
```

| Flag | What it does |
|:--|:--|
| `--classes` | Class names or ids that can fall. Default `person` |
| `--confirm S` | How long a fall signal must hold. Default 1.5 |
| `--no-fall` | Track without judging |
| `--alert-to EMAIL` | Recipients. Implies `--alerts` |
| `--alert-from EMAIL` | From address |
| `--alerts` | Enable alerts, still a dry run |
| `--send` | Actually connect to the SMTP server |
| `--smtp-host` / `--smtp-port` / `--smtp-user` | Server, 587 for STARTTLS or 465 for SSL, and the login |
| `--site NAME` | Human name for this camera, used in the subject |
| `--test-alert` | Send one fake alert and exit |

> [!WARNING]
> Alerts stay a **dry run** until `--send`. The SMTP password is only ever read from
> `$FALL_ALERT_SMTP_PASSWORD`, never from a config file, because config files get
> committed.

```bash
export FALL_ALERT_SMTP_PASSWORD='your-app-password'    # macOS, Linux, the DevKit
```

```powershell
$env:FALL_ALERT_SMTP_PASSWORD = "your-app-password"    # PowerShell
```

Gmail needs an [app password](https://support.google.com/accounts/answer/185833), not
your account password. `--test-alert` proves the whole path synchronously and needs no
board.

| Problem | Fix |
|:--|:--|
| `falls=0` on footage that has one | Lower `--confirm` and `fall.aspect_ratio`; first check the person is tracked at all with `--no-fall` |
| Alerts fire constantly | Raise `--confirm` first, then `alerts.cooldown_seconds` |
| Track ids change every few frames | Lower `tracking.iou_threshold`, raise `tracking.max_age` |
| Gmail rejects the login | App password, not the account password. `--test-alert` prints the SMTP error verbatim |
| `ssl and starttls are both on` | 465 uses `ssl`, 587 uses `starttls`. Never both |

</details>

<br>

## Commands

| Command | Board? | What it does |
|:--|:--:|:--|
| `sima-vision detect` | **yes** | Run detection on the MLA |
| `sima-vision segment` | **yes** | Run segmentation, with the optional blur |
| `sima-vision fall` | **yes** | Run fall detection, with SMTP alerts |
| `sima-vision preview` | no | Draw the overlay your config produces, to a PNG |
| `sima-vision init` | no | Write a documented `config.yaml` here |
| `sima-vision doctor` | no | What is installed, and what it lets you do |
| `sima-vision fetch` | no | Download the sample clips up front |
| `sima-vision push` | no | Copy files to the DevKit |
| `sima-vision pull` | no | Copy results back |
| `sima-vision remote` | no | Run a task on the DevKit over SSH |

Add `--validate` to any task to parse and check a config, print what it resolved to, and
exit. It loads neither pyneat nor the model, so it runs anywhere:

```bash
sima-vision segment --conf 0.5 --blur-strength 81 --validate
```

```
config OK: config.yaml
  model: assets/models/yolo26m-seg-bf16-mla_tess-b1.tar.gz
  family=yolo26-seg -> BoxDecodeType.YoloV26Seg
  source: type=video uri=assets/videos/people-walking-outside-mall.h264
  decode: conf=0.5 iou=0.6 max_det=50
  segmentation: masks=on source=auto space=auto threshold=0.5
  blur: background | method=gaussian kernel=81 sigma=auto down=2 feather=9
  output: video=segmentation.mp4 stills=frames/ every=10
```

`sima-vision <command> --help` lists every flag.

<br>

## Settings

Three layers, each beating the one above it:

```
built-in defaults   ->   config.yaml   ->   flags / Python keywords
```

So **everything is optional**. With no file and no flags you get this task's sample clip
and model. For a setup you keep:

```bash
sima-vision init detect     # a documented config.yaml, right here
sima-vision detect          # picks up ./config.yaml on its own
```

Every setting has a flag and a Python keyword under the same name. Both write the same
config key, and both go through the same validation.

### What people actually change

| I want | Flag | Python | Config key |
|:--|:--|:--|:--|
| Fewer spurious boxes | `--conf 0.5` | `conf=0.5` | `decode.score_threshold` |
| To catch more, noisily | `--conf 0.15` | `conf=0.15` | `decode.score_threshold` |
| A short test run | `--frames 100` | `frames=100` | `runtime.frames` |
| No video file | `--no-video` | `video=False` | `output.video.enable` |
| No stills | `--no-save` | `save=False` | `output.save.enable` |
| Fewer stills | `--save-every 30` | `save_every=30` | `output.save.every` |
| To see where the time goes | `--profile` | `profile=True` | `runtime.profile` |
| A live Insight view | `--insight` | `insight=True` | `output.insight.enable` |
| A stronger blur | `--blur-strength 81` | `blur_strength=81` | `blur.kernel` |
| A pixelated background | `--blur-method pixelate` | `blur_method="pixelate"` | `blur.method` |
| Only people kept sharp | `--keep-classes person` | `keep_classes=["person"]` | `blur.keep_classes` |
| People blurred, scene sharp | `--anonymise` | `anonymise=True` | `blur.invert` |
| Masks, but no blur | `--no-blur` | `blur=False` | `blur.enable` |
| To track something else | `--classes forklift` | `classes=["forklift"]` | `tracking.classes` |
| Falls confirmed faster | `--confirm 0.8` | `confirm=0.8` | `fall.confirm_seconds` |
| An email on a fall | `--alert-to me@x.com --send` | `alert_to=[...], send=True` | `alerts.*` |

### Shared flags

| Flag | Config key | What it does |
|:--|:--|:--|
| `--source`, `-s` | `source.uri` | File, `https` URL, RTSP URL, or nothing for the sample clip |
| `--source-type` | `source.type` | `video`, `rtsp` or `usb` |
| `--fps` / `--width` / `--height` | `source.*` | Leave at 0. Read from the stream |
| `--model`, `-m` | `model.path` | Compiled model archive, or an `https` URL to one |
| `--labels` | `model.labels` | Class names. Defaults to the packaged COCO list |
| `--family` | `model.family` | Detection head. Must match the model |
| `--conf` / `--iou` / `--max-det` | `decode.*` | Confidence, NMS IoU, top-K |
| `--frames`, `-n` | `runtime.frames` | Stop after N frames |
| `--timeout` / `--queue-depth` | `runtime.*` | Pull timeout, and how far ahead of the sinks to run |
| `--profile` | `runtime.profile` | Per-stage timings |
| `--video-path` / `--no-video` | `output.video.*` | The annotated recording |
| `--save-dir` / `--save-every` / `--no-save` | `output.save.*` | Annotated stills |
| `--no-hud` | `output.video.hud` | Leave the frame-rate badge off |
| `--insight` / `--insight-host` | `output.insight.*` | The live Neat Insight feed |
| `--config`, `-c` / `--no-config` | | Which config file, or none |
| `--validate` | | Check and print, then exit |

### Cameras and streams

```bash
sima-vision detect --source-type usb                            # the DevKit camera
sima-vision detect --source rtsp://cam/live --source-type rtsp
```

For a live source, raise `--queue-depth` and leave `runtime.overflow_policy` on `auto`.
For a file, `auto` resolves to `block`, which keeps every frame so the recording matches
the input length.

### Environment

| Variable | What it does |
|:--|:--|
| `SIMA_VISION_ASSETS` | Where clips and models are downloaded. Default `./assets` |
| `SIMA_VISION_DEVKIT` | The board, as `user@address`, so `push`, `pull` and `remote` stop asking |
| `FALL_ALERT_SMTP_PASSWORD` | The only place the SMTP password is ever read from |

<br>

## Python API

```python
from sima_vision import run, preview, validate

# No board: draw the overlay a setting produces, and write a PNG.
preview("segment", out="blur.png", blur_strength=81, keep_classes=["person"])

# No board: resolve and check a config, then look at what it became.
cfg = validate("detect", conf=0.5, max_det=20)
print(cfg.score_threshold, cfg.video_path)

# On the DevKit: run it. Every argument is optional, exactly like the CLI.
run("detect")
run("detect", source="clip.h264", model="det.tar.gz", conf=0.5, frames=200)
```

Every keyword is derived from the CLI's own flags, so `--blur-strength 81` and
`blur_strength=81` cannot drift apart. Anything the keywords do not cover is still
reachable by its config path:

```python
run("segment", **{"runtime.output_buffers": 2})
```

<br>

## Driving the board from your PC

Three wrappers around `ssh` and `scp`, so the awkward parts stop being yours.

```bash
sima-vision push my-clip.h264               # PC -> board
sima-vision remote -- detect --frames 200   # run it there, watch it here
sima-vision pull                            # board -> PC
```

Say the address once:

```bash
export SIMA_VISION_DEVKIT=sima@192.168.137.50     # macOS, Linux
```

```powershell
$env:SIMA_VISION_DEVKIT = "sima@192.168.137.50"   # PowerShell
```

or pass `--host` to any of the three. Authentication is `ssh`'s own business: an agent, a
key, or it asks you. Nothing here handles or stores a password.

| Command | Notes |
|:--|:--|
| `sima-vision push FILE...` | Folders are copied whole. `--dest` changes where they land, default `~/` |
| `sima-vision pull` | With no names, takes whatever a run of any task left: the video, `frames/`, `alerts/`, `config.yaml`. `--into` chooses where they land here |
| `sima-vision pull detections.mp4` | Or name exactly what you want |
| `sima-vision remote -- ARGS` | Everything after `--` runs as `sima-vision ARGS` on the board |

Three things these get right that a hand-written `scp` usually does not:

1. **On Windows, `scp D:\clips\a.h264 sima@ip:~` fails** with `could not resolve hostname
   d:`, because `scp` reads everything before the first colon as a host. `push` never
   passes a full local path: it groups files by folder and runs `scp` from inside each
   one. `pull` does the same at the other end.
2. **`ssh host cmd` without a pty means Ctrl-C never reaches the task.** It keeps running,
   keeps the MLA, and your next launch fails with a busy device. `remote` always passes
   `-tt`.
3. **`pull` with no arguments cannot be one `scp`**, because `scp` fails the whole
   transfer on a name that is not there and the outputs depend on which task ran. So the
   names are listed over `ssh` first and only what exists is fetched.

They need an OpenSSH client, which macOS and Linux ship and Windows 10/11 has under
Settings > Apps > Optional features > OpenSSH Client.

<br>

## Set up a new DevKit

<details>
<summary><b>One time, about two hours, mostly downloading. Skip this if your board already runs pyneat.</b></summary>

<br>

Written on Windows with WSL2, which is the path SiMa's own tooling expects. Every warning
below marks somewhere real time was lost.

```
   WINDOWS PC          WSL2 / UBUNTU              MODALIX DEVKIT 3.0
   ----------          -------------              ------------------
1  cable up      ----------------------------->   DHCP address
2  wsl --install ---->  Ubuntu ready
3  .wslconfig    ---->  WSL takes .137.1    --->  reachable both ways
                  4    sima-cli in a venv
                  5    docker + nfs
                  6    sdk setup           --->   pyneat on the board
```

Step 3 is load-bearing. Step 6 installs onto the board **over the network** and fails
silently if networking is not fixed first, which is the usual way to lose an afternoon.

### 1. Cable up

USB (serial console) plus Ethernet straight to your PC. Open the
[serial tool](https://docs.sima.ai/_static/tools/serial/index.html) and set the DevKit to
**DHCP**.

```powershell
arp -a | Select-String "192.168.137"     # find the board
ping <devkit-ip>
```

Must reply. Nothing else works until it does. The board's address changes between reboots;
your PC keeps `192.168.137.1`.

### 2. WSL2

```powershell
wsl --install -d Ubuntu      # PowerShell as Administrator
wsl -l -v                    # want: Ubuntu, Running, 2
```

### 3. Mirrored networking

WSL sits behind NAT by default and **cannot see your DevKit**.

```powershell
@"
[wsl2]
networkingMode=mirrored
"@ | Set-Content -Path "$env:USERPROFILE\.wslconfig" -Encoding utf8

wsl --shutdown
```

Wait ten seconds, open a WSL terminal, then verify **both** of these:

```powershell
wsl -- hostname -I                  # must list 192.168.137.1
wsl -- ping -c 2 <devkit-ip>        # must reply
```

If they fail, check `.wslconfig` was not saved as `.wslconfig.txt`.

### 4. sima-cli

Become root **first**. `sudo su -` is a login shell, so it drops you in `/root`.

```bash
sudo su -
apt update && apt install -y git python3-venv python3-pip
python3 -m venv sima
source sima/bin/activate
pip install sima-cli
sima-cli login                  # needs a community.sima.ai account
```

### 5. Docker and NFS

The Neat SDK **is** a Docker container. No Docker, no SDK.

```bash
sudo apt update && sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo apt install -y nfs-kernel-server nfs-common
```

Docker needs systemd to survive a WSL restart:

```bash
grep -q 'systemd=true' /etc/wsl.conf 2>/dev/null || sudo tee -a /etc/wsl.conf <<'EOF'

[boot]
systemd=true
EOF
```

Then `wsl --shutdown` in PowerShell, reopen WSL, and confirm:

```bash
sudo systemctl enable --now docker
sudo docker run hello-world      # must print "Hello from Docker!"
```

### 6. The Neat SDK

```bash
sudo su -
source sima/bin/activate
sima-cli install ghcr:sima-neat/sdk
sima-cli sdk setup --devkit <devkit-ip>
```

**Answer every prompt.** The ones that matter:

| Prompt | Answer |
|:--|:--|
| `Some system checks failed. Continue?` | `y`. The Firewall row says *Unverified*, not failed |
| `Install Model Compiler extension?` | `Y`. Adds 9 GB, only needed to compile your own models |
| `Install VSCode Extensions?` | `y` lowercase. A bare Enter is rejected |
| `Apply passwordless sudo on DevKit?` | `y`. Required for workspace sync |
| everything else | `Y` or Enter |

`mount.nfs: Connection timed out` is fine; setup falls back to rsync and carries on.

Then confirm the **board half** actually happened, because that is the part that fails
quietly:

```bash
ssh sima@<devkit-ip> "~/pyneat/bin/python3 -c 'import pyneat; print(pyneat.__version__)'"
```

A version means you are done. `No such file or directory` means pairing never installed
it, almost always because networking was not fixed first. Re-run
`sima-cli sdk setup --devkit <devkit-ip>` from WSL now that it works.

### Five rules that prevent most problems

| # | Rule | Because |
|:--|:--|:--|
| 1 | Networking before pairing | Pairing installs over the network. No route means a silent no-op |
| 2 | Docker before the SDK | The SDK **is** a container |
| 3 | `cd` after `sudo su -` | `-` is a login shell, so it drops you in `/root` |
| 4 | Raw `.h264`, never `.mp4` | Containers hit a demuxer bug in Neat 0.3.0 |
| 5 | Never leave the only copy on the board | A firmware update wipes its home directory |

### Firmware version mismatch

```
ERROR: DevKit/SDK version mismatch. DevKit 2.0.0, SDK 2.1.2
```

New boards often ship older firmware. eLxr cannot be updated remotely, so this runs **on
the board**, which already has internet over your Ethernet cable:

```bash
ssh sima@<devkit-ip>
sima-cli login
sima-cli update            # menu, then "Update all packages to the latest"
```

Budget 15 to 40 minutes plus a reboot, then re-run `sima-cli sdk setup`. If SSH complains
the host key changed, that is expected: `ssh-keygen -R <devkit-ip>`.

### Setup errors

| Symptom | Fix |
|:--|:--|
| `sima-cli: command not found` | The venv is not active: `sudo su -`, then `source sima/bin/activate` |
| Venv landed in `/root/sima` | You ran `cd` before `sudo su -` |
| `externally-managed-environment` | Create the venv first |
| `Error: No such command 'sdk'` | You ran it on the board. `sdk` is PC-side |
| WSL cannot ping the DevKit | `.wslconfig` missing, saved as `.txt`, or WSL not restarted |
| `Cannot connect to the Docker daemon` | `sudo systemctl start docker` |
| Docker dead after every restart | systemd not enabled in `/etc/wsl.conf` |
| `ssh: Could not resolve hostname d:` | A Windows path went to `scp`, which read `D:` as a host. Use `sima-vision push` |
| Copy hangs | The board's IP changed. Find it again with `arp -a` |
| `DevKit/SDK version mismatch` | Firmware recovery, above |

</details>

<br>

## Troubleshooting

<details>
<summary><b>Symptom to fix, for a running task.</b></summary>

<br>

| Symptom | Fix |
|:--|:--|
| `ModuleNotFoundError: pyneat` | You are on your PC, or pairing never ran. Inference only happens on the board |
| `pyneat requires numpy<2` | `pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"` |
| `model archive not found` | `sima-cli login`, then run again. It fetches the pack itself |
| `sima-cli download did not produce ...` | Not logged in. `sima-cli login`, or pass `--model` with a path or URL |
| `source file not found` | The error lists what is actually in the folder. Paths are relative to where you launch |
| `is not a raw H.264 elementary stream` | You renamed an `.mp4` instead of converting it. The error carries the ffmpeg command |
| `No src-element named "nN_demux"` | The `.mp4` demuxer bug. Convert to `.h264` |
| Device busy | An orphaned run still holds the MLA: `sima-vision remote -- doctor` first, then `ssh sima@<devkit-ip> pkill -f sima-vision` |
| Stuck after `loading model` | The first load unpacks the archive. Give it a minute |
| First run seems to hang before anything prints | That is the 13 MB clip and the model downloading. It only happens once |
| `processed=0` and a 20 second timeout | The source caps are not negotiating. Leave `--fps`, `--width` and `--height` at 0 |
| Output video shorter than the input, plays fast | Frames are being dropped. Set `runtime.overflow_policy: auto` |
| Recording only a few frames long | Usually `output.insight`. Its encoder shares the codec daemon with the decoder |
| `timed out waiting for instances` after a few frames | The graph starved the decoder's buffer pool. Lower `runtime.output_buffers`, and use `--minimal` to tell "too slow" from "wrong graph" |
| Dropped frames on a live source | Raise `--queue-depth`, keep `overflow_policy: auto` |
| `has unknown class` | A typo. The error suggests near matches from the labels file |

### Known issues

**`groups.video_input` cannot play `.mp4`, Neat 0.3.0.**

```
gst_parse_launch failed: No src-element named "n1_demux" - omitting link
```

`VideoTrackSelect` builds its fragment from one variable, so what it emits is correct. The
graph then appends an instance suffix, but the renamer only rewrites `name=<x>`
declarations, so the pad reference is never fixed. Any non-empty suffix breaks it, and
reordering does not help.

The fix is no container, so no demuxer. `sima-vision` detects `.h264`, `.264`, `.avc` and
`.bin` and builds the chain by hand:

```
FileInput -> H264Parse -> Queue -> SimaDecode -> CapsRaw
```

A container input still goes through `groups.video_input` and prints the conversion
command.

</details>

<br>

## How it works

<details>
<summary><b>The pipeline, and why it is shaped that way.</b></summary>

<br>

The pipeline is a Neat `Graph`, not a single `Model.run`, because it has several stages,
named public endpoints and a branch with a fan-in:

```
source --> branch --> frame ---------------+
              |                            +--> combine("<task>_output")
              +----> model --> results ----+

<task>_output --> parse --> overlay --> video file + stills
                        |-> MetadataSender  (JSON over UDP)
                        +-> VideoSender     (H.264 RTP over UDP)
```

Frames come off the hardware decoder, whose buffer pool is small: the boot log prints
`BufferNum=8`. Everything expensive therefore happens on a sink thread, so the pull loop
hands each buffer back immediately. Holding one across a `pull()` is what deadlocks the
decoder part-way through a clip, and it looks exactly like the app being slow.

Boxes arrive as one UInt8 tensor tagged `BBOX`, packed as a `uint32` count followed by
24-byte records of `x, y, w, h, score, class_id` in source-image pixels. A segment head
packs its masks into the tail of that same buffer.

```
sima_vision/
  cli.py        the command line          api.py      the Python API
  config.py     loading and validation    scene.py    the preview scene
  assets.py     clips and model archives  devkit.py   push, pull, remote
  media.py      H.264 and geometry        neat.py     graph assembly
  samples.py    decoding a sample         masks.py    masks and compositing
  draw.py       the overlay               sinks.py    video, stills, Insight
  runloop.py    the pull loop
  tasks/        detect.py   segment.py   fall.py
```

`assets.py` and `devkit.py` are the only modules that reach the network, and `assets.py`
only from a run. A `--validate` or a `preview` resolves the same paths and fetches
nothing.

</details>

<br>

## Contributing

```bash
git clone https://github.com/RizwanMunawar/sima-projects.git
cd sima-projects
pip install -e ".[dev,preview]"

ruff check sima_vision tests
pytest -q
```

The tests need no board. Mask decoding, compositing, the overlay, the tracker and the fall
rules are plain numpy and OpenCV, so they run anywhere, and the `ssh` and `scp` wrappers
are tested against a fake subprocess. CI covers Python 3.10 to 3.13 on Linux, macOS and
Windows, builds the wheel and installs it clean.

<br>

## License

The models used here for testing are **Ultralytics YOLO26**, under **AGPL-3.0**. All other
parts of this repository are under **Apache-2.0**. See [LICENSE](LICENSE).

## Credits

- [SiMa.ai](https://github.com/SiMa-ai) for Modalix, the Palette SDK and Neat
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO26 models

<div align="center">

<br>

Created with love by **Muhammad Rizwan Munawar**, passionate about implementing computer
vision ideas and sharing my gains with the community.

If this saved you an afternoon, **star the repo** and pass it on to someone else bringing
up a DevKit.

<br>

<a href="https://github.com/RizwanMunawar"><img src="assets/socials/github.svg" width="50" alt="GitHub"></a>
&nbsp;&nbsp;
<a href="https://www.linkedin.com/in/muhammadrizwanmunawar/"><img src="assets/socials/linkedin.svg" width="50" alt="LinkedIn"></a>
&nbsp;&nbsp;
<a href="https://x.com/muhammdrizwanmr"><img src="assets/socials/x.svg" width="50" alt="X"></a>
&nbsp;&nbsp;
<a href="https://www.youtube.com/@muhammadrizwanmunawar"><img src="assets/socials/youtube.svg" width="50" alt="YouTube"></a>
&nbsp;&nbsp;
<a href="https://muhammadrizwanmunawar.medium.com/"><img src="assets/socials/medium.svg" width="50" alt="Medium"></a>

</div>
