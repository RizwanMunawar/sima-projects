<div align="center">

<img src="assets/sima-devkit-docs-logo.png" alt="SiMa Neat SDK: live YOLO object detection on a Modalix DevKit 3.0" width="640">

<br>

[![SiMa.ai](https://img.shields.io/badge/SiMa.ai-Modalix_DevKit_3.0-E63946?style=for-the-badge)](https://sima.ai)
[![Palette SDK](https://img.shields.io/badge/Palette_SDK-2.1.2-457B9D?style=for-the-badge)](https://docs.sima.ai)
[![Neat](https://img.shields.io/badge/Neat-0.3.0-2A9D8F?style=for-the-badge)](https://docs.sima.ai)

![Windows](https://img.shields.io/badge/Windows_11-0078D6?style=flat-square&logo=windows11&logoColor=white)
![WSL2](https://img.shields.io/badge/WSL2-E95420?style=flat-square&logo=ubuntu&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-26_·_11_·_v8_·_v5-FFB703?style=flat-square&labelColor=333)

![Setup](https://img.shields.io/badge/setup-~2h-6C757D?style=flat-square)
![Download](https://img.shields.io/badge/download-12.6_GB-DC3545?style=flat-square)

</div>

## What this is

A **setup guide** and a **working detector app**, both written while actually bringing
up a Modalix DevKit 3.0. Every warning marks somewhere real time was lost.

```
   ┌──────────────┐        ┌──────────────────┐        ┌───────────────────┐
   │  WINDOWS PC  │        │   WSL2 · UBUNTU  │        │  MODALIX DEVKIT   │
   ├──────────────┤        ├──────────────────┤        ├───────────────────┤
   │  play .mp4   │<──────>│  sima-cli        │<──────>│  MLA              │
   │  serial      │  scp   │  Docker + SDK    │ ssh /  │  your app         │
   │              │        │                  │  scp   │  detections.mp4   │
   └──────────────┘        └──────────────────┘        └───────────────────┘
        review               build + deploy                 inference
```

Your app runs **on the DevKit**. It writes an annotated `.mp4` and annotated stills on
the board; you copy them back and look at them.

```bash
git clone https://github.com/RizwanMunawar/sima-projects.git
cd sima-projects
```

---

## Contents

Setup runs once per machine, about two hours and mostly downloading. Deploying and
reviewing the result is the loop you live in after that.

| Section | What it covers |
|:--|:--|
| [Test it in three commands](#test-it-in-three-commands) | Push, run, pull the result back |
| [Complete workflow](#complete-workflow) | Who does what, and in which order |
| [Cable up the DevKit](#step-1) | USB serial and Ethernet, set DHCP. 15 min |
| [Install WSL2](#step-2) | `wsl --install -d Ubuntu`. 10 min |
| [Mirrored networking](#step-3) | So WSL can see the board at all. 5 min |
| [Get the code and install sima-cli](#step-4) | Clone, venv, login. 5 min |
| [Docker Engine + NFS](#step-5) | The Neat SDK is a container. 10 min |
| [Install the Neat SDK](#step-6) | Pairs the board, 12.6 GB. 30 to 60 min |
| [Download a model](#step-7) | A YOLO26 `.tar.gz` pack. 5 min |
| [Deploy and run](#step-8) | `scp` the app over and run it. 2 min |
| [See the result](#step-9) | Pull `detections.mp4` and `frames/` back. 1 min |
| [The overlay](#the-overlay) | Every box, caption and FPS badge setting |
| [Configuration](#configuration) | `config.yaml`, and the mistakes it catches |
| [Daily loop](#daily-loop) | The commands you repeat after every edit |
| [How the app works](#how-the-app-works) | Pipeline, preprocessing, decode types |
| [Known issues](#known-issues) | The `.mp4` demuxer bug in Neat 0.3.0 |
| [Reference](#reference) | Addresses, paths, five rules |
| [Questions people ask](#questions-people-ask) | FAQ: own models, cameras, short runs, where output lands |
| [Common errors](#common-errors) | One table, symptom to fix |
| [Recovery](#recovery) | Firmware mismatch, missing pyneat |
| [License](#license) | YOLO26 under AGPL-3.0, everything else Apache-2.0 |
| [Credits](#credits) | SiMa.ai, Ultralytics, and where to find me |

## Test it in three commands

**Board already paired?** This is the whole loop. Run it from the repo root in WSL:

```bash
scp -r object-detection/ sima@<devkit-ip>:~                         # 1. push the app

ssh -tt sima@<devkit-ip> \
  'source ~/pyneat/bin/activate && cd ~/object-detection && python3 src/app.py'

scp sima@<devkit-ip>:~/object-detection/detections.mp4 .            # 3. pull the result
```

Play `detections.mp4`. Boxes on it means the whole chain works. Repeat after every edit.

**Faster still, before you touch the board.** This parses and validates `config.yaml`
without pyneat, a model, or any hardware, and runs anywhere:

```bash
python3 object-detection/src/app.py --validate-config
```
```
config OK: object-detection/config.yaml
  family=yolo26 -> BoxDecodeType.YoloV26
  preprocess: kind=image enable=on in=NV12 ... resize=letterbox pad=114 | normalize=coco_yolo
```

**A short run instead of a whole clip.** Set `runtime.frames: 100` and
`output.video.enable: false`. You get 100 annotated JPEGs in `frames/` and the app exits
by itself.

Starting from a bare machine? Work through the steps below in order.

---

## Complete workflow

```
┌───────────────────────────────────────────────────────────────────────────────┐
│      WINDOWS PC        │       WSL2 · UBUNTU        │   MODALIX DEVKIT 3.0    │
├────────────────────────┼────────────────────────────┼─────────────────────────┤
├─────────────────────────────── ONE-TIME SETUP ────────────────────────────────┤
│                        │                            │                         │
│ 1  cable up  ──────────┼────────────────────────────┼───>  DHCP address       │
│      USB + Ethernet    │                            │      board powers on    │
│                        │                            │                         │
│ 2  wsl --install ──────┼───>  Ubuntu ready          │                         │
│                        │                            │                         │
│ 3  .wslconfig  ────────┼───>  WSL takes .137.1  ────┼───>  now reachable      │
│      mirrored mode     │        same subnet         │      both directions    │
│                        │                            │                         │
│                        │ 4  git clone + sima-cli    │                         │
│                        │      repo + venv           │                         │
│                        │                            │                         │
│                        │ 5  docker + nfs            │                         │
│                        │      the SDK is a container│                         │
│                        │                            │                         │
│                        │ 6  sima-cli sdk setup ─────┼─>  pyneat + runtime     │
│                        │      12.6 GB image         │      installed on board │
│                        │                            │                         │
│                        │ 7  download model          │                         │
│                        │      yolo26m .tar.gz       │                         │
│                        │                            │                         │
├────────────────────────────── EVERY RUN, REPEAT ──────────────────────────────┤
│                        │                            │                         │
│                        │ 8  scp object-detection/ ──┼─>  ~/object-detection   │
│                        │      edit, copy, run       │      python3 src/app.py │
│                        │                            │                         │
│ 9  scp back  <─────────┼────────────────────────────┼────  VideoWriter        │
│      keeps a copy      │      detections.mp4        │      every frame        │
│                        │                            │                         │
└────────────────────────┴────────────────────────────┴─────────────────────────┘
```

Step 3 is load-bearing. Step 6 installs onto the board over the network and fails
silently if networking is not fixed first, which is the usual way to lose an afternoon.

<a id="step-1"></a>
### 1. Cable up the DevKit

USB cable (serial console) + Ethernet straight to your PC. Open the
[serial tool](https://docs.sima.ai/_static/tools/serial/index.html), set the DevKit to
**DHCP**.

```powershell
arp -a | Select-String "192.168.137"     # find the board
ping <devkit-ip>
```

> ✅ Must reply. Nothing else works until it does.

> [!IMPORTANT]
> **`<devkit-ip>` appears throughout this guide. Substitute your own.** The board gets
> its address by DHCP, so it changes between reboots: mine has been both
> `192.168.137.123` and `192.168.137.193`. Your PC keeps `192.168.137.1`, which is why
> that one is written out in full.

<a id="step-2"></a>
### 2. Install WSL2

```powershell
wsl --install -d Ubuntu      # PowerShell as Administrator
wsl -l -v                    # want: Ubuntu · Running · 2
```

<a id="step-3"></a>
### 3. Mirrored networking

WSL sits behind NAT by default and **cannot see your DevKit**. Later, pairing installs
software onto the board over the network. With no route, the PC half succeeds and the
board half silently does nothing. You find out an hour later.

```powershell
@"
[wsl2]
networkingMode=mirrored
"@ | Set-Content -Path "$env:USERPROFILE\.wslconfig" -Encoding utf8

wsl --shutdown
```

Wait 10 seconds, open a WSL terminal, then verify:

```powershell
wsl -- hostname -I                  # must list 192.168.137.1
wsl -- ping -c 2 <devkit-ip>        # must reply
```

> ✅ **Both must pass.** If they do not, check `.wslconfig` was not saved as
> `.wslconfig.txt`, then see [Common errors](#common-errors).

<a id="step-4"></a>
### 4. Get the code and install sima-cli

Become root **first**. `sudo su -` is a login shell, so it drops you in `/root`.
Cloning after that puts the repo at `/root/sima-projects`, which is why every later
step can just say `cd sima-projects`.

```bash
# WSL
sudo su -
apt update && apt install -y git python3-venv python3-pip

git clone https://github.com/RizwanMunawar/sima-projects.git
cd sima-projects

python3 -m venv sima
source sima/bin/activate
pip install sima-cli
sima-cli login                  # needs a community.sima.ai account
```

You now have the app in [`object-detection/`](object-detection/) and the `sima` venv
beside it. **Every command below runs from this directory.**

<a id="step-5"></a>
### 5. Docker Engine + NFS

The Neat SDK **is** a Docker container. No Docker, no SDK.

```bash
# WSL, from docs.docker.com/engine/install/ubuntu
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

```powershell
wsl --shutdown          # PowerShell, then reopen WSL
```

```bash
sudo systemctl enable --now docker
sudo docker run hello-world
```

> ✅ Must print **"Hello from Docker!"** If not, see
> [Common errors](#common-errors).

<a id="step-6"></a>
### 6. Install the Neat SDK

```bash
sudo su -
cd sima-projects
source sima/bin/activate
sima-cli install ghcr:sima-neat/sdk
sima-cli sdk setup --devkit <devkit-ip>
```

**Answer every prompt.** The ones that matter:

| Prompt | Answer |
|:--|:--|
| `Some system checks failed. Continue?` | `y`. The Firewall row says *Unverified*, not failed |
| `Install Model Compiler extension?` | `Y`. Adds **9 GB**, only needed to compile your own models |
| `Install VSCode Extensions?` | `y` lowercase. A bare Enter is rejected |
| `Apply passwordless sudo on DevKit?` | `y`. Required for workspace sync |
| everything else | `Y` / Enter |

Then confirm the **board half** actually happened, because it is what fails quietly:

```bash
ssh sima@<devkit-ip> "~/pyneat/bin/python3 -c 'import pyneat; print(pyneat.__version__)'"
```

> ✅ Prints a version → done.
> ❌ `No such file or directory` → see [pyneat missing on the DevKit](#pyneat-missing).

> [!NOTE]
> **`mount.nfs: Connection timed out` is fine.** Setup falls back to rsync and carries
> on. The DevKit IP also changes between reboots (DHCP), so if things hang, check with
> `arp -a | Select-String "192.168.137"`.

<a id="step-7"></a>
### 7. Download a model

```bash
sima-cli sdk neat        # WSL, starts the container and drops you inside

# in the container
sima-cli login
mkdir -p /workspace/assets/models && cd /workspace/assets/models
MODELS=https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection
sima-cli download $MODELS/yolo26m-det-bf16-mla_tess-b1.tar.gz
```

Swap `yolo26m` for `n`, `s`, `l` or `x` to trade speed against accuracy.

<a id="step-8"></a>
### 8. Deploy and run

The app lives in [`object-detection/`](object-detection/):

```
object-detection/
├── config.yaml          # every setting lives here
├── assets/
│   ├── models/          # .tar.gz model packs  (not in git)
│   └── videos/          # .h264 streams        (not in git)
└── src/
    ├── app.py           # the pipeline
    ├── coco_labels.txt  # 80 COCO class names
    └── requirements.txt
```

Models and video are gitignored, so after cloning you supply your own: a model from
step 7 in `assets/models/`, and a `.h264` stream in `assets/videos/`.

On the DevKit, once per board:

```bash
pip install -r ~/object-detection/src/requirements.txt
```

> [!CAUTION]
> **Never let pip pull numpy 2.x.** `pyneat` and every `simaai-*` package need
> `numpy<2`. The pins in `requirements.txt` handle it. If you already broke it:
> `pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"`

**One command copies everything.** Run it from the repo root in WSL, after every
change:

```bash
scp -r object-detection/ sima@<devkit-ip>:~
```

Then on the DevKit:

```bash
ssh -tt sima@<devkit-ip>                     # two t's, see below
source ~/pyneat/bin/activate
cd ~/object-detection && python3 src/app.py
```

Healthy output:

```
source: type=video uri=assets/videos/video-1.h264 stream=1920x1080@24
preprocess: ... resize=letterbox pad=114 normalize=coco_yolo
model: ... family=yolo26 decode_type=YoloV26 labels=80
runtime: preset=reliable overflow=block queue_depth=3
graph built
save: dir=frames every=1 overlay=True
video: detections.mp4 codec=mp4v fps=24 hud=True
running. press Ctrl-C to stop.
[50] 24.8 fps, 6.2 detections/frame avg
```

> [!CAUTION]
> **Use `ssh -tt`, two t's.** Without a pty, Ctrl-C never reaches the app. It keeps
> running invisibly holding the MLA and your next run fails.
> Rescue: `ssh sima@<devkit-ip> pkill -f src/app.py`

Anything other than that output (no detections, a stall, a crash) is in
[Common errors](#common-errors) and [Questions people ask](#questions-people-ask).

<a id="step-9"></a>
### 9. See the result

Every run writes an annotated video and annotated stills on the board. Pull them across:

```bash
scp sima@<devkit-ip>:~/object-detection/detections.mp4 .
scp -r sima@<devkit-ip>:~/object-detection/frames .
```

On exit the app says exactly what it wrote, so a short or empty file is obvious:

```
processed=3012 timeouts=0
video: wrote 3012 frames to detections.mp4 (48.3 MB)
```

It looks like this:

<div align="center">

<!-- ─────────────────────────────────────────────────────────────────────────
     Embed the demo video here.
     GitHub accepts a drag-and-dropped .mp4 directly in the README editor,
     or paste the user-images.githubusercontent.com URL it generates.
     ───────────────────────────────────────────────────────────────────── -->

https://github.com/user-attachments/assets/REPLACE-WITH-YOUR-VIDEO


</div>

## The overlay

```
   ┌───────────────────────────────────────────────────────────┐
   │ ┌──────────┐                                              │
   │ │ FPS: 24.7│                                              │
   │ └──────────┘   ┌────────────┐                             │
   │                │ person 94% │                             │
   │                ├────────────┴─────────┐                   │
   │                │                      │                   │
   │                │           •          │   centre marker   │
   │                │                      │                   │
   │                └──────────────────────┘                   │
   └───────────────────────────────────────────────────────────┘
```

A rectangle in the class colour, a centre dot, and a filled caption above it carrying
the class name and confidence. Captions flip inside the box rather than clipping off the
top of the frame. Colours come from a 20-entry palette keyed to class id, so a class is
always the same colour.

Every part of it is config, under `visualization:` in `config.yaml`. Sizes are pixels
**for a 1080p frame**; with `auto_scale: on` they are multiplied by
`frame height / reference_height`, so 4K does not get hairlines and 480p does not get
slabs. Colours are `[B, G, R]` because OpenCV is BGR, so `[0, 0, 255]` is red.

```yaml
visualization:
  box_thickness: 3
  text_scale: 1.0
  text_thickness: 2
  text_padding: 10

  centre_dot: on
  centre_dot_radius: 7

  show_labels: on
  show_scores: on
  score_decimals: 2

  text_color: [255, 255, 255]

  auto_scale: on
  reference_height: 1080

  hud:
    text_color: [104, 31, 17]
    bg_color: [255, 255, 255]
    text_scale: 2.0
    text_thickness: 5
    padding: 15
    min_width: 0
    min_height: 0
```

| Key | Effect |
|:--|:--|
| `box_thickness` | Detection rectangle outline weight |
| `text_scale` | Caption font size multiplier |
| `text_thickness` | Caption stroke weight |
| `text_padding` | Gap between the caption text and the edge of its band |
| `centre_dot`, `centre_dot_radius` | Filled dot at the centre of each box |
| `show_labels`, `show_scores` | Class name and confidence in the caption |
| `score_decimals` | `2` gives `person 0.57`, `0` gives `person 1` |
| `text_color` | Caption text colour |
| `auto_scale`, `reference_height` | Scale every size above with the frame height |

The FPS badge in the corner is drawn when `output.video.hud` is true, and styled by the
`hud:` block:

| Key | Effect |
|:--|:--|
| `hud.text_color`, `hud.bg_color` | Badge text and background |
| `hud.text_scale`, `hud.text_thickness` | Badge font size and stroke. `0` follows the caption settings |
| `hud.padding` | What actually sets the badge size. `0` follows `text_padding` |
| `hud.min_width`, `hud.min_height` | Floor for the badge box. `0` fits the text, and the text stays centred either way |

Where the output goes:

| Key | Effect |
|:--|:--|
| `output.video.path` | Where to write, relative to the launch directory |
| `output.video.codec` | 4-char FourCC. `mp4v` by default, auto-falls back to `MJPG`/`.avi` |
| `output.video.fps` | `0` matches the source rate |
| `output.video.hud` | Draw the FPS badge. Turn off for clean footage |
| `output.save.every` | Write every Nth frame as a JPEG. `0` disables |

## Configuration

Everything lives in `object-detection/config.yaml`. These settings matter:

```yaml
model:
  path: assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz
  family: yolo26                       # must match your model

source:
  type: video                          # video | rtsp | usb
  uri: assets/videos/video-1.h264      # DevKit path, relative to ~/object-detection

decode:
  score_threshold: 0.30                # lower it if detections are missing

visualization:                         # how the boxes and captions are drawn
  box_thickness: 3
  text_scale: 1.0
  centre_dot: on
  show_labels: on
  show_scores: on
  auto_scale: on                       # scale with the frame height
  hud:                                 # the FPS badge
    text_color: [104, 31, 17]
    bg_color: [255, 255, 255]

output:
  video:
    enable: true
    path: detections.mp4               # annotated video, written on the DevKit
    hud: true
  save:
    enable: true
    dir: frames                        # annotated stills
    every: 1                           # every Nth frame, 0 disables
```

Every drawing setting is tunable without touching the code, and takes effect on the
next run: see [The overlay](#the-overlay) for the full list, what each one does, and how
`auto_scale` keeps a 4K frame from getting hairlines.

| Mistake | What happens |
|:--|:--|
| `uri: C:\Users\...\video.mp4` | The DevKit has no `C:` drive |
| `uri: r"C:\path\file.mp4"` | `r"..."` is Python. YAML keeps the `r` and quotes |
| `family` mismatched | No detections, or every score near zero |
| A `.mp4` source | Hits a demuxer bug. Convert to `.h264`, see below |

### Video must be raw H.264

The DevKit decodes H.264 in hardware, and `.mp4` containers hit a
[known bug](#known-issues). Convert once, losslessly:

```powershell
ffmpeg -i video-4.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 video-4.h264
```

The app reads the real geometry out of the stream's SPS, so leave `source.fps`,
`source.width` and `source.height` at `0`.

## Daily loop

Start the SDK:

```bash
sudo su -
cd sima-projects
source sima/bin/activate
sima-cli sdk neat
```

Then **edit, copy, run, review**, and repeat.

| Task | Command | Run in |
|:--|:--|:--|
| Validate the config, no hardware | `python3 object-detection/src/app.py --validate-config` | anywhere |
| Start the SDK container | `sima-cli sdk neat` | WSL |
| Push the app to the board | `scp -r object-detection/ sima@<devkit-ip>:~` | WSL |
| Pull the video back | `scp sima@<devkit-ip>:~/object-detection/detections.mp4 .` | WSL |
| Pull the stills back | `scp -r sima@<devkit-ip>:~/object-detection/frames .` | WSL |
| SSH to the board | `dk shell` | SDK container |
| Check the sync method | `dk status` | SDK container |
| Component versions | `neat` | SDK container |
| Run the app | `python3 src/app.py` | DevKit |
| Kill an orphaned run | `pkill -f src/app.py` | DevKit |

## How the app works

<details>
<summary><b>Pipeline shape, preprocessing and decode types</b></summary>

Built by following the `neat-application-builder` playbook. Every pyneat call was
verified against the packaged core source in the SDK container, not written from
memory:

| What | Source of truth |
|:--|:--|
| `PreprocessOptions` fields | `include/model/PreprocessPlan.h` |
| `ModelOptions` fields | `include/model/Model.h` |
| `BoxDecodeType` members | `include/pipeline/BoxDecodeType.h` |
| Preproc semantics | `docs/reference/nodes/preproc.mdx` |
| BBOX wire payload | `docs/reference/boxdecode_decode_types.md` |
| Python enum names | `python/src/module.cpp` |
| Reference implementation | `apps-src/examples/object-detection/single-stream-object-detector` |

All relative to `/neat-resources/core-src/` inside the container.

### Pipeline

The playbook's decision map puts this on `Graph` rather than `Model.run(...)`, because
there are multiple stages, named public endpoints and a branch with a fan-in:

```
   source --> branch --> frame ----------------+
                   |                           +--> combine("detector_output")
                   +---> model --> detections -+
```

`detector_output` is pulled from the `Run` handle, the BBOX payload is parsed, and the
annotated frame fans out to the video writer and the still writer.

### Preprocessing

The `preprocess:` block is an **intent layer**, not a set of instructions. `Model`
resolves it against the archive's MPK contract and compiles the matching Preproc,
Quant, Tess or QuantTess graph. Anything left on `auto` is the planner's call.

| Config key | pyneat field | Notes |
|:--|:--|:--|
| `kind` | `preprocess.kind` | `image` for every source here |
| `enable` | `preprocess.enable` | Master switch |
| `input_format` | `color_convert.input_format` | **Must match what the source produces** |
| `output_format` | `color_convert.output_format` | `auto` takes it from the model |
| `input_max_*` | `input_max_width/height` | Buffer capacity. Defaults to 1920x1080 |
| `resize.mode` | `resize.mode` | `letterbox` preserves aspect and pads |
| `resize.width/height` | `resize.width/height` | `0` infers 640x640 for most YOLO |
| `pad_value` | `resize.pad_value` | `114`, the YOLO convention |
| `normalize.preset` | `preprocess.preset` | `coco_yolo` for every YOLO detector |
| `quantize`, `tessellate` | same | Leave on `auto` |

**The one that matters most is `input_format`.** Getting it wrong is the usual cause of
"the model runs but detects nothing":

| Source | `input_format` |
|:--|:--|
| `video`, `rtsp` (hardware H.264 decode) | `NV12` |
| `usb` (libcamera) | `NV12` |
| `cv2.imread` images | `BGR` |

**Coordinate mapping is automatic.** Preproc writes resize and letterbox metadata onto
the tensor, and BoxDecode reads it back, so boxes arrive in original-image pixels. Do
not undo the letterbox yourself. Shifted boxes mean `resize.mode` or `pad_value` is
wrong, not that inverse maths is missing.

### Model family to decode type

| `family` | `BoxDecodeType` |
|:--|:--|
| `yolo` | `Yolo` |
| `yolov5`, `yolov5-seg` | `YoloV5`, `YoloV5Seg` |
| `yolov6` | `YoloV6` |
| `yolov7`, `yolov7-seg` | `YoloV7`, `YoloV7Seg` |
| `yolov8`, `-seg`, `-pose` | `YoloV8`, `YoloV8Seg`, `YoloV8Pose` |
| `yolov9`, `yolov9-seg` | `YoloV9`, `YoloV9Seg` |
| `yolov10`, `yolov10-seg` | `YoloV10`, `YoloV10Seg` |
| **`yolo11`, `-seg`, `-pose`** | **`YoloV8`, `YoloV8Seg`, `YoloV8Pose`** |
| `yolo26`, `-seg`, `-pose` | `YoloV26`, `YoloV26Seg`, `YoloV26Pose` |
| `yolox` | `YoloX` |

> **YOLO11 has no `BoxDecodeType` of its own.** The enum goes v5, v6, v7, v8, v9, v10,
> v26, X. Ultralytics YOLO11 exports the same decoupled DFL detect head as YOLOv8, so
> `yolo11` maps to `YoloV8`. Verify against your own export: uniformly near-zero scores
> mean the head does not match the decode family.

YOLOX, v6 and v26 use raw logit heads. Do not decode them as probability-only YOLO
heads. `-seg` and `-pose` families decode, but this app renders only the leading boxes;
masks and keypoints need `decode_segmentation` or `decode_pose` in the frame handler.

### Tuning

| Symptom | Setting |
|:--|:--|
| Missing detections | Lower `decode.score_threshold` |
| Duplicate boxes | Lower `decode.nms_iou` |
| Too many detections | Raise `decode.score_threshold`, lower `max_detections` |
| Dropped frames | Raise `runtime.queue_depth` |
| Want timings | `runtime.profile: true` |

</details>

## Known issues

<details>
<summary><b><code>groups.video_input</code> cannot play <code>.mp4</code> · Neat 0.3.0</b></summary>

```
gst_parse_launch failed: No src-element named "n1_demux" - omitting link
```

`VideoTrackSelect` builds its fragment from one variable, so what it emits is correct:

```cpp
const std::string base = "n" + std::to_string(node_index) + "_demux";
ss << "qtdemux name=" << base << " " << base << ".video_" << idx_;
```

The graph then appends an instance suffix, but the renamer only rewrites `name=<x>`
declarations. `element_names()` reports just `{"n1_demux"}`, so the pad reference is
never fixed. **Any non-empty suffix breaks it**, so reordering does not help.

**Fix:** no container, no demuxer. `app.py` detects `.h264` / `.264` / `.avc` and
builds the chain by hand:

```
FileInput → H264Parse → Queue → SimaDecode → CapsRaw
```

Selected automatically by extension: `.h264`, `.264`, `.bin`, `.avc`. A container input
still uses `groups.video_input` and prints the conversion command.

</details>

## Reference

### Addresses

| What | Value | Notes |
|:--|:--|:--|
| DevKit | `<devkit-ip>`, user `sima` | DHCP, changes between reboots |
| Your PC, as the board sees it | `192.168.137.1` | Fixed by ICS, also the board's route to the internet |

### Paths

| What | Where | Machine |
|:--|:--|:--|
| Repo | `/root/sima-projects` | WSL |
| Repo, from Windows | `\\wsl$\Ubuntu\root\sima-projects` | Windows |
| Shared workspace | `/workspace` | SDK container |
| Shared workspace | `/root/workspace` | WSL |
| SDK container name | `ghcr.io-sima-neat-sdk-latest` | WSL |
| PyNeat venv | `~/pyneat` | DevKit |
| App | `~/object-detection` | DevKit |
| Output video | `~/object-detection/detections.mp4` | DevKit |
| Output stills | `~/object-detection/frames/` | DevKit |
| Playbooks | `/neat-resources/apps-src/skills/` | SDK container |
| Neat source | `/neat-resources/core-src/` | SDK container |

### Five rules that prevent most problems

| # | Rule | Because |
|:--|:--|:--|
| 1 | Networking before pairing | Pairing installs over the network. No route means a silent no-op |
| 2 | Docker before the SDK | The SDK **is** a container |
| 3 | `cd` after `sudo su -` | `-` is a login shell, so it drops you in `/root` |
| 4 | Raw `.h264`, never `.mp4` | Containers hit a demuxer bug in Neat 0.3.0 |
| 5 | Always `ssh -tt` | Ctrl-C needs a pty to reach the app and release the MLA |


## Tips, FAQ and fixes

The warnings inside each step are the ones that cost an afternoon if you skip them.
Everything else lives here.

### Questions people ask

<details>
<summary><b>Can I try anything without a DevKit?</b></summary>

Yes, the config half:

```bash
python3 object-detection/src/app.py --validate-config
```

It needs only `pyyaml`, runs on Windows or WSL, and checks the model family maps to a
real `BoxDecodeType`, that every path resolves, and what the preprocess plan will be.
Inference itself needs the board, because it runs on the MLA.

</details>

<details>
<summary><b>Can I run my own YOLO model?</b></summary>

Put the `.tar.gz` model pack in `object-detection/assets/models/`, then set two keys:

```yaml
model:
  path: assets/models/<your-pack>.tar.gz
  family: yolo11                       # the head your export actually has
```

`family` must match the export, not the name of the weights file. See
[model family to decode type](#model-family-to-decode-type); YOLO11 maps to the
`YoloV8` head. Uniformly near-zero scores mean the family is wrong.

</details>

<details>
<summary><b>How do I use a camera or an RTSP stream instead of a file?</b></summary>

```yaml
source:
  type: usb            # or rtsp
  uri: ""              # rtsp://... for rtsp, empty for the default camera
  rtsp:
    codec: h264
    tcp: true
    latency_ms: 100
  usb:
    camera_name: ""    # "" = DevKit default camera
```

Leave `preprocess.input_format: NV12`; all three sources produce NV12. Leave
`runtime.preset` and `runtime.overflow_policy` on `auto` too, which switches a live
source to keep-latest so it stays current instead of falling behind.

</details>

<details>
<summary><b>How do I make a test run short?</b></summary>

```yaml
runtime: { frames: 100 }
output:
  video: { enable: false }
  save:  { enable: true, every: 10 }
```

The app stops itself after 100 frames and leaves annotated JPEGs in `frames/`. Good for
checking a new model or clip without waiting out a whole video.

</details>

<details>
<summary><b>Where do the outputs go?</b></summary>

Onto the **board**, relative to wherever you launched `app.py`, which is normally
`~/object-detection`. So `detections.mp4` and `frames/` sit next to `config.yaml` on the
DevKit. Pull them back with `scp`; nothing is written on your PC by the run itself.

</details>

<details>
<summary><b>How do I restyle the boxes and captions?</b></summary>

Every part of the overlay is config, see [The overlay](#the-overlay). Sizes are tuned
for 1080p and scale with the frame when `auto_scale: on`, and colours are `[B, G, R]`
because OpenCV is BGR.

</details>

<details>
<summary><b>Do I have to re-run the setup after every code change?</b></summary>

No. Steps 1–7 are once per machine. After an edit it is `scp -r object-detection/` and
run again. You only re-pair (step 6) if the board's home directory is wiped, which a
firmware update can do.

</details>

<details>
<summary><b>The DevKit IP changed and nothing connects</b></summary>

Expected: the board takes its address by DHCP, so it moves between reboots.

```powershell
arp -a | Select-String "192.168.137"
```

If SSH then complains the host key changed, that is the same board on a recycled
address: `ssh-keygen -R <ip>`.

</details>

<details>
<summary><b>Why must the video be raw .h264 rather than .mp4?</b></summary>

A demuxer bug in Neat 0.3.0, written up under [Known issues](#known-issues). Convert
once, losslessly, and keep the `.h264`:

```powershell
ffmpeg -i clip.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 clip.h264
```

</details>

<details>
<summary><b>The recording is shorter than the input and plays too fast</b></summary>

Frames are being dropped, because inference is slower than decoding and the survivors
are still written at the source rate. Leave `runtime.overflow_policy: auto`, which picks
`block` for a file so every frame is kept. The run then takes longer than the clip.

</details>

<details>
<summary><b>Can I just use numpy 2?</b></summary>

No. `pyneat` and every `simaai-*` package on the board need `numpy<2`, which is why
`requirements.txt` pins it. If pip has already upgraded you:
`pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"`

</details>

### Common errors

| Symptom | Fix |
|:--|:--|
| `sima-cli: command not found` | Venv not active: `sudo su -`, `cd`, `source sima/bin/activate` |
| Venv landed in `/root/sima` | You ran `cd` before `sudo su -` |
| `externally-managed-environment` | Create the venv first |
| `Error: No such command 'sdk'` | You ran it on the board. `sdk` is PC-side |
| WSL cannot ping the DevKit | `.wslconfig` missing, saved as `.txt`, or WSL not restarted |
| `Cannot connect to the Docker daemon` | `sudo systemctl start docker` |
| Docker dead after every restart | systemd not enabled in `/etc/wsl.conf` |
| `ssh: Could not resolve hostname d:` | Windows path used in Linux. `scp` read `D:` as a hostname |
| `scp: Connection closed` | Usually follows the above |
| Copy hangs | IP changed. `arp -a \| Select-String "192.168.137"` |
| `model archive not found` | Run from `~/object-detection`, and check `find assets -type f` |
| `source file not found` | The path is relative to where you launch `app.py`. The error lists what is actually in the folder |
| `is not a raw H.264 elementary stream` | You renamed a `.mp4` instead of converting it. Use the ffmpeg command in the error |
| `pyneat requires numpy<2` | `pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"` |
| `No src-element named "nN_demux"` | `.mp4` demuxer bug. Convert to `.h264` |
| `ModuleNotFoundError: pyneat` | You are on the PC, or pairing never ran |
| Device busy | Orphaned run: `ssh sima@<ip> pkill -f src/app.py` |
| Stuck after `loading model` | First load unpacks the archive. Give it a minute |
| No detections at all | `model.family` mismatch, then lower `decode.score_threshold` |
| Boxes in the wrong place | `resize.mode: letterbox`, `pad_value: 114`. Do not add your own maths |
| Scores all near zero | Head mismatch. YOLOX, v6 and v26 use raw-logit heads |
| Output video shorter than the input, and plays fast | Frames are being dropped. Set `runtime.overflow_policy: auto`, which picks `block` for a file so every frame is kept |
| `processed=0` and a 20 s timeout | The source caps filter is not negotiating. Leave `source.width`, `source.height` and `source.fps` at 0 |
| Dropped frames on a live source | Raise `runtime.queue_depth`, keep `overflow_policy: auto` |

### Recovery

<details>
<summary><b>DevKit firmware version mismatch</b></summary>

```
ERROR: DevKit/SDK version mismatch. DevKit 2.0.0, SDK 2.1.2
```

New boards often ship older firmware. **eLxr cannot be updated remotely**, so this runs
**on the board**. It needs internet, which Windows ICS already provides over your
Ethernet cable.

```bash
ssh sima@<devkit-ip>
sima-cli login
sima-cli update            # menu → "Update all packages to the latest"
```

The sudo password is the same one you SSH in with. Budget 15–40 minutes plus a reboot.

> `--dryrun` ends with `No ELXR update was applied`. That is what a dry run does. Run
> it again without the flag.
>
> A firmware update may wipe the board's home directory. Never keep the only copy of
> anything there.

Afterwards, re-run `sima-cli sdk setup --devkit <ip>`. If SSH complains the host key
changed, that is expected: `ssh-keygen -R <ip>`.

</details>

<details>
<summary><b>pyneat missing on the DevKit</b></summary>

<a id="pyneat-missing"></a>

Means pairing never installed it, almost always because networking was not fixed first.
Re-run pairing from **WSL** now that it works:

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

Still missing? Install by hand on the board. Match the version from `neat` in the
container, and install under `/media/nvme` because the root filesystem is too small:

```bash
sudo mkdir -p /media/nvme/neat && sudo chown "$USER:$USER" /media/nvme/neat
cd /media/nvme/neat
sima-cli login
sima-cli neat install core@v0.3.0
```

> `sima-cli` downloads into the **current directory**, so `cd` somewhere you own first.
> `-t pyneat` fetches only the wheel, which is not enough to run an app.

</details>

---

## License

The object detection model used here for testing is **Ultralytics YOLO26**, released
under **AGPL-3.0**. All other parts of this code are released under **Apache-2.0**.

## Credits

- [SiMa.ai on GitHub](https://github.com/SiMa-ai): Modalix, the Palette SDK and Neat
- [Ultralytics](https://github.com/ultralytics/ultralytics): YOLO26 models

<div align="center">

Created with ❤️ by **Muhammad Rizwan Munawar**, passionate about implementing
computer vision ideas and sharing my gains with the community.

If this saved you an afternoon, **⭐ the repo** and pass it on to someone else
bringing up a DevKit.

<br>

<a href="https://github.com/RizwanMunawar"><img src="assets/socials/github.svg" width="54" alt="GitHub"></a>
&nbsp;&nbsp;
<a href="https://www.linkedin.com/in/muhammadrizwanmunawar/"><img src="assets/socials/linkedin.svg" width="54" alt="LinkedIn"></a>
&nbsp;&nbsp;
<a href="https://x.com/muhammdrizwanmr"><img src="assets/socials/x.svg" width="54" alt="X"></a>
&nbsp;&nbsp;
<a href="https://www.youtube.com/@muhammadrizwanmunawar"><img src="assets/socials/youtube.svg" width="54" alt="YouTube"></a>
&nbsp;&nbsp;
<a href="https://muhammadrizwanmunawar.medium.com/"><img src="assets/socials/medium.svg" width="54" alt="Medium"></a>

</div>
