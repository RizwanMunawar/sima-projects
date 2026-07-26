<div align="center">

<br>

# ⚡ SiMa Neat SDK

### Live YOLO object detection on a Modalix DevKit

<br>

[![SiMa.ai](https://img.shields.io/badge/SiMa.ai-Modalix-E63946?style=for-the-badge)](https://sima.ai)
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

---

## What this is

A **setup guide** and a **working detector app**, both written while actually bringing
up a DevKit. Every warning marks somewhere real time was lost.

```
   ┌──────────────┐        ┌──────────────────┐        ┌───────────────┐
   │  WINDOWS PC  │        │   WSL2 · UBUNTU  │        │    DEVKIT     │
   ├──────────────┤        ├──────────────────┤        ├───────────────┤
   │  Chrome      │◄──────►│  sima-cli        │◄──────►│  MLA          │
   │  scp / ssh   │ :9900  │  Docker + SDK    │  UDP   │  your app     │
   │              │        │  Neat Insight    │ 9000/  │               │
   │              │        │                  │ 9100   │               │
   └──────────────┘        └──────────────────┘        └───────────────┘
        viewer                build + receive             inference
```

Your app runs **on the DevKit**. It sends H.264 video and JSON detections over UDP,
and Insight recombines them in your browser.

```bash
git clone https://github.com/RizwanMunawar/sima-projects.git
cd sima-projects
```

Already have a DevKit set up? Jump to [Deploy and run](#step-9). Starting
from a bare machine? Work through the steps in order.

> [!TIP]
> **Most problems are a command typed in the wrong box.** Check your prompt:
> `PS C:\>` is Windows · `root@neat-sdk:/workspace#` is the container ·
> `sima@modalix:~$` is the DevKit. **Windows paths only work in PowerShell.**

---

## Complete workflow

Who does what, and in which order. **Click any step to jump to it.**

```
      🪟 WINDOWS PC              🐧 WSL2 · UBUNTU            🔴 MODALIX DEVKIT
   ═══════════════════        ═══════════════════════      ═══════════════════

   ┌───────────────────────────── ONE-TIME SETUP ─────────────────────────────┐

    ① cable up  ═══════════════════════════════════════════►  DHCP → .123
         USB + Ethernet                                        board powers on

    ② wsl --install ──────────►  Ubuntu ready

    ③ .wslconfig  ────────────►  WSL takes .137.1  ◄════════►  now reachable
         mirrored networking        same subnet                 both directions

    ④ firewall rules ─────────►  UDP 9000/9100 open
         Hyper-V inbound            ready to receive

                                 ⑤ git clone + sima-cli
                                      repo + venv

                                 ⑥ docker + nfs
                                      SDK needs a container

                                 ⑦ sima-cli sdk setup ═════►  pyneat + runtime
                                      12.6 GB image             installed on board

                                 ⑧ download model
                                      yolo26m .tar.gz

   └──────────────────────────────────────────────────────────────────────────┘

   ┌────────────────────────── EVERY RUN, REPEAT ─────────────────────────────┐

    ⑨ scp -r object-detection/ ══════════════════════════►  ~/object-detection
         edit → copy → run                                   python3 src/app.py
                                                                    │
                                                                    ▼
                                                             MLA inference
                                                             ┌──────────────┐
    ⑩ browser  ◄──── Insight ◄──── UDP 9000 + 9100 ◄════════╡ VideoSender  │
         localhost:9900   live overlay                       │ MetadataSend │
                                                             ├──────────────┤
                                        detections.mp4  ◄════╡ VideoWriter  │
                                        frames/*.jpg         └──────────────┘
                                          on the board

   └──────────────────────────────────────────────────────────────────────────┘
```

| # | Step | Runs on | Time | |
|:--|:--|:--|:--|:--|
| ① | [Cable up the DevKit](#step-1) | 🪟 PowerShell | 15 min | |
| ② | [Install WSL2](#step-2) | 🪟 PowerShell | 10 min | |
| ③ | [Mirrored networking](#step-3) | 🪟 PowerShell | 5 min | 🔴 |
| ④ | [Firewall](#step-4) | 🪟 PowerShell | 2 min | 🔴 |
| ⑤ | [Get the code and install sima-cli](#step-5) | 🐧 WSL | 5 min | |
| ⑥ | [Docker Engine + NFS](#step-6) | 🐧 WSL | 10 min | |
| ⑦ | [Install the Neat SDK](#step-7) | 🐧 WSL | 30 to 60 min | |
| ⑧ | [Download a model](#step-8) | 🐳 Container | 5 min | |
| ⑨ | [Deploy and run](#step-9) | 🔴 DevKit | 2 min | ♻️ |
| ⑩ | [Watch it](#step-10) | 🌐 Browser | 2 min | ♻️ |

🔴 **Load-bearing.** Doing these late is the usual way to lose an afternoon, because
step ⑦ installs onto the board over the network and fails silently without it.
♻️ Repeats on every code change.

<br>

<a id="step-1"></a>
### 1. Cable up the DevKit

USB cable (serial console) + Ethernet straight to your PC. Open the
[serial tool](https://docs.sima.ai/_static/tools/serial/index.html), set the DevKit to
**DHCP**.

```powershell
ping 192.168.137.123
```

> ✅ Must reply. Nothing else works until it does.

<br>

<a id="step-2"></a>
### 2. Install WSL2

```powershell
wsl --install -d Ubuntu      # PowerShell as Administrator
wsl -l -v                    # want: Ubuntu · Running · 2
```

<br>

<a id="step-3"></a>
### 3. Mirrored networking ![critical](https://img.shields.io/badge/-CRITICAL-E63946?style=flat-square)

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
wsl -- ping -c 2 192.168.137.123    # must reply
```

> ✅ **Both must pass.** Be stubborn here.
>
> 💡 Made the file in Notepad? Check it is not secretly `.wslconfig.txt`.

<br>

<a id="step-4"></a>
### 4. Firewall ![critical](https://img.shields.io/badge/-CRITICAL-E63946?style=flat-square)

Mirrored mode puts WSL behind the Hyper-V firewall, which blocks inbound by default.
Your DevKit pushing video **is** inbound. Skip this and Insight loads perfectly and
shows nothing, with no error anywhere.

```powershell
# PowerShell as Administrator
New-NetFirewallHyperVRule -Name "NeatVideo" -DisplayName "Neat Insight video" `
  -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
  -Protocol UDP -LocalPorts 9000-9079 -Action Allow

New-NetFirewallHyperVRule -Name "NeatMeta" -DisplayName "Neat Insight metadata" `
  -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
  -Protocol UDP -LocalPorts 9100-9179 -Action Allow
```

> 💡 These live in Windows, not the distro, so they survive `wsl --unregister`.

<br>

<a id="step-5"></a>
### 5. Get the code and install sima-cli

```bash
# WSL
sudo su -                              # become root FIRST
apt update && apt install -y git python3-venv python3-pip

cd /mnt/d/work                         # or wherever you keep projects
git clone https://github.com/RizwanMunawar/sima-projects.git
cd sima-projects

python3 -m venv sima
source sima/bin/activate
pip install sima-cli
sima-cli login                         # needs a community.sima.ai account
```

This gives you the detector app in [`object-detection/`](object-detection/) and the
`sima` venv beside it. Everything below assumes you are in the repo root.

> [!CAUTION]
> **`cd` after `sudo su -`, never before.** The `-` makes it a login shell that drops
> you in `/root`. Reverse them and your venv is silently built at `/root/sima`.

> 💡 Every new session needs three lines again: `sudo su -`, `cd` to the repo, and
> `source sima/bin/activate`.

> 💡 Cloning into `/mnt/d/...` keeps the repo on your Windows drive, so you can edit it
> in VS Code on Windows and run it from WSL. If the repo is private, use
> `gh repo clone RizwanMunawar/sima-projects` or an SSH remote.

<br>

<a id="step-6"></a>
### 6. Docker Engine + NFS

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

> ✅ Must print **"Hello from Docker!"**

<br>

<a id="step-7"></a>
### 7. Install the Neat SDK ![size](https://img.shields.io/badge/-12.6_GB_·_30--60_min-DC3545?style=flat-square)

```bash
sudo su -
cd /mnt/d/work/sima-projects
source sima/bin/activate
sima-cli install ghcr:sima-neat/sdk
sima-cli sdk setup --devkit 192.168.137.123
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
ssh sima@192.168.137.123 "~/pyneat/bin/python3 -c 'import pyneat; print(pyneat.__version__)'"
```

> ✅ Prints a version → done.
> ❌ `No such file or directory` → see [Recovery](#recovery).

> [!NOTE]
> **`mount.nfs: Connection timed out` is fine.** Setup falls back to rsync and carries
> on. Also: **the DevKit IP changes between reboots** (DHCP). If things hang, check with
> `arp -a | Select-String "192.168.137"`.

<br>

<a id="step-8"></a>
### 8. Download a model

```bash
sima-cli sdk neat        # WSL, starts the container and drops you inside

# in the container
sima-cli login
mkdir -p /workspace/assets/models && cd /workspace/assets/models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz
```

| Variant | Speed | Accuracy |
|:--|:--:|:--:|
| `yolo26n` | ⚡⚡⚡⚡⚡ | ★★☆☆☆ |
| `yolo26s` | ⚡⚡⚡⚡ | ★★★☆☆ |
| **`yolo26m`** | ⚡⚡⚡ | ★★★★☆ ⭐ |
| `yolo26l` | ⚡⚡ | ★★★★☆ |
| `yolo26x` | ⚡ | ★★★★★ |

<br>

<a id="step-9"></a>
### 9. Deploy and run

The app lives in [`object-detection/`](object-detection/):

```
object-detection/
├── config.yaml          # every setting lives here
├── assets/
│   ├── models/          # .tar.gz model packs  (not in git)
│   └── video/           # .h264 streams        (not in git)
└── src/
    ├── app.py           # the pipeline
    ├── coco_labels.txt  # 80 COCO class names
    └── requirements.txt
```

Models and video are gitignored, so after cloning you supply your own: download a model
(step 8) into `assets/models/`, and put a `.h264` stream in `assets/video/`.

On the DevKit, once per board:

```bash
pip install -r ~/object-detection/src/requirements.txt
```

> [!CAUTION]
> **Never let pip pull numpy 2.x.** `pyneat` and every `simaai-*` package need
> `numpy<2`. The pins in `requirements.txt` handle it. If you already broke it:
> `pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"`

**One command copies everything.** Do this after every change:

```powershell
scp -r object-detection/ sima@192.168.137.193:~
```

> [!IMPORTANT]
> Run it from `d:\work\sima-projects`, and mind the **IP**, yours may differ.
> This overwrites the board copy, so keep your originals on the PC.

Then on the DevKit:

```bash
ssh -tt sima@192.168.137.193                     # two t's, see below
source ~/pyneat/bin/activate
cd ~/object-detection && python3 src/app.py --config config.yaml
```

Healthy output:

```
source: type=video uri=assets/video/video-4.h264 stream=1920x1080@25
preprocess: ... resize=letterbox pad=114 normalize=coco_yolo
model: ... family=yolo26 decode_type=YoloV26 labels=80
graph built
insight: host=192.168.137.1 video=9000 metadata=9100 channel=0
running. press Ctrl-C to stop.
[50] 24.8 fps, 6.2 detections/frame avg
```

> [!CAUTION]
> **Use `ssh -tt`, two t's.** Without a pty, Ctrl-C never reaches the app. It keeps
> running invisibly holding the MLA and your next run fails.
> Rescue: `ssh sima@192.168.137.193 pkill -f src/app.py`

<br>

<a id="step-10"></a>
### 10. Watch it

<div align="center">

## 🔗 [https://localhost:9900](https://localhost:9900)

**select channel 0**

</div>

> [!IMPORTANT]
> **Open Insight and select the channel _before_ you start the app.** Video and
> detections go out over UDP, which buffers nothing. A viewer opened after the stream
> ends sees an empty page, even though everything worked.
>
> Short clips make this worse. A 12-second video is over before you can switch windows.
> Loop it into something you can actually watch:
> ```powershell
> ffmpeg -stream_loop 9 -i clip.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 video-loop.h264
> ```

> [!WARNING]
> **Ignore the address `neat` prints.** It says `https://192.168.137.1:9900`, which
> Windows cannot reach: that counts as inbound to the WSL VM and the firewall drops it.
> `localhost` takes a different path. Same for the VS Code URL, keep the token and swap
> the host.

**How to tell it is actually streaming.** The app prints these on the way out:

```
metadata: sent=102 failures=0 would_block=0
video: wrote 3012 frames to sandbox/detections.mp4 (48.3 MB)
```

`failures=0` means every datagram left the board. If Insight is still blank after that,
the problem is on the receiving side: the firewall (step 4), or `insight.host` pointing
at `127.0.0.1`.

### The recording

**Every processed frame** is written to an annotated video on the DevKit, so a run
always leaves something to review even if nobody was watching Insight at the time.

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

A plain rectangle in the class colour, a centre dot, and a filled caption directly
above carrying the class name and confidence in white. Captions flip inside the box
rather than clipping off the top of the frame. Colours come from a 20-entry palette
keyed to class id, so the same class is always the same colour. Line weight, text size
and padding scale with the frame, so 4K does not get hairlines and 480p does not get
slabs.

The same annotated frame is what Insight receives, so the browser view and the
recording look identical. Set `output.insight.annotated: false` to send the raw frame
instead and let Insight draw its own overlay from the metadata stream.

Pull the results back to your PC:

```powershell
scp sima@192.168.137.193:~/object-detection/sandbox/detections.mp4 .
scp -r sima@192.168.137.193:~/object-detection/sandbox/frames .
```

| Setting | Meaning |
|:--|:--|
| `output.video.path` | Where to write, relative to the launch directory |
| `output.video.codec` | 4-char FourCC. `mp4v` by default, auto-falls back to `MJPG`/`.avi` |
| `output.video.fps` | `0` matches the source rate |
| `output.video.hud` | Small FPS badge in the corner. Turn off for clean footage |
| `output.save.every` | Write every Nth frame as a JPEG. `0` disables |
| `output.insight.annotated` | `true` sends our overlay to Insight, `false` sends the raw frame |

---

## Configuration

Everything lives in `object-detection/config.yaml`. These settings matter:

```yaml
model:
  path: assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz
  family: yolo26                       # must match your model

source:
  type: video                          # video | rtsp | usb
  uri: assets/video/video-loop.h264    # DevKit path, relative to ~/object-detection

output:
  video:
    enable: true
    path: sandbox/detections.mp4       # annotated video, written on the DevKit
    hud: true                          # small FPS badge
  insight:
    annotated: true                    # Insight shows our overlay
  save:
    enable: true
    dir: sandbox/frames                # annotated stills
    every: 25
  insight:
    host: 192.168.137.1                # NOT 127.0.0.1
```

| ❌ Mistake | What happens |
|:--|:--|
| `uri: C:\Users\...\video.mp4` | The DevKit has no `C:` drive |
| `uri: r"C:\path\file.mp4"` | `r"..."` is Python. YAML keeps the `r` and quotes |
| `host: 127.0.0.1` | Means "the board itself". Video goes nowhere |
| `family` mismatched | No detections, or every score near zero |
| A `.mp4` source | Hits a demuxer bug. Convert to `.h264`, see below |

### Video must be raw H.264

The DevKit decodes H.264 in hardware, and `.mp4` containers hit a
[known bug](#known-issues). Convert once, losslessly:

```powershell
ffmpeg -i video-4.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 video-4.h264
```

Raw streams carry no metadata, so set geometry explicitly:

```yaml
source:
  fps: 25
  width: 1920
  height: 1080
```

### First run: prove the model offline

Debugging the model and the network at once is what makes this painful. Set:

```yaml
runtime: { frames: 100, profile: true }
output:
  save:   { enable: true }
  insight: { enable: false }
```

100 frames, annotated JPEGs to disk, timings, exits by itself, **zero networking**.
Boxes look right? Model and config are correct, so anything later is transport.

---

## Daily loop

```bash
sudo su - && cd /mnt/d/work/sima-projects && source sima/bin/activate && sima-cli sdk neat
```

Then: **edit → copy → run → watch**, repeat.

```powershell
scp -r object-detection/ sima@192.168.137.193:~
```

<details>
<summary><b>📌 Handy extras</b></summary>

<br>

```powershell
# pull annotated frames back
scp -r sima@192.168.137.193:~/object-detection/sandbox .

# stop typing your password
ssh-keygen -t ed25519 -C "devkit"
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh sima@192.168.137.193 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

```bash
# from inside the SDK container
dk shell      # SSH to the DevKit
dk status     # sync method and remote path
neat          # component versions and ports
```

</details>

---

## Recovery

<details>
<summary><b>DevKit firmware version mismatch</b></summary>

<br>

```
ERROR: DevKit/SDK version mismatch. DevKit 2.0.0, SDK 2.1.2
```

New boards often ship older firmware. **eLxr cannot be updated remotely**, so this runs
**on the board**. It needs internet, which Windows ICS already provides over your
Ethernet cable.

```bash
ssh sima@192.168.137.123
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

<a id="recovery"></a>

<br>

Means pairing never installed it, almost always because networking was not fixed first.
Re-run pairing from **WSL** now that it works:

```bash
sima-cli sdk setup --devkit 192.168.137.123
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

## Troubleshooting

<details open>
<summary><b>Setup</b></summary>

<br>

| Symptom | Fix |
|:--|:--|
| `sima-cli: command not found` | Venv not active: `sudo su -`, `cd`, `source sima/bin/activate` |
| Venv landed in `/root/sima` | You ran `cd` before `sudo su -` |
| `externally-managed-environment` | Create the venv first |
| `Error: No such command 'sdk'` | You ran it on the board. `sdk` is PC-side |
| WSL cannot ping the DevKit | `.wslconfig` missing, saved as `.txt`, or WSL not restarted |
| `Cannot connect to the Docker daemon` | `sudo systemctl start docker` |
| Docker dead after every restart | systemd not enabled in `/etc/wsl.conf` |

</details>

<details>
<summary><b>Copying and paths</b></summary>

<br>

| Symptom | Fix |
|:--|:--|
| `ssh: Could not resolve hostname d:` | Windows path used in Linux. `scp` read `D:` as a hostname |
| `scp: Connection closed` | Usually follows the above |
| `model archive not found` | Run from `~/object-detection`, and check `find assets -type f` |
| `failed to open source` | Same, for the video |
| Copy hangs | IP changed. `arp -a \| Select-String "192.168.137"` |

</details>

<details>
<summary><b>Running</b></summary>

<br>

| Symptom | Fix |
|:--|:--|
| `pyneat requires numpy<2` | `pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"` |
| `No src-element named "nN_demux"` | `.mp4` demuxer bug. Convert to `.h264` |
| `ModuleNotFoundError: pyneat` | You are on the PC, or pairing never ran |
| Device busy | Orphaned run: `ssh sima@<ip> pkill -f src/app.py` |
| Stuck after `loading model` | First load unpacks the archive. Give it a minute |

</details>

<details>
<summary><b>Detections and display</b></summary>

<br>

| Symptom | Fix |
|:--|:--|
| Insight blank but `sent=N failures=0` | The stream already ended. UDP buffers nothing, so open the viewer **first** and use a looped source |
| Insight loads, no video | **Firewall.** Step 4 skipped. Most common failure by far |
| Insight blank, no errors | `insight.host` is `127.0.0.1`. Use `192.168.137.1` |
| Stops early with `timed out waiting for detections` | Source stalled. Raise `runtime.pull_timeout_ms`, or re-run: partial playback still proves the pipeline |
| `192.168.137.1:9900` will not load | Expected. Use `https://localhost:9900` |
| No detections at all | `model.family` mismatch, then lower `decode.score_threshold` |
| Boxes in the wrong place | `resize.mode: letterbox`, `pad_value: 114`. Do not add your own maths |
| Scores all near zero | Head mismatch. YOLOX, v6 and v26 use raw-logit heads |
| Dropped frames | Raise `runtime.queue_depth`, keep `overflow_policy: keep_latest` |

</details>

---

## How the app works

<details>
<summary><b>Pipeline shape, preprocessing and decode types</b></summary>

<br>

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
| Insight senders | `docs/develop-apps/advanced-concepts/application-design/` |
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
frame fans out to the video writer, the still writer, the `MetadataSender`, and a
second small graph (`Input -> VideoSender`) that encodes for Insight.

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

---

## Known issues

<details>
<summary><b><code>groups.video_input</code> cannot play <code>.mp4</code> · Neat 0.3.0</b></summary>

<a id="known-issues"></a>

<br>

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

---

## Reference

| | |
|:--|:--|
| **DevKit** | `192.168.137.123` (DHCP, changes), user `sima` |
| **Your PC, as the board sees it** | `192.168.137.1` |
| **Insight** | `https://localhost:9900` |
| **Workspace** | `/workspace` = `/root/workspace` = `\\wsl$\Ubuntu\root\workspace` |
| **Container** | `ghcr.io-sima-neat-sdk-latest` |
| **Board venv** | `~/pyneat` |
| **Playbooks** | `/neat-resources/apps-src/skills/` |
| **Neat source** | `/neat-resources/core-src/` |

| Port | | |
|:--|:--|:--|
| `9000-9079` | UDP | Video to Insight |
| `9100-9179` | UDP | Detection metadata |
| `9900` | TCP | Insight web UI |
| `8554` | TCP | RTSP |
| `8022` | TCP | Web SSH |

---

<div align="center">

## Six rules

**1.** Networking before pairing · **2.** Docker before the SDK ·
**3.** `cd` after `sudo su -` <br>
**4.** `insight.host = 192.168.137.1` · **5.** Raw `.h264`, not `.mp4` ·
**6.** Always `ssh -tt`

<br>

Built with the `neat-application-builder` playbook <br>
API details verified against `/neat-resources/core-src`, not from memory

<br>

![SiMa](https://img.shields.io/badge/SiMa.ai-E63946?style=flat-square)
![Modalix](https://img.shields.io/badge/Modalix-457B9D?style=flat-square)
![Neat](https://img.shields.io/badge/Neat-2A9D8F?style=flat-square)

</div>
