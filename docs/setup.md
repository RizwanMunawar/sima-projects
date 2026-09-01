# Bringing up a Modalix DevKit 3.0

Written while actually doing it. Every warning marks somewhere real time was lost.

**You only need this page if you have a board.** To see what the apps do, and to
tune every setting, start at the [README](../README.md) -- none of that needs
hardware.

Setup is once per machine: about two hours, mostly downloading.

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
├──────────────────────── THEN PICK AN APP · EVERY RUN ─────────────────────────┤
│                        │                            │                         │
│                        │ 7  fetch model + video     │                         │
│                        │      into <app>/assets/    │                         │
│                        │                            │                         │
│                        │ 8  pip install sima-vision ─────┼─>  ~/<app>, ~/src       │
│                        │      edit, copy, run       │      sima-vision <task> │
│                        │                            │                         │
│ 9  scp the result back<┼────────────────────────────┼────  annotated .mp4     │
│      keep a local copy │      and frames/           │      on the board       │
│                        │                            │                         │
└────────────────────────┴────────────────────────────┴─────────────────────────┘
```

Steps 1 to 6 are **this page**. Steps 7 to 9 are **the app README you pick**, and they
are the loop you live in afterwards.

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
> **`<devkit-ip>` appears throughout this guide and every app guide. Substitute your
> own.** The board gets its address by DHCP, so it changes between reboots: mine has been
> both `192.168.137.123` and `192.168.137.193`. Your PC keeps `192.168.137.1`, which is
> why that one is written out in full.

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
> `.wslconfig.txt`, then see [Setup errors](#setup-errors).

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

You now have both apps and the `sima` venv beside them. **Every command in this repo runs
from this directory.**

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

> ✅ Must print **"Hello from Docker!"** If not, see [Setup errors](#setup-errors).

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

> ✅ Prints a version → setup is done.
> ❌ `No such file or directory` → see [pyneat missing on the DevKit](#pyneat-missing).

> [!NOTE]
> **`mount.nfs: Connection timed out` is fine.** Setup falls back to rsync and carries
> on. The DevKit IP also changes between reboots (DHCP), so if things hang, check with
> `arp -a | Select-String "192.168.137"`.

## Now pick an app

Setup is finished. **Everything from here is app-specific**, so continue in one of these
and do not come back except for the shared reference below.

| Guide | Start here if you want |
|:--|:--|
| [**Object detection →**](detect.md) | Boxes, labels and confidence. The simplest thing that proves the whole chain works |
| [**Instance segmentation →**](segment.md) | Per-pixel masks and a background blur |
| [**Fall detection →**](fall.md) | People tracking, a fall state machine and SMTP alerts |

Each of those covers, for its own app: fetching the model and a test video, deploying,
running, pulling the result back, every config key, the overlay, tuning and its own error
table.

### Video must be raw H.264

The DevKit decodes H.264 in hardware, and `.mp4` containers hit a
[known bug](#known-issues). **Every app in this repo needs raw `.h264`.** Convert once,
losslessly:

```powershell
ffmpeg -i clip.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 clip.h264
```

Both apps read the real geometry out of the stream's SPS, so leave `source.fps`,
`source.width` and `source.height` at `0`. Renaming an `.mp4` to `.h264` does not work
and is caught at startup with the command above in the error message.

Ready-made clips, already converted, are on the
[releases page](https://github.com/RizwanMunawar/sima-projects/releases/tag/0.0.1). Each
app README has the one-line `curl` that puts them in the right place.

### Known issues

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

**Fix:** no container, no demuxer. Every app here detects `.h264` / `.264` / `.avc` /
`.bin` and builds the chain by hand:

```
FileInput → H264Parse → Queue → SimaDecode → CapsRaw
```

A container input still uses `groups.video_input` and prints the conversion command.

</details>

### Reference

#### Addresses

| What | Value | Notes |
|:--|:--|:--|
| DevKit | `<devkit-ip>`, user `sima` | DHCP, changes between reboots |
| Your PC, as the board sees it | `192.168.137.1` | Fixed by ICS, also the board's route to the internet |

#### Paths

| What | Where | Machine |
|:--|:--|:--|
| Repo | `/root/sima-projects` | WSL |
| Repo, from Windows | `\\wsl$\Ubuntu\root\sima-projects` | Windows |
| Shared workspace | `/workspace` | SDK container |
| Shared workspace | `/root/workspace` | WSL |
| SDK container name | `ghcr.io-sima-neat-sdk-latest` | WSL |
| PyNeat venv | `~/pyneat` | DevKit |
| An app, once deployed | `~/<app-name>` | DevKit |
| Playbooks | `/neat-resources/apps-src/skills/` | SDK container |
| Neat source | `/neat-resources/core-src/` | SDK container |

> [!IMPORTANT]
> **`/workspace` in the container is `/root/workspace` in WSL, not the repo.** Anything
> downloaded there is not in `/root/sima-projects` and no app will find it. Both app
> READMEs download straight into the repo for exactly this reason.

#### Five rules that prevent most problems

| # | Rule | Because |
|:--|:--|:--|
| 1 | Networking before pairing | Pairing installs over the network. No route means a silent no-op |
| 2 | Docker before the SDK | The SDK **is** a container |
| 3 | `cd` after `sudo su -` | `-` is a login shell, so it drops you in `/root` |
| 4 | Raw `.h264`, never `.mp4` | Containers hit a demuxer bug in Neat 0.3.0 |
| 5 | Always `ssh -tt` | Ctrl-C needs a pty to reach the app and release the MLA |

### Setup questions

<details>
<summary><b>Can I try anything without a DevKit?</b></summary>

Yes, the config half of either app:

```bash
sima-vision detect --validate
sima-vision segment --validate
```

Both need only `pyyaml`, run on Windows or WSL, and check that the model family resolves
to a real `BoxDecodeType` and that every setting is in range. Inference itself needs the
board, because it runs on the MLA.

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
<summary><b>Do I have to re-run the setup after every code change?</b></summary>

No. Steps 1 to 6 are once per machine. After an edit it is `pip install sima-vision` and run again.
You only re-pair (step 6) if the board's home directory is wiped, which a firmware update
can do.

</details>

<details>
<summary><b>Can I run both apps on the same board?</b></summary>

Yes. They deploy to separate directories (`~/object-detection` and
`~/instance-segmentation`) and never share state. Run them one at a time: both want the
MLA, and the second one will fail with a busy device if the first is still alive.

```bash
ssh sima@<devkit-ip> pkill -f sima-vision     # before switching apps
```

</details>

<details>
<summary><b>Can I just use numpy 2?</b></summary>

No. `pyneat` and every `simaai-*` package on the board need `numpy<2`, which is why every
`sima-vision` depends on neither, so installing it cannot upgrade them.
If pip has already upgraded you:
`pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"`

</details>

<details>
<summary><b>Where does the SDK container fit in day to day?</b></summary>

Start it from the repo root when you need `dk` or `neat`:

```bash
sudo su -
cd sima-projects
source sima/bin/activate
sima-cli sdk neat
```

| Task | Command | Run in |
|:--|:--|:--|
| SSH to the board | `dk shell` | SDK container |
| Check the sync method | `dk status` | SDK container |
| Component versions | `neat` | SDK container |

Deploying and running an app does **not** need the container. That is plain `scp` and
`ssh` from WSL, and it is written out in each app README.

</details>

### Setup errors

Bring-up problems only. Anything about a running app is in that app's own error table:
[object detection](detect.md#common-errors) ·
[instance segmentation](segment.md#common-errors) ·
[fall detection](fall.md#common-errors).

| Symptom | Fix |
|:--|:--|
| `sima-cli: command not found` | Venv not active: `sudo su -`, `cd sima-projects`, `source sima/bin/activate` |
| Venv landed in `/root/sima` | You ran `cd` before `sudo su -` |
| `externally-managed-environment` | Create the venv first |
| `Error: No such command 'sdk'` | You ran it on the board. `sdk` is PC-side |
| WSL cannot ping the DevKit | `.wslconfig` missing, saved as `.txt`, or WSL not restarted |
| `Cannot connect to the Docker daemon` | `sudo systemctl start docker` |
| Docker dead after every restart | systemd not enabled in `/etc/wsl.conf` |
| `ssh: Could not resolve hostname d:` | Windows path used in Linux. `scp` read `D:` as a hostname |
| `scp: Connection closed` | Usually follows the above |
| Copy hangs | IP changed. `arp -a \| Select-String "192.168.137"` |
| `ModuleNotFoundError: pyneat` | You are on the PC, or pairing never ran. See [below](#pyneat-missing) |
| `pyneat requires numpy<2` | `pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"` |
| `No src-element named "nN_demux"` | `.mp4` demuxer bug. Convert to [`.h264`](#video-must-be-raw-h264) |
| Device busy | Orphaned run: `ssh sima@<ip> pkill -f sima-vision` |
| DevKit/SDK version mismatch | [Firmware recovery](#recovery) below |

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

