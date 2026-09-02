<div align="center">

<img src="../assets/sima-devkit-docs-logo-fall-detection.png" alt="SiMa Neat SDK: live YOLO fall detection on a Modalix DevKit 3.0" width="640">

<br>

[![SiMa.ai](https://img.shields.io/badge/SiMa.ai-Modalix_DevKit_3.0-E63946?style=for-the-badge)](https://sima.ai)
[![Palette SDK](https://img.shields.io/badge/Palette_SDK-2.1.2-457B9D?style=for-the-badge)](https://docs.sima.ai)
[![Neat](https://img.shields.io/badge/Neat-0.3.0-2A9D8F?style=for-the-badge)](https://docs.sima.ai)

![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics_YOLO26-FFB703?style=flat-square&labelColor=333)
![Alerts](https://img.shields.io/badge/alerts-SMTP_email-E63946?style=flat-square)
![Runs on](https://img.shields.io/badge/inference-on_the_MLA-2A9D8F?style=flat-square)

</div>

## What this is

**Fall detection for warehouses and malls**, running on a Modalix DevKit 3.0. A YOLO26
detect head finds people on the MLA, a tracker follows each of them between frames, and a
small state machine decides when one of them has gone down. A confirmed fall emails
whoever is on shift, with a snapshot attached.

```
   ┌──────────────────────────────────────────────────────────────────┐
   │ ┌──────────┐                                                     │
   │ │ FPS: 25  │   ┌──────────────────┐                              │
   │ └──────────┘   │ #7 falling 0.6/1.5s│      amber, timer running  │
   │  ┌───────────┐ ├──────────────────┬─┘                            │
   │  │#3 upright │ │                  │      ┌──────────────┐        │
   │  ├───────────┤ │                  │      │ #11 fallen   │        │
   │  │           │ │                  │      ├──────────────┴─────┐  │
   │  │     •     │ │        •         │      │        •           │  │
   │  └───────────┘ └──────────────────┘      └────────────────────┘  │
   │    green                                    red, alert sent      │
   ├──────────────────────────────────────────────────────────────────┤
   │            FALL DETECTED  #11  -  Warehouse camera 1             │
   └──────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **This page assumes a paired board.** The one-time DevKit bring-up (cabling, WSL2,
> networking, Docker, the Neat SDK) is in the [root README](../README.md) and is shared
> by every app here. Do steps 1 to 6 there once, then come back.
>
> ✅ You are ready when this prints a version:
> ```bash
> ssh sima@<devkit-ip> "~/pyneat/bin/python3 -c 'import pyneat; print(pyneat.__version__)'"
> ```

> [!CAUTION]
> **This is a safety aid, not a safety system.** It watches bounding boxes from a single
> camera. It will miss falls behind a rack, out of frame, or into a pile of boxes, and it
> will occasionally page someone for a crouch. Treat an alert as "go and look", never as
> the only thing standing between a person and help.

---

## Contents

| Section | What it covers |
|:--|:--|
| [See it before you deploy](#see-it-before-you-deploy) | The overlay on a laptop, no hardware |
| [Run it in three commands](#run-it-in-three-commands) | Push, run, pull the result back |
| [Get a model and a test video](#get-a-model-and-a-test-video) | What the first run fetches into `assets/`, by hand |
| [Set up email alerts](#set-up-email-alerts) | SMTP, the password rule, `--test-alert` |
| [Deploy and run](#deploy-and-run) | `scp` the app over and run it |
| [See the result](#see-the-result) | Pull `falls.mp4`, `frames/` and `alerts/` back |
| [How a fall is decided](#how-a-fall-is-decided) | The three signals and the state machine |
| [Tuning](#tuning) | What to change when it misses, or cries wolf |
| [The overlay](#the-overlay) | Track colours, ids, countdown, alert banner |
| [Configuration](#configuration) | `config.yaml`, and the mistakes it catches |
| [Daily loop](#daily-loop) | The commands you repeat after every edit |
| [Questions people ask](#questions-people-ask) | FAQ: Gmail, false alarms, cameras, privacy |
| [Common errors](#common-errors) | One table, symptom to fix |

## See it before you deploy

No board, no model, no SDK. This draws tracked people, their states and the alert banner
using your own config:

```bash
pip install "sima-vision[preview]"
sima-vision init fall      # writes a documented config.yaml
sima-vision preview --task fall -o preview.png
```

<div align="center">
<img src="../assets/preview-fall.png" alt="Fall detection preview: per-state box colours and the alert banner" width="680">
</div>

It runs the same drawing code the board does, over a synthetic scene, so every
`visualization:` value below can be tuned here first. **No model is run** --
the detections are placed for you so there is something to draw.

## Run it in three commands

On the DevKit. There is nothing to clone and nothing to `scp`:

```bash
pip install sima-vision       # 1. install

sima-cli login                # 2. the model packs need a community.sima.ai account

sima-vision fall              # 3. run it
```

Step 3 needs no arguments. The sample clip and the model archive are this task's
defaults; both land in `./assets/` on the first run, the clip straight from a public
GitHub release and the model through `sima-cli`, and every run after that reuses them.

To use your own instead, pass a path or an `https` URL:

```bash
sima-vision fall --source my-clip.h264 --model my-model.tar.gz
sima-vision fall --source https://example.com/my-clip.h264
```

Then pull the result back and look at it:

```bash
scp sima@<devkit-ip>:~/falls.mp4 .
```

Coloured boxes that follow people means the whole chain works.

A config file is optional, and so are the flags. For a setup you keep:

```bash
sima-vision init fall      # documented config.yaml, here
sima-vision fall           # picks it up on its own
```

Every setting has a flag too, and flags win over the file. See
[the CLI reference](../README.md#reference).

## Get a model and a test video

This app uses the **ordinary YOLO26 detect head**, the same pack the
[detector app](detect.md) uses. No pose or segmentation model is
needed: a bounding box is enough to tell upright from horizontal.

Run both blocks **in WSL, from the repo root**:

```bash
# WSL
sudo su -
cd sima-projects
source sima/bin/activate
sima-cli login                                    # needs a community.sima.ai account

MODELS=https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection
MODEL=yolo26m-det-bf16-mla_tess-b1.tar.gz

mkdir -p assets/models
mkdir -p assets/models
(cd assets/models && sima-cli download "$MODELS/$MODEL")
```

```bash
VIDEOS=https://github.com/RizwanMunawar/sima-projects/releases/download/0.0.1

mkdir -p assets/videos
curl -L -o assets/videos/people-walking-inside-mall.h264 \
  $VIDEOS/people-walking-inside-mall.h264
curl -L -o assets/videos/people-walking-outside-mall.h264 \
  $VIDEOS/people-walking-outside-mall.h264
```

> [!IMPORTANT]
> **`sima-cli download` writes into the current directory**, which is the whole reason for
> the subshell -- it keeps the `cd` from leaking into the rest of your session.
> Downloading from
> anywhere else, including the container's `/workspace`, is how the pack ends up
> [somewhere `config.yaml` cannot see](setup.md#paths).

> [!NOTE]
> **Neither shipped clip contains a fall**, so a first run should report `falls=0`. That
> is the useful baseline: it tells you the tracker is stable and nothing is crying wolf.
> To see the alert path fire, point `source.uri` at footage of an actual fall, or lower
> `fall.aspect_ratio` until someone bending over trips it.

Bringing your own footage? It must be raw H.264, not `.mp4` — see
[Video must be raw H.264](setup.md#video-must-be-raw-h264).

<a id="set-up-email-alerts"></a>
## Set up email alerts

> [!CAUTION]
> **Never put the SMTP password in `config.yaml`.** That file is committed to git, and a
> password in it is a password on GitHub. There is deliberately no key for it. The app
> reads an environment variable instead, named by `alerts.smtp.password_env`.

**1. Fill in the addresses** in `config.yaml`:

```yaml
alerts:
  enable: on
  dry_run: on                    # still not sending; see step 3
  site: "Warehouse camera 1"     # goes in the subject, so you know where to go
  from: "alerts@example.com"
  to: ["safety@example.com", "shift-lead@example.com"]
  cooldown_seconds: 60
  smtp:
    host: smtp.gmail.com
    port: 587
    starttls: on
    username: "alerts@example.com"
    password_env: FALL_ALERT_SMTP_PASSWORD
```

**2. Export the password** on the DevKit, in the shell that runs the app:

```bash
export FALL_ALERT_SMTP_PASSWORD='your-app-password'
```

For Gmail this must be an [App Password](https://myaccount.google.com/apppasswords), not
your account password, and it needs 2-factor authentication on the account. Office 365
and most corporate relays use the same shape on port 587 with `starttls: on`; a relay on
port 465 wants `ssl: on` and `starttls: off`.

**3. Prove the wiring** without waiting for anyone to fall over:

```bash
sima-vision fall --test-alert
```

With `dry_run: on` it composes the message and prints the subject, touching no network.
Turn `dry_run: off` and run it again to actually send one. It is synchronous and reports
the real SMTP exception, which is why it is the command to reach for when mail is not
arriving.

| Port | `starttls` | `ssl` | Typical use |
|:--|:--|:--|:--|
| 587 | `on` | `off` | Gmail, Office 365, most providers |
| 465 | `off` | `on` | Implicit TLS |
| 25 | `off` | `off` | Internal relay, usually with `username: ""` |

**Cooldown.** `cooldown_seconds` is a floor on the gap between alerts across *every*
track, not per person. One fall is one event; a camera pointed at a busy aisle with a
badly tuned threshold is forty, and forty emails get filtered to a folder nobody reads.

<a id="deploy-and-run"></a>
## Deploy and run

There is nothing to copy. Install the package on the board and run it:

```bash
pip install sima-vision
```

Your working directory holds only what is yours:

```
~/
├── config.yaml          # sima-vision init fall
└── assets/
    ├── models/          # .tar.gz model packs   (sima-cli download)
    └── videos/          # .h264 streams         (sima-vision fetch)
```

> [!CAUTION]
> **Never let pip pull numpy 2.x.** `pyneat` and every `simaai-*` package need
> `numpy<2`. `sima-vision` depends on neither numpy nor OpenCV precisely so that
> installing it cannot upgrade them -- the board provides both. If you already broke it:
> `pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"`

```bash
pip install sima-vision
```

Then on the DevKit:

```bash
ssh -tt sima@<devkit-ip>                     # two t's, see below
source ~/pyneat/bin/activate
export FALL_ALERT_SMTP_PASSWORD='your-app-password'
sima-vision fall
```

Healthy output:

```
source: type=video uri=assets/videos/people-walking-inside-mall.h264 stream=1920x1080@30
model: ... family=yolo26 decode_type=YoloV26 labels=80
fall: watching person | aspect>=1.2 height<=55% descent>=55%/s | confirm=1.5s recover=3.0s
alerts: smtp.gmail.com:587 starttls as alerts@example.com -> safety@example.com | cooldown=60.0s snapshot=on
runtime: preset=reliable overflow=block queue_depth=3
graph built
running. press Ctrl-C to stop.
[50] 24.8 fps, 3.2 people/frame avg, 0 fall(s) so far
[FALL] track #11 (person) at frame 412 aspect=1.83 descent=612.4px/s snapshot=alerts/fall_track011_frame000412.jpg
[100] 24.6 fps, 3.4 people/frame avg, 1 fall(s) so far
```

> [!CAUTION]
> **Use `ssh -tt`, two t's.** Without a pty, Ctrl-C never reaches the app. It keeps
> running invisibly holding the MLA and your next run fails.
> Rescue: `ssh sima@<devkit-ip> pkill -f sima-vision`

<a id="see-the-result"></a>
## See the result

```bash
scp sima@<devkit-ip>:~/falls.mp4 .
scp -r sima@<devkit-ip>:~/alerts .      # one snapshot per alert
scp -r sima@<devkit-ip>:~/frames .
```

On exit the app reports what happened, including what the alert thread managed:

```
processed=3012 timeouts=0 falls=2
alerts: sent=2 failed=0 dropped=0 suppressed_by_cooldown=5
video: wrote 3012 frames to falls.mp4 (44.1 MB)
```

`suppressed_by_cooldown` is not an error. It is the cooldown doing its job.

## How a fall is decided

<details>
<summary><b>The three signals, the tracker and the state machine</b></summary>

### Why a tracker is needed at all

A fall is a *change*, so a single frame cannot show one. The same person has to stay
recognisable between frames before "they were upright and now they are not" means
anything. Association is greedy IoU: match every box to the nearest existing track,
highest overlap first, above `tracking.iou_threshold`.

It has no motion model, so two people who cross while heavily overlapping can swap ids.
That costs a duplicate alert at worst, which is the right way round for a safety feature.
A Kalman filter would be more correct and considerably more to get wrong.

### The three signals

All of them come from the bounding box. No pose model is involved.

| Signal | Fires when | Catches |
|:--|:--|:--|
| **aspect** | `width / height >= fall.aspect_ratio` | Someone lying down. A standing person is tall and narrow; a fallen one is wide and short |
| **collapse** | `height <= fall.height_drop x` that person's own learned upright height | A person who is down but foreshortened by the camera angle, so the box never goes wide |
| **descent** | box centre drops faster than `fall.descent_rate` of the frame height per second | The fall itself, which is what separates it from lying down deliberately |

`collapse` compares against a **per-track** reference, not a constant. A person near the
camera and a person at the far end of an aisle have wildly different box heights, so the
app learns each person's own upright height while their aspect ratio says they are
standing, and smooths it so a moment clipped by the frame edge does not poison it.

### The state machine

Any one signal can fire spuriously for a frame or two, so nothing is reported until it
has held:

```
   upright ──any signal──> falling ──held confirm_seconds──> fallen ──> ALERT
      ^                       |                                 |
      |                  signal clears                    signal clears
      |                       v                                 v
      └──── upright for recover_seconds ──── recovering <───────┘
```

`confirm_seconds` is the single most important number in the file. Too low and a forklift
driver bending to a bottom shelf pages the safety officer; too high and the alert is
late. 1.5 s is a reasonable start for a mall or warehouse aisle.

Once a track reaches `fallen` it does not alert again until it has been `upright` for
`recover_seconds`, so one person on the floor is one email, not one per frame.

### Where the clock comes from

Timestamps are the **source's** (`sample.pts_ns`), not wall-clock. With
`runtime.overflow_policy: auto` a file is processed slower than realtime, so wall-clock
seconds would make every descent look slow and the descent signal would never fire. Using
stream time means the thresholds mean the same thing on a file and on a live camera.

### Why alerts run on their own thread

Sending mail takes anywhere from tens of milliseconds to a TCP timeout, and the run loop
cannot afford either — the app that stalls while emailing is the app that misses the
*next* fall. Alerts go onto a bounded queue and a background thread sends them. If the
queue is full the alert is dropped and counted, which is the correct trade.

</details>

## Tuning

Start with `fall.enable: off`, watch the video, and get the **tracking** stable first: ids
that stick to people and do not flicker. Only then turn the rules on.

| Symptom | Change |
|:--|:--|
| Alerts for people bending or crouching | Raise `fall.confirm_seconds`, then `fall.aspect_ratio` |
| Real falls missed | Lower `fall.confirm_seconds`, then `fall.aspect_ratio` toward 1.0 |
| Alerts from someone sitting on the floor | Raise `fall.confirm_seconds`; sitting is a genuine ambiguity for a box-based detector |
| Ids flicker, one person becomes many | Lower `tracking.iou_threshold`, raise `tracking.max_age` |
| Distant people trigger nonsense | Raise `fall.min_box_height` |
| Nothing is tracked at all | `decode.score_threshold` too high, or `tracking.classes` does not say `person` |
| Too many emails | Raise `alerts.cooldown_seconds` |
| Camera is mounted high, looking down | Aspect is unreliable from directly above. Lean on `height_drop` and `descent_rate`, and raise `aspect_ratio` well above 1.2 |

A camera angle that shows people **side-on** is worth far more than any threshold here.
Aspect ratio is the strongest signal, and it only works when a fallen person is wide in
the image.

## The overlay

Boxes are coloured by state rather than by class, because on this app the state is the
information:

| Colour | State | Meaning |
|:--|:--|:--|
| Green | `upright` | Nothing to see |
| Amber | `falling` | A signal is holding; the caption counts up to `confirm_seconds` |
| Red, heavier box | `fallen` | Confirmed, alert raised |
| Cyan | `recovering` | Back up, still being watched until `recover_seconds` |

```yaml
visualization:
  show_track_ids: on         # "#7" in the caption
  show_labels: on            # the state: upright / falling / fallen
  show_scores: on

  banner:
    enable: on
    alpha: 0.75              # translucent, so it never hides what it points at
    bg_color: [56, 56, 255]
    text_color: [255, 255, 255]
```

The banner is a full-width strip across the bottom while anyone is down. A red box around
one person is easy to miss on a wall of camera tiles; a band across the frame is not.
Everything else — caption sizes, the FPS badge, `auto_scale` — behaves exactly as in the
[detector's overlay](detect.md#the-overlay).

## Configuration

Everything lives in `config.yaml`.

```yaml
model:
  path: assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz
  family: yolo26                       # the ordinary detect head

decode:
  score_threshold: 0.35                # raise it to cut false tracks

tracking:
  classes: [person]                    # only these can fall
  iou_threshold: 0.30
  max_age: 30                          # frames a track survives an occlusion
  min_hits: 3

fall:
  enable: on
  aspect_ratio: 1.20
  height_drop: 0.55
  descent_rate: 0.55
  confirm_seconds: 1.5                 # tune this first
  recover_seconds: 3.0
  min_box_height: 0.08

alerts:
  enable: off
  dry_run: on
  # ... see "Set up email alerts" above

output:
  video: { enable: true, path: falls.mp4 }
  save:  { enable: true, dir: frames, every: 1 }
```

| Mistake | What happens |
|:--|:--|
| Password in `config.yaml` | There is no key for it. Use `$FALL_ALERT_SMTP_PASSWORD` |
| `tracking.classes: []` | Tracks all 80 COCO classes, so a fallen *chair* alerts you |
| `tracking.history_seconds` < `fall.descent_window` | Refused at startup: the descent test could never see far enough back |
| `ssl: on` **and** `starttls: on` | Refused at startup. 465 uses one, 587 the other |
| `alerts.enable: on` with no `to:` | Refused at startup |
| A `.mp4` source | Hits a [demuxer bug](setup.md#video-must-be-raw-h264). Convert to `.h264` |

## Daily loop

| Task | Command | Run in |
|:--|:--|:--|
| Validate the config, no hardware | `sima-vision fall --validate` | anywhere |
| Check the mail settings | `sima-vision fall --test-alert` | DevKit |
| Push the app to the board | `pip install sima-vision` | WSL |
| Run the app | `sima-vision fall` | DevKit |
| Pull the video back | `scp sima@<devkit-ip>:~/falls.mp4 .` | WSL |
| Pull the alert snapshots | `scp -r sima@<devkit-ip>:~/alerts .` | WSL |
| Kill an orphaned run | `pkill -f sima-vision` | DevKit |

## Questions people ask

<details>
<summary><b>Gmail rejects the login</b></summary>

Gmail will not accept your account password over SMTP. Turn on 2-factor authentication,
create an [App Password](https://myaccount.google.com/apppasswords), and export that:

```bash
export FALL_ALERT_SMTP_PASSWORD='abcd efgh ijkl mnop'
```

Then `sima-vision fall --test-alert` with `dry_run: off`. The error it prints is the
real SMTP error, which is usually enough to say whether it is the password, the port or
the TLS mode.

</details>

<details>
<summary><b>It alerts when someone crouches or sits down</b></summary>

Expected on a box-only detector: a crouching person really is short and wide-ish. In
order of what to try:

1. Raise `fall.confirm_seconds`. Most crouches are brief; a fall is not.
2. Raise `fall.aspect_ratio` toward 1.4, so only a properly horizontal box counts.
3. Raise `fall.height_drop` so a partial crouch no longer counts as collapsed.

Sitting on the floor is genuinely ambiguous from a bounding box. If you must separate the
two, that is the point at which a pose model earns its cost.

</details>

<details>
<summary><b>Can I use a camera or an RTSP stream?</b></summary>

Yes, exactly as the other apps:

```yaml
source:
  type: rtsp
  uri: "rtsp://user:pass@camera.local/stream1"
```

Leave `preprocess.input_format: NV12` and `runtime.preset`/`overflow_policy` on `auto`,
which switches a live source to keep-latest so it stays current. Live is also where the
timestamps matter most: the app uses stream time, so the descent threshold means the same
thing as it did on your test file.

</details>

<details>
<summary><b>How many cameras can one board watch?</b></summary>

This app is one stream per process, and one process holds the MLA. For several cameras,
run the app once per board, or extend the graph to a multi-stream source — the tracker
and the fall rules are per-track already, so the state machine needs no changes; the
plumbing does.

</details>

<details>
<summary><b>Does anything leave the building?</b></summary>

Only the alert email, and only when `alerts.enable: on` and `dry_run: off`. Video and
snapshots are written to the DevKit's own filesystem. The snapshot attached to an alert
is an annotated frame, so it contains identifiable images of people — worth checking
against your site's privacy policy before pointing it at a public mall, and worth setting
`attach_snapshot: off` if the answer is no.

</details>

<details>
<summary><b>Why not a pose model?</b></summary>

Keypoints would give a torso angle and separate sitting from lying convincingly, and
`BoxDecodeType` does expose `YoloV26Pose`. It is the natural upgrade. This app uses the
detect head because it runs on the model pack you already have from the detector app, and
because box aspect plus descent gets most of the way there. If you have a pose pack, the
signals in `fall_signals()` are the only thing that would need rewriting.

</details>

## Common errors

Problems with a **running fall detector**. Bring-up problems are in the
[root README](setup.md#setup-errors).

| Symptom | Fix |
|:--|:--|
| `alerts.username is set but $FALL_ALERT_SMTP_PASSWORD is empty` | Export it in the shell that runs the app, or set `dry_run: on` |
| `tracking.classes has unknown class` | Typo. The error suggests near matches from the labels file |
| `tracking.history_seconds ... is shorter than fall.descent_window` | Raise `history_seconds` |
| `alerts.smtp.ssl and starttls are both on` | 465 uses `ssl`, 587 uses `starttls`. Never both |
| `alerts.to must list at least one recipient` | Fill in `to:`, or set `dry_run: on` |
| `[warn] alert send failed: ...` | The SMTP error is printed verbatim. `--test-alert` reproduces it synchronously |
| `falls=0` on footage that has a fall | Lower `fall.confirm_seconds` and `fall.aspect_ratio`; check the person is being tracked at all |
| Alerts fire constantly | Raise `fall.confirm_seconds` first, then `alerts.cooldown_seconds` |
| Ids change every few frames | Lower `tracking.iou_threshold`, raise `tracking.max_age` |
| `model archive not found` | `sima-cli login`, then run again -- it fetches the pack itself. Check `find assets -type f` |
| `is not a raw H.264 elementary stream` | You renamed a `.mp4` instead of converting it |
| Device busy | Orphaned run: `ssh sima@<ip> pkill -f sima-vision` |
| Recording is a few frames long | See the [detector's note](detect.md#questions-people-ask); usually `output.insight` |

---

## License

The detection model used here for testing is **Ultralytics YOLO26**, released under
**AGPL-3.0**. All other parts of this code are released under **Apache-2.0**.

## Credits

- [SiMa.ai on GitHub](https://github.com/SiMa-ai): Modalix, the Palette SDK and Neat
- [Ultralytics](https://github.com/ultralytics/ultralytics): YOLO26 models

<div align="center">

Created with ❤️ by **Muhammad Rizwan Munawar**, passionate about implementing
computer vision ideas and sharing my gains with the community.

If this saved you an afternoon, **⭐ the repo** and pass it on to someone else
bringing up a DevKit.

<br>

<a href="https://github.com/RizwanMunawar"><img src="../assets/socials/github.svg" width="50" alt="GitHub"></a>
&nbsp;&nbsp;
<a href="https://www.linkedin.com/in/muhammadrizwanmunawar/"><img src="../assets/socials/linkedin.svg" width="50" alt="LinkedIn"></a>
&nbsp;&nbsp;
<a href="https://x.com/muhammdrizwanmr"><img src="../assets/socials/x.svg" width="50" alt="X"></a>
&nbsp;&nbsp;
<a href="https://www.youtube.com/@muhammadrizwanmunawar"><img src="../assets/socials/youtube.svg" width="50" alt="YouTube"></a>
&nbsp;&nbsp;
<a href="https://muhammadrizwanmunawar.medium.com/"><img src="../assets/socials/medium.svg" width="50" alt="Medium"></a>

</div>
