<div align="center">

<img src="assets/sima-devkit-docs-logo-home.png" alt="SiMa Neat SDK: live YOLO computer vision on a Modalix DevKit 3.0" width="640">

<br>

[![SiMa.ai](https://img.shields.io/badge/SiMa.ai-Modalix_DevKit_3.0-E63946?style=for-the-badge)](https://sima.ai)
[![Palette SDK](https://img.shields.io/badge/Palette_SDK-2.1.2-457B9D?style=for-the-badge)](https://docs.sima.ai)
[![Neat](https://img.shields.io/badge/Neat-0.3.0-2A9D8F?style=for-the-badge)](https://docs.sima.ai)

![Windows](https://img.shields.io/badge/Windows_11-0078D6?style=flat-square&logo=windows11&logoColor=white)
![WSL2](https://img.shields.io/badge/WSL2-E95420?style=flat-square&logo=ubuntu&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics_YOLO26-FFB703?style=flat-square&labelColor=333)

[![PyPI](https://img.shields.io/badge/pip_install-sima--vision-3775A9?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/sima-vision/)
![License](https://img.shields.io/badge/license-Apache--2.0-6C757D?style=flat-square)

</div>

**Live YOLO computer vision on a SiMa Modalix DevKit 3.0** — object detection, instance
segmentation with a background blur, and fall detection with email alerts. One command,
one pipeline, three apps.

Inference runs on the board's MLA. Everything else — checking a config, seeing exactly
what the overlay will look like — runs anywhere, so you can do all of it before you own
any hardware.

## Try it right now

**No DevKit. No SDK. No model. No download.**

```bash
pip install "sima-vision[preview]"
sima-vision preview --task segment -o blur.png
```

<div align="center">
<img src="assets/preview-segment.png" alt="Instance segmentation preview: masks, captions and a blurred background, rendered with no board" width="720">
</div>

That is the **real overlay code** — the same functions that draw the recording on the
board — run against a synthetic scene. No model is involved: the detections are placed
for you so the drawing has something to draw.

```bash
sima-vision preview --task detect       # boxes and labels
sima-vision preview --task fall         # tracked people, states, alert banner
sima-vision preview --task segment --anonymise --keep-classes person
```

## Run it on a DevKit

Three commands on the board. There is nothing to clone and nothing to `scp`.

```bash
pip install sima-vision                 # 1. install

sima-vision fetch detect                # 2. sample clips, and the model command to run

sima-vision detect \
  --source assets/videos/people-walking-outside-mall.h264 \
  --model  assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz
```

`fetch` downloads the two sample clips and prints the one `sima-cli download` line for
the model, which needs a [community.sima.ai](https://community.sima.ai) login and so is
yours to run. Swap `detect` for `segment` or `fall` and everything above is the same.

Out comes `detections.mp4` and a `frames/` directory on the board. Copy them back and
look:

```bash
scp sima@<devkit-ip>:~/detections.mp4 .
```

> **First time with the board?** It has to be brought up once — cabling, WSL2, Docker,
> the 12.6 GB Neat SDK. That is [docs/setup.md](docs/setup.md), about two hours, once per
> machine. Everything on *this* page works before you start it.

## Adjust it

Every setting has a flag for one run, and a config key for every run after.

```bash
sima-vision detect --conf 0.5 --frames 200 --no-save        # this run only
```

```bash
sima-vision init detect        # writes a documented config.yaml here
$EDITOR config.yaml
sima-vision detect             # picks up ./config.yaml on its own
```

`init` writes the same commented file the repo ships — every key, what it does, and what
goes wrong if you get it wrong. Check it without a board, then look at it:

```bash
sima-vision detect --validate           # does it parse, and what did it resolve to?
sima-vision preview --task detect       # what does the overlay look like?
```

## The three apps

| App | What it does | Model | Guide |
|:--|:--|:--|:--|
| `sima-vision detect` | Boxes, class names and confidence on every frame | YOLO26 detect | [docs/detect.md](docs/detect.md) |
| `sima-vision segment` | Per-pixel masks, and a background blur that keeps the subject sharp | YOLO26 segment | [docs/segment.md](docs/segment.md) |
| `sima-vision fall` | Tracks people and emails when one of them goes down | YOLO26 detect | [docs/fall.md](docs/fall.md) |

All three share one pipeline: the same source handling, graph, decoding, drawing and
sinks. Each guide covers only what is genuinely its own — its model, its settings, its
tuning and its errors.

> The command is `sima-vision`, **not** `sima-cli`. `sima-cli` is SiMa.ai's own tool,
> used to log in and download models — a different program, and one you still need.

## Every command

| Command | Needs a board? | What it does |
|:--|:--:|:--|
| `sima-vision preview` | no | Draw the overlay your config produces, to a PNG |
| `sima-vision init <task>` | no | Write a documented `config.yaml` here |
| `sima-vision <task> --validate` | no | Parse and check a config, print what it resolved to |
| `sima-vision doctor` | no | What is installed, and what it lets you do |
| `sima-vision fetch [task]` | no | Download the sample clips, print the model command |
| `sima-vision detect` | **yes** | Run detection on the MLA |
| `sima-vision segment` | **yes** | Run segmentation, with the optional blur |
| `sima-vision fall` | **yes** | Run fall detection, with SMTP alerts |

`sima-vision <command> --help` lists every flag.

<a id="reference"></a>

## Reference

#### Where settings come from

Three layers, each beating the one above it:

```
built-in defaults        a complete, runnable configuration
      ↓
config.yaml              whatever the file sets
      ↓
command-line flags       whatever you typed
```

So a config file is **optional**. These are equivalent:

```bash
sima-vision detect --model assets/models/yolo26m-det.tar.gz \
                   --source assets/videos/mall.h264 --conf 0.4
sima-vision detect --config config.yaml --conf 0.4
```

With no `--config`, the CLI picks up `config.yaml` from the directory you run in, which
is what `sima-vision init` writes. `--no-config` ignores it and runs on defaults plus
flags alone.

#### Common adjustments

The things people actually change, and the two ways to change each. A flag is for one
run; the config key is for every run after it.

| I want | Flag | Config key |
|:--|:--|:--|
| Fewer spurious boxes | `--conf 0.5` | `decode.score_threshold` |
| To catch more, at the cost of noise | `--conf 0.15` | `decode.score_threshold` |
| A short test run | `--frames 100` | `runtime.frames` |
| No video file | `--no-video` | `output.video.enable` |
| No stills | `--no-save` | `output.save.enable` |
| Fewer stills | `--save-every 30` | `output.save.every` |
| To see where the time goes | `--profile` | `runtime.profile` |
| A live view in Neat Insight | `--insight` | `output.insight.enable` |
| A stronger background blur | `--blur-strength 81` | `blur.kernel` |
| A pixelated background | `--blur-method pixelate` | `blur.method` |
| Only people kept sharp | `--keep-classes person` | `blur.keep_classes` |
| People blurred, scene sharp | `--anonymise --keep-classes person` | `blur.invert` |
| No blur, just masks | `--no-blur` | `blur.enable` |
| To track something other than people | `--classes person forklift` | `tracking.classes` |
| Falls confirmed faster | `--confirm 0.8` | `fall.confirm_seconds` |
| An email when someone falls | `--alert-to me@example.com --send` | `alerts.*` |

**When it runs too slowly**, in the order worth trying: `blur.downscale: 4` (the biggest
single win at 1080p), `output.save.every: 30`, `--no-save`, then `blur.feather: 0`.

**When a run stops part-way through a clip**, that is the decoder running out of buffers.
Lower `runtime.output_buffers`, and use `sima-vision segment --minimal` to tell "the app
is too slow" apart from "the graph is wrong" in a single run.

Every one of these can be checked before deploying:

```bash
sima-vision segment --conf 0.5 --blur-strength 81 --validate   # does it parse?
sima-vision preview --task segment --conf 0.5                  # what does it look like?
```

#### Check a config without a board

`--validate` parses everything, resolves it and prints the result. It loads neither
pyneat nor the model, so it runs on your laptop, in the SDK container, anywhere:

```bash
sima-vision segment --config config.yaml --validate
```

```
config OK: config.yaml
  model: assets/models/yolo26m-seg-bf16-mla_tess-b1.tar.gz
  family=yolo26-seg -> BoxDecodeType.YoloV26Seg
  source: type=video uri=assets/videos/people-walking-outside-mall.h264
  decode: conf=0.3 iou=0.6 max_det=50
  segmentation: masks=on source=auto space=auto threshold=0.5 net=<from the first mask>
  blur: background | method=gaussian kernel=41 sigma=auto down=2 feather=9
  output: video=segmentation.mp4 stills=frames/ every=10
```

#### Flags every app takes

| Flag | Config key | What it does |
|:--|:--|:--|
| `--source`, `-s` | `source.uri` | File, RTSP URL, or nothing for the camera |
| `--source-type` | `source.type` | `video`, `rtsp` or `usb` |
| `--model`, `-m` | `model.path` | Compiled model archive |
| `--labels` | `model.labels` | Class names. Defaults to the packaged COCO list |
| `--family` | `model.family` | Detection head. Must match the model |
| `--conf` | `decode.score_threshold` | Minimum confidence |
| `--iou` | `decode.nms_iou` | NMS IoU threshold |
| `--max-det` | `decode.max_detections` | Top-K per frame |
| `--frames`, `-n` | `runtime.frames` | Stop after N frames |
| `--profile` | `runtime.profile` | Per-stage timings |
| `--video` / `--no-video` | `output.video.*` | The annotated recording |
| `--save-dir` / `--save-every` / `--no-save` | `output.save.*` | Annotated stills |
| `--insight` / `--insight-host` | `output.insight.*` | The live Neat Insight feed |
| `--config`, `-c` / `--no-config` | — | Which config file, or none |
| `--validate` | — | Check and print, then exit |

#### Flags per app

```bash
# segment
--blur / --no-blur          --blur-method gaussian|pixelate|none
--blur-strength PX          --keep-classes person car
--anonymise                 # blur the instances instead of the background
--mask-threshold T          --no-masks       --minimal

# fall
--classes person            --confirm S      --no-fall
--alert-to EMAIL...         --alert-from EMAIL
--alerts                    --send           # --send is required to really email
--smtp-host / --smtp-port / --smtp-user      --site NAME
```

`--anonymise --keep-classes person` blurs people and leaves the scene sharp.

Alerts stay a dry run until you pass `--send`, and the SMTP password is only ever read
from `$FALL_ALERT_SMTP_PASSWORD` — never from a config file, which is committed.

#### Drawing on your own image

`preview` normally paints a synthetic scene. Point it at a photo instead and it will
draw over that, which is the quickest way to see how a caption size or a blur reads
against real content:

```bash
sima-vision preview --task fall --source my-photo.jpg
```

Anything OpenCV can open works. Raw `.h264` cannot be — it has no container to parse —
so those fall back to the synthetic scene with a warning.

## License

The models used here for testing are **Ultralytics YOLO26**, released under **AGPL-3.0**.
All other parts of this code are released under **Apache-2.0**.

## Credits

- [SiMa.ai on GitHub](https://github.com/SiMa-ai): Modalix, the Palette SDK and Neat
- [Ultralytics](https://github.com/ultralytics/ultralytics): YOLO26 models

<div align="center">

Created with ❤️ by **Muhammad Rizwan Munawar**, passionate about implementing
computer vision ideas and sharing my gains with the community.

If this saved you an afternoon, **⭐ the repo** and pass it on to someone else
bringing up a DevKit.

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
