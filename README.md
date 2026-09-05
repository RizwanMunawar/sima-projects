<div align="center">

<img src="assets/sima-devkit-docs-logo-home.png" alt="sima-vision: live YOLO computer vision on a SiMa Modalix DevKit 3.0" width="640">

[![SiMa.ai](https://img.shields.io/badge/SiMa.ai-Modalix_DevKit_3.0-E63946)](https://sima.ai)
[![Palette SDK](https://img.shields.io/badge/Palette_SDK-2.1.2-457B9D)](https://docs.sima.ai)
[![Neat](https://img.shields.io/badge/Neat-0.3.0-2A9D8F)](https://docs.sima.ai)

[![CI](https://github.com/RizwanMunawar/sima-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/RizwanMunawar/sima-projects/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/badge/pip_install-sima--vision-3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/sima-vision/)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-6C757D)](LICENSE)
[![YOLO26](https://img.shields.io/badge/Ultralytics-YOLO26-FFB703&labelColor=333)](https://github.com/ultralytics/ultralytics)

[![Fall detection](https://img.shields.io/badge/Fall-detection-111F68?style=flat-square&labelColor=333)](https://github.com/ultralytics/ultralytics)
[![Segmentation and blur](https://img.shields.io/badge/Segmentation-blur-FF64DA?style=flat-square&labelColor=333)](https://github.com/ultralytics/ultralytics)
[![Object detection](https://img.shields.io/badge/Object-detection-042AFF?style=flat-square&labelColor=333)](https://github.com/ultralytics/ultralytics)

</div>

## Usage

```bash
# 1. Install the SiMa.ai Neat Core
sima-cli login
sima-cli neat install core@v0.3.0

# 2. Install the sima-vision Python package
pip install sima-vision

# 3. Download the YOLO26 detection model
mkdir -p assets/models
sima-cli download \
  https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz \
  -o assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz

# 4. Run YOLO26 object detection on the DevKit
sima-vision detect
```

Steps 1 to 3 are one-time. After them the run is the only command: `sima-vision detect`
finds the Neat runtime itself, puts the board's numpy and OpenCV on the path, fetches a
sample clip if you have not given it one, and prints what it is doing at every stage.

It writes `detections.mp4` and a `frames/` directory beside itself, on the board.

## Additional commands

The other two apps. They run exactly like `detect` and share its clip and settings:

```bash
# Instance segmentation, with an optional background blur
sima-vision segment
sima-vision segment --blur --keep-classes person

# Fall detection, with SMTP alerts. Nothing is emailed until you pass --send
sima-vision fall
sima-vision fall --alert-to ops@example.com
```

Your own footage or your own model, as a path or an `https` URL. Video must be raw H.264,
never `.mp4`: the board decodes H.264 in hardware, and a container hits a demuxer bug in
Neat 0.3.0. Convert once, losslessly, with `ffmpeg -i clip.mp4 -c:v copy -bsf:v
h264_mp4toannexb -f h264 clip.h264`.

```bash
sima-vision detect --source my-clip.h264 --model my-model.tar.gz
sima-vision detect --source https://example.com/my-clip.h264
```

Moving files between your PC and the board. Set `SIMA_VISION_DEVKIT` first and neither
needs `--host`:

```bash
sima-vision push my-clip.h264      # host -> DevKit
sima-vision pull                   # DevKit -> host, whatever the run left
sima-vision pull --into results/
```

On a laptop, with no board and no network:

```bash
sima-vision detect --validate      # resolve and check the settings, then stop
```

The flags worth knowing. `sima-vision <command> --help` lists the rest:

| Flag | What it does |
|:--|:--|
| `--frames 200` | Stop after N frames. The quickest way to try something |
| `--conf 0.5` | Raise the confidence floor. Default `0.30` |
| `--no-video` / `--no-save` | Skip the recording or the stills. Together they are the cheapest possible run, which is how you tell a slow app apart from a stalled graph |
| `--quiet` | Warnings, errors and the closing report only |
| `--profile` | Per-stage timings, when a run is slower than it should be |

Settings can also come from a `config.yaml` in the working directory, which is picked up
on its own. Flags win over it, and it wins over the built-in defaults.

## Environment

| Variable | What it does |
|:--|:--|
| `SIMA_VISION_DEVKIT` | The board, as `user@address`, so `push` and `pull` stop asking |
| `SIMA_VISION_ASSETS` | Where clips and models are downloaded. Default `./assets` |
| `SIMA_VISION_PYNEAT` | The `pyneat` virtualenv, when the search does not find it |
| `SIMA_VISION_PYNEAT_INDEX` | A pip index carrying a `pyneat` wheel, if your site publishes one |
| `SIMA_VISION_AUTO_INSTALL` | `0` to look but never install |
| `SIMA_VISION_QUIET` | Non-empty is `--quiet` for every command |
| `SIMA_VISION_COLOR` | `0` or `1` to force colour off or on. `NO_COLOR` also works |
| `FALL_ALERT_SMTP_PASSWORD` | The only place the SMTP password is ever read from |

## Contributing

```bash
git clone https://github.com/RizwanMunawar/sima-projects.git
cd sima-projects
pip install -e ".[dev]"

ruff check sima_vision tests
pytest -q
```

The tests need no board.

## License

The models used here for testing are **Ultralytics YOLO26**, under **AGPL-3.0**. All other
parts of this repository are under **Apache-2.0**. See [LICENSE](LICENSE).

## Credits

- [SiMa.ai](https://github.com/SiMa-ai) for Modalix, the Palette SDK and Neat
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO26 models

<div align="center">

Built by **Muhammad Rizwan Munawar**. If this saved you an afternoon, **star the repo**
and pass it on to someone else bringing up a DevKit.

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
