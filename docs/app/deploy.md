# Deploy and run

## Prepare the board, once

```bash title="DevKit"
pip install -r ~/object-detection/src/requirements.txt
```

!!! danger "Never let pip pull numpy 2.x"

    `pyneat` and every `simaai-*` package need `numpy<2`. The pins in
    `requirements.txt` handle it, but an unpinned install produces this, and it is easy
    to scroll past because the app still starts:

    ```
    pyneat 0.3.0 requires numpy<2,>=1.24, but you have numpy 2.4.6 which is incompatible.
    ```

    If you already broke it:

    ```bash
    pip install "numpy>=1.24,<2" "opencv-python>=4.7,<5"
    ```

## Your video must be raw H.264

The DevKit decodes H.264 in hardware, and `.mp4` containers hit a
[known bug](../help/known-issues.md). Convert once, losslessly, and loop it while you
are there so there is time to open Insight:

```powershell title="Any machine with ffmpeg"
ffmpeg -stream_loop 9 -i clip.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 video-loop.h264
```

`-c:v copy` remuxes without re-encoding, so it is fast and lossless. Put the result in
`object-detection/assets/video/`.

Raw streams carry no container metadata, so set the geometry explicitly in
`config.yaml`:

```yaml
source:
  uri: assets/video/video-loop.h264
  fps: 25
  width: 1920
  height: 1080
```

## Check the config before deploying

Runs the full parser and validator without needing the board:

```bash title="SDK container"
cd /workspace/object-detection
/opt/sima-cli/venv/bin/python3 src/app.py --config config.yaml --validate-config
```

!!! success "Exit criteria"

    Prints `config OK` along with the resolved family and preprocessing line.

## Copy everything across

One command, run from the repo root in WSL, after every change:

```bash title="WSL"
scp -r object-detection/ sima@<devkit-ip>:~
```

!!! warning

    This overwrites the copy on the board, so the repo in WSL stays the original.
    Mind the IP: it changes between reboots.

First connection asks you to confirm a fingerprint. Type `yes`. Normal, and it only
happens once.

## Run it

```powershell title="PowerShell"
ssh -tt sima@<devkit-ip>
```

```bash title="DevKit"
source ~/pyneat/bin/activate
cd ~/object-detection
python3 src/app.py --config config.yaml
```

A healthy startup prints each stage. Read it, because it confirms several things at
once:

```
source: type=video uri=assets/video/video-loop.h264 stream=1920x1080@24
preprocess: kind=image enable=on in=NV12 out=AUTO ... resize=letterbox ... pad=114
loading model (first load unpacks the archive, this can take a minute)...
model: ...tar.gz family=yolo26 decode_type=YoloV26 labels=80
runtime: preset=reliable overflow=block queue_depth=3
       block keeps every frame, so the run takes longer than the clip.
graph built
insight: host=192.168.137.1 video=9000 metadata=9100 channel=0
video: detections.mp4 codec=mp4v fps=24 hud=True
running. press Ctrl-C to stop.
[50] 24.8 fps, 6.2 detections/frame avg
```

!!! danger "Use `ssh -tt`, with two t's"

    Without a pty, Ctrl-C never reaches the application. It keeps running invisibly on
    the board holding the MLA, and your next run fails with a busy device.

    ```bash
    ssh sima@<devkit-ip> pkill -f src/app.py
    ```

---

Next: [Seeing the results](results.md)
