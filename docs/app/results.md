# Seeing the results

Two ways: a file the board writes as it goes, and a live feed you watch in a browser.
The file is the reliable one.

## The recording

Every processed frame is written to an annotated video on the board, so a run always
leaves something to review even if nobody was watching at the time.

```bash title="WSL"
scp sima@<devkit-ip>:~/object-detection/detections.mp4 .
scp -r sima@<devkit-ip>:~/object-detection/sandbox .      # annotated stills
```

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

A plain rectangle in the class colour, a centre dot, and a filled caption directly above
carrying the class name and confidence in white. Captions flip inside the box rather
than clipping off the top. Colours come from a 20-entry palette keyed to class id, so
the same class is always the same colour. Line weight, text size and padding scale with
the frame, so 4K does not get hairlines and 480p does not get slabs.

| Setting | Meaning |
| :-- | :-- |
| `output.video.path` | Where to write, relative to the launch directory |
| `output.video.codec` | 4-char FourCC. `mp4v` by default, auto-falls back to `MJPG`/`.avi` |
| `output.video.fps` | `0` matches the source rate |
| `output.video.hud` | Small FPS badge in the corner. Turn off for clean footage |
| `output.save.every` | Write every Nth frame as a JPEG. `0` disables |

---

## The live feed

<div align="center" markdown>

## [https://localhost:9900](https://localhost:9900)

**select channel 0**

</div>

!!! warning "Ignore the address `neat` prints"

    It reports `https://192.168.137.1:9900`, and that address does **not** work from
    Windows. Connecting to the mirrored interface counts as inbound traffic to the WSL
    VM, so the Hyper-V firewall drops it. `localhost` takes a different path.

    The same substitution applies to the VS Code browser URL. Keep the token, swap the
    host.

!!! important "Open Insight *before* you start the app"

    Video and detections go out over UDP, which buffers nothing. A viewer opened after
    the stream ends sees an empty page, even though everything worked.

    Short clips make this worse. A 12-second video is over before you can switch
    windows, which is why [Deploy and run](deploy.md#your-video-must-be-raw-h264) loops
    the source.

By default Insight receives the **annotated** frame, so the browser view and the
recording look identical. Set `output.insight.annotated: false` to send the raw frame
and let Insight draw its own overlay from the metadata stream instead.

---

## How to tell it is actually streaming

The app prints these on the way out:

```
processed=380
metadata: sent=380 failures=0 would_block=0
insight: dropped 42 preview frames because the feed was busy. The recording is unaffected.
video: wrote 380 frames to detections.mp4 (61.2 MB)
```

| Line | Means |
| :-- | :-- |
| `failures=0` | Every datagram left the board. A blank viewer is a receiving-side problem |
| `dropped N preview frames` | Expected. The preview is best effort, the recording is not |
| `video: wrote N frames` | The authoritative count. Compare against clip length times fps |

If Insight is still blank with `failures=0`, the problem is on your PC: the
[firewall](../setup/wsl.md#4-firewall), or `insight.host` pointing at `127.0.0.1`.

---

## First run: prove the model offline

Debugging the model and the network at once is what makes this stack feel harder than it
is. Set:

```yaml
runtime:
  frames: 100
  profile: true
output:
  save:
    enable: true
  insight:
    enable: false
```

100 frames, annotated JPEGs to disk, per-stage timings, exits on its own, **zero
networking**. If the boxes look right in those images then your model, preprocessing and
decode family are all correct, and anything that breaks afterwards is transport.
