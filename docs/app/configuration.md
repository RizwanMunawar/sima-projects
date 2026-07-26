# Configuration

Everything lives in `object-detection/config.yaml`. Paths are **DevKit paths**, relative
to where you launch `app.py`.

## The settings that matter

```yaml
model:
  path: assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz
  family: yolo26                       # must match your model

source:
  type: video                          # video | rtsp | usb
  uri: assets/video/video-loop.h264    # DevKit path, .h264 only
  fps: 25
  width: 1920
  height: 1080

output:
  video:
    enable: true
    path: detections.mp4               # annotated video, written on the DevKit
    hud: true                          # small FPS badge
  insight:
    annotated: true                    # Insight shows our overlay
    host: 192.168.137.1                # NOT 127.0.0.1
```

## Five ways to get it wrong

| Mistake | What actually happens |
| :-- | :-- |
| `uri: C:\Users\...\video.mp4` | The DevKit has no `C:` drive. Use a Linux path |
| `uri: r"C:\path\file.mp4"` | `r"..."` is Python syntax. YAML keeps the `r` and both quotes as part of the filename |
| `host: 127.0.0.1` | From the board that means "the board itself". Video goes nowhere |
| `family` mismatched | Detections come back empty, or every score sits near zero |
| A `.mp4` source | Hits a [demuxer bug](../help/known-issues.md). Convert to `.h264` |

---

## Full reference

### `model`

| Key | Meaning |
| :-- | :-- |
| `path` | Compiled model archive on the DevKit |
| `labels` | Newline-separated class names |
| `family` | Detection head. See [decode types](internals.md#model-family-to-decode-type) |
| `decode_type_option` | Head packing override. Leave `auto` for SiMa model packs |
| `num_classes` | `0` reads it from the archive |

### `source`

| Key | Meaning |
| :-- | :-- |
| `type` | `video`, `rtsp` or `usb` |
| `uri` | File path or stream URL, as seen on the DevKit |
| `fps`, `width`, `height` | `0` probes. Raw `.h264` cannot be probed, so set them |
| `rtsp.codec` | `h264` or `mjpeg` |
| `rtsp.tcp` | TCP transport for RTSP |
| `rtsp.latency_ms` | Jitter buffer latency |
| `usb.camera_name` | libcamera device name, empty for the default |

### `preprocess`

An **intent layer**, not instructions. The route planner resolves it against the model
archive and compiles the matching graph. `auto` means the planner decides. Full detail
in [How it works](internals.md#preprocessing).

| Key | Meaning |
| :-- | :-- |
| `kind` | `image`, `tensor` or `auto` |
| `enable` | Master switch |
| `input_format` | **Must match what the source produces.** `NV12` for video, rtsp and usb |
| `output_format` | `auto` takes it from the model contract |
| `input_max_width/height` | Buffer capacity. `0` uses the probed source size |
| `resize.mode` | `letterbox` for YOLO, or `stretch` / `crop` |
| `resize.width/height` | `0` infers from the model, usually 640x640 |
| `resize.pad_value` | `114`, the YOLO convention |
| `resize.scaling_type` | `BILINEAR`, `NEAREST_NEIGHBOUR`, `BICUBIC`, `INTERAREA`, `NO_SCALING` |
| `normalize.preset` | `coco_yolo` for every YOLO detector |
| `normalize.mean/stddev` | Only read when the preset is `none` |
| `quantize`, `tessellate` | Leave on `auto` |

### `decode`

| Key | Meaning |
| :-- | :-- |
| `score_threshold` | Minimum confidence. `0.0` keeps the packaged value |
| `nms_iou` | Overlap threshold for non-max suppression |
| `max_detections` | Top-K cap per frame |

### `runtime`

| Key | Meaning |
| :-- | :-- |
| `frames` | `0` runs until interrupted |
| `pull_timeout_ms` | How long to wait for a frame before giving up |
| `queue_depth` | Runtime queue depth |
| `preset` | `auto` resolves to `realtime` |
| `overflow_policy` | `auto` resolves to `keep_latest` |
| `profile` | Per-stage timings |

!!! danger "Do not set `overflow_policy: block`"

    Every stage in the graph applies backpressure. Forbidding drops deadlocks it, and
    the run produces **zero** frames before the first pull times out. Full-length output
    comes from the non-blocking Insight push, not from refusing to drop.

### `output`

| Key | Meaning |
| :-- | :-- |
| `video.enable` | Write an annotated video on the DevKit |
| `video.path` | Relative to the launch directory |
| `video.codec` | 4-char FourCC. Falls back to MJPG/`.avi` automatically |
| `video.fps` | `0` matches the source |
| `video.hud` | Small FPS badge in the corner |
| `save.enable` | Write annotated stills |
| `save.every` | Write every Nth frame. `0` disables |
| `insight.enable` | Stream to Neat Insight |
| `insight.annotated` | `true` sends our overlay, `false` sends the raw frame |
| `insight.host` | Your PC as the board sees it |
| `insight.channel` | Added to both port bases |
| `insight.bitrate_kbps` | H.264 encoder bitrate for the preview |

---

## Tuning

| Symptom | Setting |
| :-- | :-- |
| Missing detections | Lower `decode.score_threshold` |
| Duplicate boxes | Lower `decode.nms_iou` |
| Too many detections | Raise `decode.score_threshold`, lower `max_detections` |
| Dropped frames | Raise `runtime.queue_depth` |
| Want timings | `runtime.profile: true` |
