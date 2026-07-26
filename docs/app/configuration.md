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
| `preset` | `auto` picks `reliable` for a file, `realtime` for a camera |
| `overflow_policy` | `auto` picks `block` for a file, `keep_latest` for a camera |
| `profile` | Per-stage timings |

!!! important "Leave both on `auto`. A file and a camera want opposites"

    A **file** has no deadline, so `auto` gives it `block`. Backpressure reaches
    `filesrc`, decoding slows to the speed of inference, and every frame survives. The
    run takes longer than the clip, which is correct rather than a stall.

    A **camera** cannot be paused, so `auto` gives it `keep_latest`. Blocking a live
    source only buys unbounded latency.

    Forcing `keep_latest` on a file is what produces a short recording that plays fast.
    Inference is several times slower than decoding, so most frames are discarded, and
    the survivors are still written at the source rate.

### `visualization`

How the overlay looks. Colours are `[B, G, R]`, because OpenCV is BGR, so
`[0, 0, 255]` is red rather than blue.

| Key | Meaning |
| :-- | :-- |
| `box_thickness` | Detection rectangle outline weight |
| `text_scale` | Caption font size multiplier |
| `text_thickness` | Caption stroke weight |
| `text_padding` | Gap between caption text and the edge of its band |
| `centre_dot` | Filled dot at the centre of each box |
| `centre_dot_radius` | Radius of that dot |
| `show_labels` | Class name in the caption |
| `show_scores` | Confidence in the caption |
| `score_decimals` | `2` gives `person 0.57`, `0` gives `person 1` |
| `text_color` | Caption text colour |
| `auto_scale` | Scale every size above with the frame height |
| `reference_height` | The height those sizes are tuned for |

Sizes are written for a 1080p frame. With `auto_scale` on they are multiplied by
`frame height / reference_height`, so 4K does not get hairlines and 480p does not get
slabs. Turn it off to use the numbers literally.

#### `visualization.hud`

The frame-rate badge, drawn when `output.video.hud` is true.

| Key | Meaning |
| :-- | :-- |
| `text_color` | Badge text colour |
| `bg_color` | Badge fill colour |
| `text_scale` | Badge font size. `0` follows `text_scale` above |
| `text_thickness` | Badge stroke weight. `0` follows `text_thickness` above |
| `padding` | Gap between badge text and badge edge. `0` follows `text_padding` above |
| `min_width` | Floor on badge width. `0` fits the text |
| `min_height` | Floor on badge height. `0` fits the text |

`padding` is what sets the badge size. The two minimums can only make it bigger, never
smaller, and the text stays centred on both axes whichever way it is sized.

```yaml title="A larger badge, dark text on white"
visualization:
  hud:
    text_color: [104, 31, 17]
    bg_color: [255, 255, 255]
    text_scale: 1.6
    padding: 18
    min_width: 460
```

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
