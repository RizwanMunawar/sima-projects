# YOLO Object Detector — SiMa Neat (2.1.2_Palette_SDK)

Video-file / RTSP / camera YOLO detection on Modalix, with annotated frames written
to disk **and** a live Neat Insight feed (H.264 RTP/UDP video + JSON metadata).

Built by following the `neat-application-builder` playbook. Every pyneat call was
verified against the packaged core source in this SDK image, not from memory:

| What | Source of truth |
| --- | --- |
| `PreprocessOptions` fields | `/neat-resources/core-src/include/model/PreprocessPlan.h` |
| `ModelOptions` fields | `/neat-resources/core-src/include/model/Model.h` |
| `BoxDecodeType` members | `/neat-resources/core-src/include/pipeline/BoxDecodeType.h` |
| Preproc semantics | `/neat-resources/core-src/docs/reference/nodes/preproc.mdx` |
| BBOX wire payload | `/neat-resources/core-src/docs/reference/boxdecode_decode_types.md` |
| Insight senders | `docs/develop-apps/advanced-concepts/application-design/{video,metadata}_sender.md` |
| Python enum names | `/neat-resources/core-src/python/src/module.cpp` |
| Reference implementation | `/neat-resources/apps-src/examples/object-detection/single-stream-object-detector` |

## Layout

```
object-detection/
├── config.yaml              # everything is configured here
├── README.md
└── src/
    ├── main.py              # the pipeline
    ├── coco_labels.txt      # 80 COCO classes
    └── requirements.txt
```

## API shape

The playbook's decision map puts this on `Graph`, not `Model.run(...)`: there are
multiple stages, named public endpoints, and a branch/fan-in.

```
source ──> branch ──> frame ───────────────┐
                │                          ├─> combine("detector_output")
                └──> model ──> detections ─┘
```

`detector_output` is pulled from the `Run` handle, the BBOX payload is parsed, and
the frame fans out to the disk writer, the `MetadataSender`, and a second small
Graph (`Input -> VideoSender`) that encodes and streams to Insight.

## Pre-processing

`preprocess:` in `config.yaml` is the **intent layer** — `pyneat.ModelOptions.preprocess`.
`Model` resolves it against the model archive's MPK contract and compiles the matching
Preproc / Quant / Tess / QuantTess graph. Anything left on `auto` is the planner's call.

Applied in [`apply_preprocess_options()`](src/main.py):

| Config key | pyneat field | Notes |
| --- | --- | --- |
| `kind` | `preprocess.kind` | `image` for every source here — all three produce decoded pixels. |
| `enable` | `preprocess.enable` | Master switch. `off` skips preprocess entirely. |
| `input_format` | `color_convert.input_format` | **Must match what the source hands to Preproc.** |
| `output_format` | `color_convert.output_format` | `auto` takes it from the model contract (RGB for YOLO packs). |
| `input_max_width/height` | `input_max_width/height` | Preproc buffer capacity. `0` here → filled from the probed stream size (the field's own default is 1920×1080). |
| `resize.mode` | `resize.mode` | `letterbox` for YOLO — preserves aspect ratio and pads. |
| `resize.width/height` | `resize.width/height` | `0` = infer from the model input contract (640×640 for most YOLO). |
| `resize.pad_value` | `resize.pad_value` | `114`, the YOLO convention. |
| `resize.scaling_type` | `resize.scaling_type` | `BILINEAR`, `NEAREST_NEIGHBOUR`, `BICUBIC`, `INTERAREA`, `NO_SCALING`. |
| `normalize.preset` | `preprocess.preset` | `coco_yolo` for every YOLO detector. |
| `normalize.mean/stddev` | `normalize.mean/stddev` | Only read when `preset: none`. |
| `quantize.*`, `tessellate.*` | `quantize`, `tessellate` | Leave on `auto`. A `-mla_tess-` pack already implies tessellation. |

### The one setting that matters most

**`input_format` must match the source.** Getting this wrong is the usual cause of
"the model runs but detects nothing":

| Source | `input_format` |
| --- | --- |
| `type: video` (hardware H.264 decode) | `NV12` |
| `type: rtsp` (hardware H.264/MJPEG decode) | `NV12` |
| `type: usb` (libcamera) | `NV12` |
| `cv2.imread()` images | `BGR` |

The default config ships `NV12`, correct for all three sources here.

### Coordinate mapping is automatic

Preproc writes resize/letterbox metadata (`original_*`, `scaled_*`, `pad_*`, `affine_*`)
onto the tensor, and BoxDecode reads it back. **Boxes come out in original-image
pixels already** — do not undo the letterbox yourself. If boxes look shifted or
scaled, the cause is `resize.mode` or `pad_value`, not missing inverse math.

## Model family → decode type

`model.family` maps to `pyneat.BoxDecodeType`:

| `family` | `BoxDecodeType` |
| --- | --- |
| `yolo` | `Yolo` |
| `yolov5`, `yolov5-seg` | `YoloV5`, `YoloV5Seg` |
| `yolov6` | `YoloV6` |
| `yolov7`, `yolov7-seg` | `YoloV7`, `YoloV7Seg` |
| `yolov8`, `yolov8-seg`, `yolov8-pose` | `YoloV8`, `YoloV8Seg`, `YoloV8Pose` |
| `yolov9`, `yolov9-seg` | `YoloV9`, `YoloV9Seg` |
| `yolov10`, `yolov10-seg` | `YoloV10`, `YoloV10Seg` |
| **`yolo11`, `yolo11-seg`, `yolo11-pose`** | **`YoloV8`, `YoloV8Seg`, `YoloV8Pose`** |
| `yolo26`, `yolo26-seg`, `yolo26-pose` | `YoloV26`, `YoloV26Seg`, `YoloV26Pose` |
| `yolox` | `YoloX` |

> **YOLO11 has no `BoxDecodeType` of its own.** `BoxDecodeType` in this SDK goes
> v5, v6, v7, v8, v9, v10, v26, X — there is no v11 member. Ultralytics YOLO11
> exports the same decoupled DFL detect head as YOLOv8, so `yolo11` maps to
> `BoxDecodeType.YoloV8`. Verify against your own export: if scores come back
> uniformly near-zero, the head format does not match the decode family.

YOLOX and YOLOv6 use raw/logit-style heads — do not decode them as probability-only
YOLO heads.

Note: `-seg` and `-pose` families decode, but this app only renders the leading
boxes. Full mask/keypoint rendering needs `pyneat.decode_segmentation(...)` /
`pyneat.decode_pose(...)` in the frame handler.

## Get a model

```bash
mkdir -p assets/models && cd assets/models
sima-cli download https://docs.sima.ai/pkg_downloads/SDK<platform-version>/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz
cd ../..
```

Other batch-1 YOLO26 packs: `yolo26{n,s,l,x}-det-bf16-mla_tess-b1.tar.gz`,
`yolo26m-det-bf16-b1.tar.gz`, `yolo26m-det-int8-b1.tar.gz`.

Then point `model.path` at it and set `model.family` to match.

## Deploy and run

The app runs **on the Modalix DevKit** — `pyneat` ships as an aarch64 wheel
(`/opt/toolchain/aarch64/modalix/neat-install-packages/pyneat-0.3.0-cp311-cp311-linux_aarch64.whl`)
and will not import in the x86 SDK container.

Copy the project into the SDK workspace, which is NFS-exported to the DevKit:

```bash
cp -r object-detection /root/workspace/
```

On the DevKit:

```bash
source ~/pyneat/bin/activate
pip install -r object-detection/src/requirements.txt
python3 object-detection/src/main.py --config object-detection/config.yaml
```

Validate the config first — this parses and checks everything without touching
pyneat or the MLA, so it also runs in the SDK container:

```bash
python3 src/main.py --config config.yaml --validate-config
```

### Stopping cleanly

`dk` / `devkit-run` execute over SSH **without a pty**, so a terminal Ctrl-C is not
forwarded and the app can be orphaned on the DevKit still holding the MLA and
streaming UDP. For interactive runs:

```bash
ssh -tt <devkit> 'cd ~/workspace/object-detection && python3 src/main.py --config config.yaml'
```

`main.py` installs `SIGINT`/`SIGTERM`/`SIGHUP` handlers that close the `Run` and the
video `Run` on the way out. If a run is ever orphaned:

```bash
ssh <devkit> pkill -f 'src/main.py'
```

`dk`/`devkit-run` is still the right tool for short bounded smoke runs
(`runtime.frames: 100`) that exit on their own.

## Neat Insight

Ports follow the base + channel rule: video `9000 + channel`, metadata `9100 + channel`.
Set `output.insight.host` to the machine running the Insight receiver. If Insight sits
behind container port remapping, pass the mapped host and matching port bases.

Metadata is UTF-8 JSON with the `object-detection` type:

```json
{"type":"object-detection","timestamp":12345,"frame_id":"7",
 "data":{"objects":[{"id":"obj_1","label":"car","confidence":0.92,"bbox":[120,80,96,64]}]}}
```

`bbox` is `[x, y, width, height]` in original-image pixels. Sends are nonblocking
(`MSG_DONTWAIT`) — a congested socket drops that packet rather than stalling the
inference loop. `metadata: sent=… failures=… would_block=…` prints at shutdown.

## Tuning

| Symptom | Check |
| --- | --- |
| No detections at all | `model.family` vs the actual exported head; then `decode.score_threshold`. |
| Boxes shifted or scaled | `preprocess.resize.mode` and `pad_value`. Do not add your own inverse transform. |
| `expected a BBOX tensor but got …` | The route returned raw heads — the model archive did not include BoxDecode at that point. |
| Scores uniformly near zero | Head format mismatch. YOLOX/v6/v26 are raw-logit heads. |
| Dropped frames on live sources | `runtime.queue_depth`, `runtime.overflow_policy: keep_latest`. |
| Slow | `runtime.profile: true` for per-stage `pull` / `decode` / `sinks` timings. |
| USB camera not found | `CameraInput` is libcamera-backed. Run `cam -l` on the DevKit and put the exact name in `source.usb.camera_name`. |

For graph build failures, read the structured diagnostics in order: `error_code`,
`repro_note`, the first terminal entry in `bus`, then `repro_gst_launch`.

## Validation status

Verified in the SDK container:

- config parsing, validation, and every error path (bad family, resize mode, scaler,
  colour format, normalize preset, tri-state flag, threshold range, no-sink)
- `family` → `BoxDecodeType` mapping for v5 / v8 / v11 / v26
- all pyneat enum member names checked against `python/src/module.cpp`

**Not yet verified:** runtime behaviour. Graph build, MLA inference, BBOX decode,
and the Insight feed need a run on the DevKit with a real model archive — a
container-side check is not proof of Modalix runtime behaviour.
