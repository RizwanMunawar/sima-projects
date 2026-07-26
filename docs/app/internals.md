# How it works

Built by following the `neat-application-builder` playbook. Every pyneat call was
verified against the packaged core source inside the SDK container, not written from
memory.

| What | Source of truth |
| :-- | :-- |
| `PreprocessOptions` fields | `include/model/PreprocessPlan.h` |
| `ModelOptions` fields | `include/model/Model.h` |
| `BoxDecodeType` members | `include/pipeline/BoxDecodeType.h` |
| Preproc semantics | `docs/reference/nodes/preproc.mdx` |
| BBOX wire payload | `docs/reference/boxdecode_decode_types.md` |
| Insight senders | `docs/develop-apps/advanced-concepts/application-design/` |
| Python enum names | `python/src/module.cpp` |
| Reference implementation | `apps-src/examples/object-detection/single-stream-object-detector` |

All relative to `/neat-resources/core-src/` inside the container.

---

## Pipeline

The playbook's decision map puts this on `Graph` rather than `Model.run(...)`, because
there are multiple stages, named public endpoints and a branch with a fan-in:

```
   source --> branch --> frame ----------------+
                   |                           +--> combine("detector_output")
                   +---> model --> detections -+
```

`detector_output` is pulled from the `Run` handle, the BBOX payload is parsed, and the
frame fans out to the video writer, the still writer, the `MetadataSender`, and a second
small graph (`Input -> VideoSender`) that encodes for Insight.

```python title="app.py, abbreviated"
def build_detector_graph(cfg, model, width, height, fps):
    """source -> branch -> {frame, model -> detections} -> combine."""
    source = make_source_graph(cfg, width, height, fps)
    branch = pyneat.graphs.branch("source", ["frame", "model"])

    frame_graph = pyneat.Graph("frame")
    frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)

    joined = pyneat.graphs.combine(
        ["frame", "detections"], "detector_output", pyneat.CombinePolicy.ByFrame
    )

    graph = pyneat.Graph("object_detection")
    graph.connect(source, branch)
    graph.connect(branch, frame_graph)
    graph.connect(branch, model_graph)
    graph.connect(frame_graph, joined)
    return graph
```

The annotated frame is rendered **once** per iteration and shared across every sink that
wants it, so three outputs cost one draw.

```python title="app.py, the run loop"
annotated = render_annotated(cfg, pipeline, frame, boxes, live_fps) if need_annotated else None

push_video(pipeline, sample, annotated if cfg.insight_annotated else frame)
send_metadata(pipeline, sample, boxes)

if pipeline.writer is not None:
    pipeline.writer.write(annotated)      # every processed frame
    pipeline.writer_frames += 1
if need_jpeg:
    save_frame(cfg, processed, annotated if cfg.save_overlay else frame)
```

---

## Preprocessing

The `preprocess:` block is an **intent layer**, not a set of instructions. `Model`
resolves it against the archive's MPK contract and compiles the matching Preproc, Quant,
Tess or QuantTess graph. Anything left on `auto` is the planner's call.

| Config key | pyneat field | Notes |
| :-- | :-- | :-- |
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

### The one that matters most

`input_format`. Getting it wrong is the usual cause of "the model runs but detects
nothing".

| Source | `input_format` |
| :-- | :-- |
| `video`, `rtsp` (hardware H.264 decode) | `NV12` |
| `usb` (libcamera) | `NV12` |
| `cv2.imread` images | `BGR` |

### Coordinate mapping is automatic

Preproc writes resize and letterbox metadata onto the tensor, and BoxDecode reads it
back, so boxes arrive in **original-image pixels**. Do not undo the letterbox yourself.
Shifted boxes mean `resize.mode` or `pad_value` is wrong, not that inverse maths is
missing.

---

## Model family to decode type

| `family` | `BoxDecodeType` |
| :-- | :-- |
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

!!! warning "YOLO11 has no `BoxDecodeType` of its own"

    The enum goes v5, v6, v7, v8, v9, v10, v26, X. Ultralytics YOLO11 exports the same
    decoupled DFL detect head as YOLOv8, so `yolo11` maps to `YoloV8`.

    Verify against your own export: uniformly near-zero scores mean the head does not
    match the decode family.

YOLOX, v6 and v26 use raw logit heads. Do not decode them as probability-only YOLO
heads.

`-seg` and `-pose` families decode, but this app renders only the leading boxes. Masks
and keypoints need `decode_segmentation` or `decode_pose` in the frame handler.

---

## Flow control

| Setting | Resolves to | Why |
| :-- | :-- | :-- |
| `preset: auto` | `realtime` | |
| `overflow_policy: auto` | `keep_latest` | Dropping is the only safe option here |

The Insight preview `appsrc` is created with `block=False`. That matters more than it
sounds: the default is `block=true`, and once the H.264 encoder or UDP egress falls
behind, `push()` stops returning, the run loop never reaches its next `pull()`, the
detector appsinks fill, and the whole graph stalls part-way through the clip.

A refused preview push is counted as a drop and reported at exit. It never ends a run.

!!! danger "`overflow_policy: block` deadlocks the graph"

    Every stage applies backpressure. With nothing allowed to drop, the run never
    reaches steady state and the first pull times out having produced **zero** frames.

---

## The BBOX wire payload

BoxDecode emits one `UInt8` tensor tagged `BBOX` per frame:

```
[uint32 N][RawBox 24B] × N ... trailing padding
RawBox = <iiiifi  →  x, y, w, h, score, class_id
```

Coordinates are in source-image pixels, not normalised and not in the model's
letterboxed space.
