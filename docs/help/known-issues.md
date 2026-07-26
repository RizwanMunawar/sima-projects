# Known issues

Bugs in the SDK itself, with the workarounds the app already applies.

## `groups.video_input` cannot play `.mp4`

**Affects Neat 0.3.0. Status: open.**

```
gst_parse_launch failed: No src-element named "n1_demux" - omitting link
```

### Root cause

`VideoTrackSelect` builds its fragment from a single variable, so what it emits is
internally consistent:

```cpp
const std::string base = "n" + std::to_string(node_index) + "_demux";
ss << "qtdemux name=" << base << " " << base << ".video_" << idx_;
```

The graph then appends an instance suffix, but the renamer only rewrites `name=<x>`
declarations. `element_names()` reports just `{"n1_demux"}`, so the pad reference
`n1_demux.video_0` is never rewritten:

```
qtdemux name=n1_demux_8   ...   n1_demux.video_0 ! queue
             ^^^^^^^^^^                ^^^^^^^^^
             declared with _8          referenced without it
```

**Any non-empty suffix breaks it**, so reordering graph construction does not help.

### Workaround: drop the container

No demuxer, no bug. `app.py` detects a raw H.264 elementary stream by extension and
builds the source chain by hand, skipping `VideoTrackSelect` entirely:

```
FileInput → H264Parse → Queue → SimaDecode → CapsRaw
```

Recognised extensions: `.h264`, `.264`, `.bin`, `.avc`. Anything else still uses
`groups.video_input` and prints the conversion command.

```bash
ffmpeg -i clip.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 clip.h264
```

`-c:v copy` remuxes without re-encoding, so it is fast and lossless.

### Alternative

`groups.rtsp_decoded_input` builds no demuxer either, and is the path SiMa's own
reference example exercises.

---

## The Insight preview can stall the whole pipeline

**Affects any app that pushes into a second graph. Worked around in `app.py`.**

The Insight video `appsrc` defaults to `block=true`. Once the H.264 encoder or UDP
egress falls behind, `push()` stops returning, the run loop never reaches its next
`pull()`, the detector appsinks fill at `max-buffers=4`, and the whole graph stalls
part-way through the clip.

The symptom is an output video far shorter than the input, with the run reporting an
end-of-stream that never happened.

### Workaround

`app.py` creates the preview `appsrc` with `block=False`, so a push is refused rather
than waiting, and counts the refusal as a dropped preview frame instead of raising.

!!! note "This is separate from `overflow_policy`"

    `overflow_policy` governs the detector graph. The preview `appsrc` is a different
    stage on a different graph, and it is non-blocking regardless, so a stalled viewer
    can never hold up the recording.

---

## Reporting

If you hit something not listed here, the structured diagnostics are worth reading in
order before anything else:

1. `error_code`
2. `repro_note`
3. the first terminal entry in `bus`
4. `repro_gst_launch`
