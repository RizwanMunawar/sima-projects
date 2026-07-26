"""Minimal source-only probe. Run this on the DevKit to isolate video input.

Builds nothing except the source group and one output node, then pulls a single
frame. Use it to tell a source/GStreamer problem apart from a model or graph
problem, and to see the exact pipeline string the group generates.

    python3 src/probe_source.py assets/video/video-4.mp4
    python3 src/probe_source.py rtsp://192.168.137.1:8554/stream

Why this exists: `groups.video_input` names its elements with a per-graph
instance suffix (`n1_demux_8`) but emits the demuxer pad link without it
(`n1_demux.video_0`). GStreamer then fails with:

    gst_parse_launch failed: No src-element named "n1_demux" - omitting link

The suffix number depends on how many Graph objects the process has already
created, so a minimal graph may produce a different one. This probe reports the
suffix it got so you can confirm whether the mismatch is unconditional.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path


def load_deps():
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)
    import numpy as np
    import pyneat

    return np, pyneat


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    uri = argv[0]
    np, pyneat = load_deps()

    is_rtsp = uri.startswith("rtsp://")
    print(f"source: {'rtsp' if is_rtsp else 'file'} {uri}")

    if is_rtsp:
        opt = pyneat.RtspDecodedInputOptions()
        opt.url = uri
        opt.latency_ms = 100
        opt.tcp = True
        opt.insert_queue = True
        opt.decoder_name = "decoder"
        opt.decoder_raw_output = True
        opt.codec = pyneat.RtspCodec.H264
        opt.payload_type = 96
        opt.auto_caps_from_stream = True
        source = pyneat.groups.rtsp_decoded_input(opt)
    else:
        if not Path(uri).exists():
            print(f"[ERR] file not found: {uri}", file=sys.stderr)
            return 2
        opt = pyneat.VideoInputGroupOptions()
        opt.path = uri
        opt.insert_queue = True
        opt.sync_mode = False
        opt.out_format = pyneat.Format.NV12
        source = pyneat.groups.video_input(opt)

    # Deliberately minimal: source plus one output, nothing else.
    graph = pyneat.Graph("probe")
    graph.add(source)
    graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(1)))

    print("--- backend ---")
    try:
        print(graph.describe_backend())
    except Exception as exc:
        print(f"(describe_backend unavailable: {exc})")

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy

    run = graph.build(run_options)
    try:
        sample = run.pull("frame", 20000)
        if sample is None:
            print("[ERR] timed out waiting for a frame", file=sys.stderr)
            return 1
        tensor = sample.tensor if sample.kind == pyneat.SampleKind.Tensor else sample.tensors[0]

        def dim(name):
            value = getattr(tensor, name)
            return int(value() if callable(value) else value)

        print(f"OK: pulled one frame {dim('width')}x{dim('height')} nv12={tensor.is_nv12()}")
        return 0
    finally:
        run.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
