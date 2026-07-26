# Running the app

The detector lives in `object-detection/` in the repo.

```
object-detection/
├── config.yaml          # every setting lives here
├── assets/
│   ├── models/          # .tar.gz model packs  (not in git)
│   └── video/           # .h264 streams        (not in git)
└── src/
    ├── app.py           # the pipeline
    ├── coco_labels.txt  # 80 COCO class names
    └── requirements.txt
```

Models and video are gitignored, so after cloning you supply your own.

| Page | What it covers |
| :-- | :-- |
| [Deploy and run](deploy.md) | Copying to the board and starting the app |
| [Seeing the results](results.md) | The recorded video and the live Insight feed |
| [Configuration](configuration.md) | Every setting in `config.yaml` |
| [How it works](internals.md) | Pipeline shape, preprocessing, decode types |
| [Daily loop](daily-loop.md) | The edit, copy, run cycle |

## The loop

```
   edit config or code   ──►   scp to the board   ──►   run   ──►   watch
          ▲                                                            │
          └────────────────────────────────────────────────────────────┘
```

Everything runs **on the DevKit**. `pyneat` is compiled for the board's ARM processor
and will not import on your PC.
