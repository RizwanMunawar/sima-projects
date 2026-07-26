# 8. Download a model

Start the SDK container from WSL:

```bash title="WSL"
sudo su -
cd sima-projects
source sima/bin/activate
sima-cli sdk neat
```

That drops you into a shell inside the container. From there:

```bash title="SDK container"
sima-cli login
mkdir -p /workspace/assets/models && cd /workspace/assets/models
MODELS=https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/yolo26-detection
sima-cli download $MODELS/yolo26m-det-bf16-mla_tess-b1.tar.gz
ls -la
```

!!! success "Exit criteria"

    The `.tar.gz` appears in the listing, about 63 MB.

Swap `yolo26m` for `n`, `s`, `l` or `x` to trade speed against accuracy. `m` is a good
starting point.

## The shared workspace

One folder, three names. This is how files move between machines.

```
   WINDOWS                              WSL                   SDK CONTAINER
   \\wsl$\Ubuntu\root\workspace   ───   /root/workspace  ───   /workspace
```

## Other model families

The app supports far more than YOLO26. See
[Model family to decode type](../app/internals.md#model-family-to-decode-type) for the
full mapping, including the YOLO11 special case.

---

Setup complete. Next: [Deploy and run](../app/deploy.md)
