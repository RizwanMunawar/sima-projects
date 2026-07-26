# SiMa Neat SDK

**Live YOLO object detection on a Modalix DevKit.**

A setup guide and a working detector app, both written while actually bringing up a
DevKit. Every warning in these pages marks somewhere real time was lost.

---

## Three machines, not one

Almost every problem in this stack comes from running a command in the wrong place.

```
   ┌──────────────┐        ┌──────────────────┐        ┌───────────────┐
   │  WINDOWS PC  │        │   WSL2 · UBUNTU  │        │    DEVKIT     │
   ├──────────────┤        ├──────────────────┤        ├───────────────┤
   │  Chrome      │◄──────►│  sima-cli        │◄──────►│  MLA          │
   │  scp / ssh   │ :9900  │  Docker + SDK    │  UDP   │  your app     │
   │              │        │  Neat Insight    │ 9000/  │               │
   │              │        │                  │ 9100   │               │
   └──────────────┘        └──────────────────┘        └───────────────┘
        viewer                build + receive             inference
```

| Machine | Role |
| :-- | :-- |
| **Windows PC** | Browser, `scp`, `ssh`. Nothing runs here. |
| **WSL2 Ubuntu** | `sima-cli`, Docker, the SDK container, the Insight receiver. |
| **Modalix DevKit** | Your application. All inference happens here. |

Your app runs **on the DevKit**. It sends H.264 video and JSON detections over UDP, and
Insight recombines them in your browser.

---

## Get the code

```bash
git clone https://github.com/RizwanMunawar/sima-projects.git
cd sima-projects
```

Already have a DevKit paired? Go straight to [Deploy and run](app/deploy.md).
Starting from a bare machine? Work through [Setup](setup/index.md) in order.

---

## What you end up with

An annotated video on the board, written frame by frame, plus a live feed you can watch
in a browser while it runs.

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

---

## Six rules that prevent most problems

| # | Rule | Because |
| :-- | :-- | :-- |
| 1 | Networking before pairing | Pairing installs over the network. No route means a silent no-op |
| 2 | Docker before the SDK | The SDK **is** a container |
| 3 | `cd` after `sudo su -` | `-` is a login shell, so it drops you in `/root` |
| 4 | `insight.host = 192.168.137.1` | `127.0.0.1` means the board itself |
| 5 | Raw `.h264`, never `.mp4` | Containers hit a demuxer bug in Neat 0.3.0 |
| 6 | Always `ssh -tt` | Ctrl-C needs a pty to reach the app and release the MLA |
