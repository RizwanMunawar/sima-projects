# SiMa Neat SDK

**Live YOLO object detection on a Modalix DevKit.**

<div class="hero-badges" markdown>

[![SiMa.ai](https://img.shields.io/badge/SiMa.ai-Modalix-000000?style=for-the-badge&labelColor=000)](https://sima.ai)
[![Palette SDK](https://img.shields.io/badge/Palette_SDK-2.1.2-333333?style=for-the-badge&labelColor=000)](https://docs.sima.ai)
[![Neat](https://img.shields.io/badge/Neat-0.3.0-666666?style=for-the-badge&labelColor=000)](https://docs.sima.ai)

![Windows](https://img.shields.io/badge/Windows_11-eaeaea?style=flat-square&logo=windows11&logoColor=000&labelColor=fafafa)
![WSL2](https://img.shields.io/badge/WSL2-eaeaea?style=flat-square&logo=ubuntu&logoColor=000&labelColor=fafafa)
![Docker](https://img.shields.io/badge/Docker-eaeaea?style=flat-square&logo=docker&logoColor=000&labelColor=fafafa)
![Python](https://img.shields.io/badge/Python_3.10+-eaeaea?style=flat-square&logo=python&logoColor=000&labelColor=fafafa)
![YOLO](https://img.shields.io/badge/YOLO_26_·_11_·_v8_·_v5-eaeaea?style=flat-square&labelColor=fafafa)

![Setup](https://img.shields.io/badge/setup-~2_hours-fafafa?style=flat-square&labelColor=fafafa)
![Download](https://img.shields.io/badge/download-12.6_GB-fafafa?style=flat-square&labelColor=fafafa)

</div>

A setup guide and a working detector app, both written while actually bringing up a
DevKit. Every warning in these pages marks somewhere real time was lost.

[Start the setup](setup/index.md){ .md-button .md-button--primary }
[Deploy and run](app/deploy.md){ .md-button }

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
