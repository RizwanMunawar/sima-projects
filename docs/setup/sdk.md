# sima-cli, Docker and the SDK

Steps 5 to 7. Everything here runs in WSL.

## 5. Get the code and install sima-cli

Become root **first**. `sudo su -` is a login shell, so it drops you in `/root`.
Cloning after that puts the repo at `/root/sima-projects`, which is why every later
step can just say `cd sima-projects`.

```bash title="WSL"
sudo su -
apt update && apt install -y git python3-venv python3-pip

git clone https://github.com/RizwanMunawar/sima-projects.git
cd sima-projects

python3 -m venv sima
source sima/bin/activate
pip install sima-cli
sima-cli login                  # needs a community.sima.ai account
```

You now have the app in `object-detection/` and the `sima` venv beside it.
**Every command below runs from this directory.**

!!! success "Exit criteria"

    `sima-cli --version` prints 2.1.15 or newer, and login succeeds.

!!! tip "Each new terminal needs three lines"

    ```bash
    sudo su -
    cd sima-projects
    source sima/bin/activate
    ```

---

## 6. Docker Engine and NFS

The Neat SDK **is** a Docker container. There is no separate installer:
`sima-cli install` pulls a 12.6 GB image and `sima-cli sdk neat` runs it. Without
Docker, step 7 fails on its first command.

NFS matters too. The SDK exports your workspace so the board can mount it.

!!! tip

    Docker Desktop is not required and not recommended here. Docker Engine natively
    inside WSL is lighter and avoids the Desktop integration layer.

### Install the packages

From the [official Docker docs](https://docs.docker.com/engine/install/ubuntu/). Note
the `deb822` `.sources` format, which replaced the older one-line `deb [arch=...]` entry
you will still find in most blog posts.

```bash title="WSL"
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo apt install -y nfs-kernel-server nfs-common
```

### Enable systemd so Docker survives a restart

WSL only runs a service manager if you ask for one. Without systemd, Docker has to be
started by hand after every single WSL restart.

```bash title="WSL"
grep -q 'systemd=true' /etc/wsl.conf 2>/dev/null || sudo tee -a /etc/wsl.conf <<'EOF'

[boot]
systemd=true
EOF
```

```powershell title="PowerShell"
wsl --shutdown
```

Reopen WSL, then:

```bash title="WSL"
sudo systemctl enable --now docker
sudo docker run hello-world
```

!!! success "Exit criteria"

    Prints **"Hello from Docker!"**. If you get `Cannot connect to the Docker daemon`,
    the daemon is not running.

??? note "Optional: drop sudo from docker commands"

    ```bash
    sudo usermod -aG docker $USER
    ```

    Then `wsl --shutdown` and reopen WSL. Not needed if you work as root.

---

## 7. Install the Neat SDK

```bash title="WSL"
sudo su -
cd sima-projects
source sima/bin/activate
sima-cli install ghcr:sima-neat/sdk
sima-cli sdk setup --devkit <devkit-ip>
```

The image is **12.6 GB**, so expect 30 to 60 minutes.

### Every prompt it asks

Setup is interactive in more places than the docs suggest.

| Prompt | Answer | Why |
| :-- | :-- | :-- |
| `Some system checks failed. Continue?` | `y` | The Firewall row says *Unverified*, not failed. `sima-cli` cannot inspect the Windows firewall from inside WSL |
| `Select SDK images to start` | Space, then Enter | Usually one entry |
| `Use this workspace?` | `Y` | |
| `Remove and recreate container?` | `Y` | Safe, rebuilds from the same image |
| `Install the Model Compiler extension?` | `Y` | Adds about **9 GB** and up to 15 minutes. Only needed to compile your own models |
| `Install VSCode Extensions?` | `y` | Lowercase. A bare Enter is rejected |
| `Apply passwordless sudo on the DevKit?` | `y` | Required for workspace sync |
| `Enter sudo password` | your DevKit password | |

!!! note "NFS often fails and falls back to rsync. That is fine"

    ```
    mount.nfs: Connection timed out
    WARNING: using rsync fallback: /workspace -> sima@…:/workspace-rsync
    ```

    Setup continues and `dk` still works. The board sees `/workspace-rsync` rather
    than a live mount. This guide copies with `scp` anyway.

### Verify the board half

This is the part that fails quietly, so check it explicitly:

```bash title="WSL"
ssh sima@<devkit-ip> "~/pyneat/bin/python3 -c 'import pyneat; print(pyneat.__version__)'"
```

| Result | Next |
| :-- | :-- |
| Prints a version | Done. Continue to [step 8](model.md) |
| `No such file or directory` | The board half never ran. See [Recovery](../help/recovery.md) |

!!! warning

    `sdk` is a **PC-side** command. Running `sima-cli sdk setup` on the DevKit fails
    with `Error: No such command 'sdk'`, because the board ships a different build of
    the CLI.

### Timing

| Phase | Typical | What you see |
| :-- | :-- | :-- |
| Pull the 12.6 GB image | 20 to 45 min | Docker layer progress bars |
| Requirements check | seconds | Python / Docker / CPU table |
| Container first start | 1 to 3 min | "Starting Neat SDK container…" |
| NFS export | seconds | Little or no output |
| DevKit pairing | 5 to 20 min | Package installs on the board |

??? note "It is not hung if…"

    * Docker is still drawing layer progress. The image really is 12.6 GB.
    * The screen sits on "Starting Neat SDK container" for a couple of minutes.
    * Nothing prints during pairing for several minutes. Packages are installing on
      the board and output is sparse. This is where people hit Ctrl-C too early.

    Watch real progress from a second WSL terminal:

    ```bash
    sudo docker ps
    cat /etc/exports.d/*.exports 2>/dev/null
    ```

---

Next: [Download a model](model.md)
