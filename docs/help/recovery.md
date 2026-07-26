# Recovery

Two situations where the DevKit ends up in a state the normal setup path cannot fix.

## DevKit firmware version mismatch

```
ERROR: DevKit/SDK version mismatch. DevKit 2.0.0, SDK 2.1.2
```

New boards often ship older firmware, so hitting this is normal rather than a sign
something went wrong.

!!! danger "eLxr firmware cannot be updated remotely"

    Pushing the update from your PC with `--ip` is rejected outright:

    ```
    ELXR does not support remote update.
    Please connect the DevKit to the Internet and run:  sima-cli update
    ```

    `--dryrun` is on-device only as well. **The update runs on the board.**

### Step 1. Give the board internet access

```
   Internet ──> Wi-Fi ──> [ Windows ICS ] ──> Ethernet ──> DevKit
                           192.168.137.1                 <devkit-ip>
```

If you followed [step 1](../setup/hardware.md), Windows Internet Connection Sharing is
already doing this. Sharing `192.168.137.1` is precisely how the board received its
address, so this usually needs no work.

```powershell title="PowerShell"
Get-Service SharedAccess | Select-Object Name, Status
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Services\SharedAccess\Parameters' | Select-Object ScopeAddress
```

!!! success "Exit criteria"

    `Status = Running` and `ScopeAddress = 192.168.137.1`.

If ICS is off: **Network Connections → right-click your Wi-Fi adapter → Properties →
Sharing → allow sharing → select Ethernet**. Or plug the board into a router with
internet.

### Step 2. Update, on the board

On eLxr this is an **APT package upgrade** driven by `simaai-ota`, not a monolithic
firmware flash.

```bash title="DevKit"
ip route
ping -c 2 8.8.8.8
sima-cli login
sima-cli update --dryrun
```

A healthy dry run ends:

```
ELXR APT channel already set to external release.
ELXR dry run complete. Would run: sudo /usr/bin/simaai-ota -f -o
No ELXR update was applied.
```

!!! warning "`No ELXR update was applied` is the correct ending of a dry run"

    Nothing has changed yet. Run the command again **without** `--dryrun`.

```bash title="DevKit"
sima-cli update
```

Two prompts appear:

| Prompt | What to do |
| :-- | :-- |
| Update menu | Choose **"Update all packages to the latest"** |
| `[sudo] password for sima` | The **same password you SSH in with** |

Budget 15 to 40 minutes for a few hundred packages plus a reboot.

!!! danger "Do not interrupt power or the network while it runs"

    Also **assume the board's home directory does not survive**. `~/pyneat`,
    `~/object-detection`, your model and video may all be gone afterwards. That is
    fine, you re-pair below and re-copy the app. The rule it teaches is worth keeping:
    never leave the only copy of anything on the DevKit.

### Step 3. Confirm and re-pair

```bash title="WSL"
ssh sima@<devkit-ip> "cat /etc/buildinfo | head -5"
sima-cli sdk setup --devkit <devkit-ip>
```

??? note "Still on the old version afterwards?"

    The APT release channel does not carry the version you need. Two options:

    1. **[Net Boot Recovery](https://developer.sima.ai/hardware/getting-started/firmware-update/net-boot)**
       TFTP boots the board from your host and flashes eMMC directly.
    2. **Move the SDK instead of the board.** Compatibility only requires the two to
       agree, not that either be newest.

??? note "SSH complains the host key changed"

    Expected after a reflash, and not a security problem:

    ```bash
    ssh-keygen -R <devkit-ip>
    ```

    Any SSH key that pairing installed earlier is likely gone too, so expect password
    prompts until you re-pair.

---

## pyneat missing on the DevKit

Means pairing never installed it, almost always because networking was not fixed first.
Re-run pairing from **WSL** now that it works:

```bash title="WSL"
sima-cli sdk setup --devkit <devkit-ip>
```

Still missing? Install by hand on the board. Match the version from `neat` in the
container, and install under `/media/nvme` because the root filesystem is too small:

```bash title="DevKit"
sudo mkdir -p /media/nvme/neat && sudo chown "$USER:$USER" /media/nvme/neat
cd /media/nvme/neat
sima-cli login
sima-cli neat install core@v0.3.0
```

```bash title="DevKit"
source ~/pyneat/bin/activate
python3 -c "import pyneat; print(pyneat.__version__)"
```

!!! warning

    `sima-cli` downloads into the **current directory**, so `cd` somewhere you own
    first. Running the install straight from `/media/nvme` gives
    `Current directory '/media/nvme' is not writable`.

    `-t pyneat` fetches only the wheel, which is not enough to run an application. The
    full `core` install also brings the runtime and the GStreamer plugins.
