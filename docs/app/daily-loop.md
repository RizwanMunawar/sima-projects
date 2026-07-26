# Daily loop

Once everything is installed, a session is four moves: start, edit, copy, run.

## Start the SDK

```bash title="WSL"
sudo su -
cd sima-projects
source sima/bin/activate
sima-cli sdk neat
```

If Docker did not start automatically, run `sudo service docker start` first.

## The commands you actually use

| Task | Command | Run in |
| :-- | :-- | :-- |
| Start the SDK container | `sima-cli sdk neat` | WSL |
| Push the app to the board | `scp -r object-detection/ sima@<devkit-ip>:~` | WSL |
| Pull the video back | `scp sima@<devkit-ip>:~/object-detection/detections.mp4 .` | WSL |
| Pull the stills back | `scp -r sima@<devkit-ip>:~/object-detection/sandbox .` | WSL |
| SSH to the board | `dk shell` | SDK container |
| Check the sync method | `dk status` | SDK container |
| Component versions and ports | `neat` | SDK container |
| Run the app | `python3 src/app.py --config config.yaml` | DevKit |
| Kill an orphaned run | `pkill -f src/app.py` | DevKit |

## Stop typing the password

`sima-cli sdk setup` installs an SSH key for the **container's** root user, not for your
WSL user. Add one for yourself:

```bash title="WSL"
ssh-keygen -t ed25519 -C "devkit"
ssh-copy-id sima@<devkit-ip>
```

!!! tip "Check the address first if a copy hangs"

    The board gets its IP by DHCP, so it moves between reboots.

    ```powershell
    arp -a | Select-String "192.168.137"
    ```
