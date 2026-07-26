# 1. Connect the DevKit

Physical setup first. The direct Ethernet cable matters more than it looks: it gives you
a private subnet where Windows takes `192.168.137.1` and the board gets an address by
DHCP, and later it doubles as the board's route to the internet.

## Cable it up

1. Connect the **USB cable** supplied by SiMa. This is the serial console.
2. Connect an **Ethernet cable** directly from your PC to the DevKit.
3. Open the [serial console tool](https://docs.sima.ai/_static/tools/serial/index.html)
   and set the DevKit network to **DHCP**.

## Find it and check it answers

```powershell title="PowerShell"
arp -a | Select-String "192.168.137"     # find the board
ping <devkit-ip>
```

!!! success "Exit criteria"

    The ping must reply. Nothing else in this guide works until it does.

!!! warning "`<devkit-ip>` appears throughout these docs"

    Substitute your own. The board gets its address by DHCP, so it changes between
    reboots: mine has been both `192.168.137.123` and `192.168.137.193`.

    Your PC keeps `192.168.137.1`, which is why that one is always written out in
    full. If a command suddenly hangs, re-check the board's address before assuming
    anything is broken.

---

Next: [Install WSL2 and fix networking](wsl.md)
