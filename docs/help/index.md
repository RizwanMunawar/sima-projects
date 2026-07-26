# Help

| Page | Use it when |
| :-- | :-- |
| [Troubleshooting](troubleshooting.md) | You have an error message and want the fix |
| [Recovery](recovery.md) | The DevKit is in a bad state: version mismatch, missing pyneat |
| [Known issues](known-issues.md) | Bugs in the SDK itself, with the workarounds |

## The three failures that cost the most time

| Symptom | Cause |
| :-- | :-- |
| `source ~/pyneat/bin/activate` not found | Pairing never installed onto the board, because networking was not fixed first |
| Insight loads but shows nothing | The Hyper-V firewall is dropping the inbound UDP |
| Output video far shorter than the input | The run stalled on backpressure part-way through |

None of them produce an error at the point where the mistake was made, which is what
makes them expensive.
