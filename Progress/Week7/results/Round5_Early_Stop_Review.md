# Round 5 - Early stop decision

November 24, 2026. The 35 mm reference comparison was deliberately stopped
after **409 of 600 planned updates**, with 6,701,056 transitions. The STOP
handler saved `model_408_stopped.pt`; this was a controlled stop, not a crash
or a claim that all planned updates were completed.

## Measured reason

Matched six-second checks use seed 1742, four starts, seven held arm poses
and commands 0.06, 0.12 and 0.20 m/s. The original walking gates remain unchanged.

| Checkpoint | Walking passes | Home fast lift L/R, mm | Home fast inward toe L/R, degrees |
| --- | --- | --- | --- |
| Round 4 parent 200 | 84/84 | 38.1 / 28.8 | 5.5 / 6.6 |
| Round 5 75 | 84/84 | 35.6 / 26.8 | 5.1 / 5.6 |
| Round 5 200 | 84/84 | 32.9 / 27.7 | 4.6 / 5.7 |
| Round 5 400 | 84/84 | 31.6 / 24.3 | 3.0 / 4.7 |
| Round 5 stopped 408 | 84/84 | 31.9 / 25.0 | 2.8 / 4.7 |

Heading improved, but foot lift regressed rather than satisfying the user's
higher-foot requirement. Continuing this branch had less value than completing
the 45 mm-reference experiments. No inference of global convergence or causal
isolation follows from this early stop: the reference and several objective
weights differ, and all displayed values are finite sampled measurements.

Source reports are `r26_round4_checkpoint200_matrix.json`,
`r26_round5_checkpoint75_matrix.json`, `r26_round5_checkpoint200_matrix.json`
and `r26_round5_checkpoint400_matrix.json`. The stopped checkpoint receives
its own short check, `r26_round5_stopped408_matrix.json`; the table does not
mislabel checkpoint 400 as the final saved checkpoint.

The local runtime status initially retained the older periodic checkpoint
path after STOP. That pointer was corrected to the existing stopped file,
with an explicit correction note. Training counters and per-update metrics
were preserved. `train.py` now updates that pointer after saving a STOP
checkpoint; this reporting fix does not change any training loss or physics.

Round 5 is not selected as the experimental progress candidate. Its ignored
local run retains checkpoints, metrics and immutable source/configuration
snapshots. No hardware deployment or new IMU validation is claimed.
