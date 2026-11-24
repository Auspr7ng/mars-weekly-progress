# r26_round4_checkpoint600 - Experimental simulation candidate

Checkpoint: `model_600.pt`, Round 4, source run
`r26_round4_raise_only_20261124_04`. Read the [measured results](../../results/Round4_Checkpoint600_Review.md)
before choosing this candidate. It is not a hardware release and does not
replace the accepted R25 baseline.

## Files

- `model_600.pt`: original training checkpoint, including critic/optimizer.
- `actor.npz`: deterministic NumPy actor and observation normalizer.
- `actor_contract.json`: input/action ownership, checksums and interface limits.
- `actor_parity.json`: numerical export comparison against checkpoint tensors.
- `run.json`: original training metadata and configuration/source hashes.
- `training_source.zip`, `source_manifest.json`: hash-verified training snapshot.
- `package.json`, `SHA256SUMS.txt`: package provenance and integrity metadata.

The source snapshot is the code present at training launch. Current evaluation
and packaging helpers may be newer and are versioned separately in Git.
Archived paths inside the historical run metadata need not exist locally.
Checksum-covered text files use LF line endings for Git portability; the
source archive preserves the original training-file bytes.

The actor has 510 inputs and retains 22 raw outputs/history values, but only
twelve leg outputs actuate. Ten arm targets come from held references.
Moon's 460-input R25 runner and historical BNO055 driver are incompatible
without adaptation. The current LSM6DSV16X wiring/mount/driver and body-speed
estimator have not been validated by this simulation package.
