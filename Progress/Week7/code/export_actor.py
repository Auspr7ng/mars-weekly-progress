"""Export an experimental 510-input NumPy actor; no hardware access."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NumpyActor:
    """Deterministic actor only. Does not build observations or send motor goals."""

    def __init__(self, folder):
        folder = Path(folder)
        self.contract = json.loads((folder / "actor_contract.json").read_text())
        path = folder / "actor.npz"
        if sha(path) != self.contract["actor_npz_sha256"]:
            raise ValueError("Actor checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            self.params = {key: data[key].copy() for key in data.files}
        shapes = {
            "mlp_0_weight": (256, 510),
            "mlp_2_weight": (128, 256),
            "mlp_4_weight": (128, 128),
            "mlp_6_weight": (22, 128),
        }
        if any(self.params[key].shape != value for key, value in shapes.items()):
            raise ValueError("Unexpected Week7 actor architecture")

    def __call__(self, observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape[-1] != 510 or not np.isfinite(x).all():
            raise ValueError("Expected finite 510-input observations")
        x = (x - self.params["obs_normalizer__mean"]) / (
            self.params["obs_normalizer__std"] + 0.01
        )
        for index in (0, 2, 4, 6):
            x = (
                x @ self.params[f"mlp_{index}_weight"].T
                + self.params[f"mlp_{index}_bias"]
            )
            if index != 6:
                negative = x < 0
                x[negative] = np.expm1(x[negative])
        if not np.isfinite(x).all():
            raise FloatingPointError("Nonfinite actor output")
        return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(4)
    spec = json.loads((args.checkpoint.parent / "run.json").read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["actor_state_dict"]
    if tuple(state["mlp.0.weight"].shape) != (256, 510):
        raise ValueError("This exporter requires the Week7 510-input contract")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    values = {
        key.replace(".", "_"): value.detach().numpy()
        for key, value in state.items()
        if key.startswith(("mlp.", "obs_normalizer."))
    }
    np.savez(output / "actor.npz", **values)
    contract = dict(
        schema="sunday.week7.r26.actor_numpy.v1",
        task_version=spec["task_version"],
        training_round=spec.get("training_round", 1),
        observation_dim=510,
        action_dim=22,
        applied_leg_outputs=12,
        held_arm_outputs=10,
        history_length=5,
        control_hz=50,
        arm_reference_order=spec["arm_reference_order"],
        model_sha256=spec["model_sha256"],
        nominal_sha256=spec["nominal_sha256"],
        source_checkpoint_sha256=sha(args.checkpoint),
        actor_npz_sha256=sha(output / "actor.npz"),
        training_swing_reference_m=spec["curriculum_final"]["swing_target_m"],
        training_forward_cap_m_s=spec["curriculum_final"]["forward_cap_m_s"],
        training_reference_is_achieved_performance=False,
        hardware_release=False,
        accepted=False,
        old_moon_460_input_runner_compatible=False,
        current_imu="ST LSM6DSV16X",
        hardware_imu_adapter_tested=False,
        observation_layout="Old term-major 460-feature prefix, then five frames of ten held-arm-reference offsets; do not interleave complete frames",
        action_contract="Keep 22 raw-action history values; apply only 12 leg outputs with archived filtering/slew/limits; ten arm targets come from the reference command",
        velocity_source_requirement="Measured/estimated body velocity; neither command nor fixed synthetic X value is a validated measurement",
    )
    (output / "actor_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    actor = NumpyActor(output)
    rng = np.random.default_rng(4742)
    observation = (
        values["obs_normalizer__mean"]
        + values["obs_normalizer__std"] * rng.normal(size=(64, 510))
    ).astype(np.float32)
    tensor = torch.from_numpy(observation)
    with torch.inference_mode():
        tensor = (tensor - state["obs_normalizer._mean"]) / (
            state["obs_normalizer._std"] + 0.01
        )
        for index in (0, 2, 4, 6):
            tensor = functional.linear(
                tensor, state[f"mlp.{index}.weight"], state[f"mlp.{index}.bias"]
            )
            if index != 6:
                tensor = functional.elu(tensor)
    difference = float(np.max(np.abs(actor(observation) - tensor.numpy())))
    report = dict(
        probes=64,
        max_abs_numpy_torch_output_difference=difference,
        passed=difference < 1e-5,
        scope="Deterministic actor numerical parity only; not observation/driver or hardware validation",
    )
    (output / "actor_parity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise ValueError("NumPy/Torch export parity failed")
    print("WEEK7_NUMPY_EXPORT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
