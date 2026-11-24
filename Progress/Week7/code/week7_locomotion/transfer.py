"""Shape-aware R25 -> R26 input expansion with deterministic output parity."""

import torch
from tensordict import TensorDict


def expand_state(source, destination):
    result = {k: v.clone() for k, v in destination.items()}
    old_dim = source["mlp.0.weight"].shape[1]
    for key, old in source.items():
        new = result[key]
        if new.shape == old.shape:
            new.copy_(old)
        elif (
            key == "mlp.0.weight"
            and new.shape[0] == old.shape[0]
            and new.shape[1] > old_dim
        ):
            new.zero_()
            new[:, :old_dim] = old
        elif (
            key
            in ("obs_normalizer._mean", "obs_normalizer._std", "obs_normalizer._var")
            and new.shape[-1] > old_dim
        ):
            new.fill_(
                0.0
                if key.endswith("_mean")
                else (0.35 if key.endswith("_std") else 0.35**2)
            )
            new[..., :old_dim] = old
        else:
            raise ValueError(
                f"Unsupported transfer tensor {key}: {old.shape} -> {new.shape}"
            )
    # Keep the initial normalization exactly, but permit adaptation to new data.
    result["obs_normalizer.count"].clamp_(max=200_000)
    return result


def transfer(runner, checkpoint):
    saved = torch.load(checkpoint, map_location=runner.device, weights_only=False)
    report = {
        "source": str(checkpoint),
        "optimizer_reset": True,
        "new_reference_columns_zeroed": True,
    }
    for name, network, old_dim in (
        ("actor", runner.alg.actor, 460),
        ("critic", runner.alg.critic, 98),
    ):
        old = saved[name + "_state_dict"]
        assert old["mlp.0.weight"].shape[1] == old_dim
        expanded = expand_state(old, network.state_dict())
        network.load_state_dict(expanded)
        network.eval()
        # Compare to an explicit forward pass through the original saved MLP.
        probe = old["obs_normalizer._mean"] + old["obs_normalizer._std"] * torch.randn(
            64, old_dim, device=runner.device
        )
        x = (probe - old["obs_normalizer._mean"]) / (old["obs_normalizer._std"] + 0.01)
        for layer in (0, 2, 4, 6):
            x = x @ old[f"mlp.{layer}.weight"].T + old[f"mlp.{layer}.bias"]
            if layer != 6:
                x = torch.nn.functional.elu(x)
        new_dim = expanded["mlp.0.weight"].shape[1]
        extra = torch.randn(64, new_dim - old_dim, device=runner.device)
        with torch.inference_mode():
            actual = network(
                TensorDict({name: torch.cat((probe, extra), dim=1)}, batch_size=[64])
            )
        delta = float((actual - x).abs().max())
        if delta > 5e-5:
            raise AssertionError(
                f"{name} transfer changed deterministic output: {delta}"
            )
        report[name + "_max_abs_difference"] = delta
        report[name + "_inputs"] = new_dim
    with torch.no_grad():
        runner.alg.actor.distribution.std_param.clamp_(min=0.05, max=0.07)
    runner.alg.train_mode()
    return report
