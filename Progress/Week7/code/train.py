"""Bounded, resumable Week7 PPO training. No hardware or remote logging."""

import bootstrap

bootstrap.configure_warp()

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time
from zoneinfo import ZoneInfo

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from week7_locomotion.curriculum import ARM_NAMES, TASK_VERSION, env_targets, targets
from week7_locomotion.env_cfg import make_env, agent_cfg, DYNAMICS
from week7_locomotion.transfer import transfer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now(ZoneInfo("America/Indianapolis")).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class StopTraining(Exception):
    pass


class LocalRunner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        # Atomic replacement prevents an evaluator from reading a partial file.
        final = Path(path)
        staging = final.with_suffix(".saving")
        super().save(str(staging), infos)
        staging.replace(final)
        self.last_checkpoint = str(final.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--ramp-iterations", type=int, default=600)
    parser.add_argument("--seed", type=int, default=742)
    parser.add_argument("--run-dir", type=Path, required=True)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument(
        "--init-from",
        type=Path,
        help="Warm-start weights; fresh optimizer, local iteration and curriculum",
    )
    parser.add_argument(
        "--round", dest="training_round", type=int, choices=(1, 2, 3, 4, 5), default=1
    )
    parser.add_argument("--name", help="Human-readable experiment name")
    parser.add_argument(
        "--progress", type=float, help="Fixed curriculum level for a diagnostic run"
    )
    args = parser.parse_args()
    if not 1 <= args.envs <= 1024 or not 1 <= args.iterations <= 10000:
        parser.error("envs must be 1..1024 and iterations must be 1..10000")
    if args.ramp_iterations < 1 or (
        args.progress is not None and not 0 <= args.progress <= 1
    ):
        parser.error("ramp iterations must be positive and progress must be 0..1")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "run.json").exists():
        raise ValueError("Use a new run directory, including for a resumed run")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((bootstrap.BASE / "config/accepted_policy.json").read_text())
    source = bootstrap.BASE / manifest["checkpoint"]
    if sha(source) != manifest["checkpoint_sha256"]:
        raise ValueError("Accepted R25 checkpoint hash mismatch")
    for file, key in (
        ("sunday_a1_locomotion.xml", "model_sha256"),
        ("arm_home.json", "nominal_sha256"),
    ):
        if sha(DYNAMICS / file) != manifest[key]:
            raise ValueError("Archived model or Home hash mismatch")
    cfg = make_env(
        args.envs,
        args.seed,
        progress_override=args.progress,
        ramp_iterations=args.ramp_iterations,
        training_round=args.training_round,
    )
    acfg = agent_cfg(args.training_round)
    acfg.seed = args.seed
    acfg.max_iterations = args.iterations
    spec = {
        "task_version": TASK_VERSION,
        "revision": 26,
        "training_round": args.training_round,
        "reward_version": {
            1: "r26.round1.v1",
            2: "r26.round2.heading_worst_foot.v1",
            3: "r26.round3.phase_matched_mirror.v1",
            4: "r26.round4.raise_only_lift_gates.v1",
            5: "r26.round5.bilateral_35mm_reference.v1",
        }[args.training_round],
        "randomized_training_start_phase": cfg.week7_randomize_phase,
        "mirror_loss_coefficient": (
            (
                0.10
                if args.training_round >= 5
                else (0.05 if args.training_round >= 4 else 0.20)
            )
            if args.training_round >= 3
            else 0.0
        ),
        "phase_matched_height_cost_weight": (
            (
                -0.8
                if args.training_round >= 5
                else (-0.35 if args.training_round >= 4 else -1.0)
            )
            if args.training_round >= 3
            else 0.0
        ),
        "run_name": args.name or run.name,
        "actor_inputs": 510,
        "critic_inputs": 108,
        "actor_outputs": 22,
        "applied_leg_outputs": 12,
        "arm_reference_inputs_per_frame": 10,
        "history_length": 5,
        "arm_reference_order": list(ARM_NAMES),
        "num_envs": args.envs,
        "iterations_requested": args.iterations,
        "ramp_iterations": args.ramp_iterations,
        "seed": args.seed,
        "progress_override": args.progress,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha(source),
        "model_sha256": manifest["model_sha256"],
        "nominal_sha256": manifest["nominal_sha256"],
        "resume": str(args.resume.resolve()) if args.resume else None,
        "init_from": str(args.init_from.resolve()) if args.init_from else None,
        "parent_checkpoint_sha256": (
            sha(args.resume or args.init_from)
            if (args.resume or args.init_from)
            else None
        ),
        "simulation_only": True,
        "accepted": False,
        "body_velocity_source": "MuJoCo body velocity, not a fixed 0.06 m/s assumption",
        "physical_limits_changed": False,
        "started": timestamp(),
        "curriculum_initial": targets(0.0, args.training_round),
        "curriculum_final": targets(1.0, args.training_round),
        "dependencies": {
            n: importlib.metadata.version(n)
            for n in ("torch", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib")
        },
    }
    (run / "source").mkdir()
    for file in (bootstrap.WEEK / "code").rglob("*.py"):
        relative = file.relative_to(bootstrap.WEEK / "code")
        destination = run / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(file.read_bytes())
    spec["source_hashes"] = {
        str(f.relative_to(run / "source")): sha(f)
        for f in (run / "source").rglob("*.py")
    }
    (run / "dynamics").mkdir()
    for name in ("sunday_a1_locomotion.xml", "arm_home.json"):
        (run / "dynamics" / name).write_bytes((DYNAMICS / name).read_bytes())
    write_json(run / "run.json", spec)
    dump_yaml(run / "env.yaml", asdict(cfg))
    dump_yaml(run / "agent.yaml", asdict(acfg))
    status = {
        "state": "initializing",
        "pid": os.getpid(),
        "run_dir": str(run),
        "run_name": spec["run_name"],
        "training_round": args.training_round,
        "iterations_requested": args.iterations,
        "started": timestamp(),
    }
    write_json(run / "status.json", status)
    env = None
    try:
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        obs = wrapper.get_observations()
        if obs["actor"].shape[-1] != 510 or obs["critic"].shape[-1] != 108:
            raise ValueError(
                "Actual observation layout differs from the versioned 510/108 schema"
            )
        runner = LocalRunner(wrapper, asdict(acfg), str(run), device="cuda:0")
        if args.training_round >= 4:
            from week7_locomotion.symmetry import LegOnlyMirrorLoss

            LegOnlyMirrorLoss.install(runner)
        parent = args.resume or args.init_from
        if parent:
            old_spec = json.loads((parent.parent / "run.json").read_text())
            if (
                old_spec["task_version"] != TASK_VERSION
                or old_spec["model_sha256"] != spec["model_sha256"]
            ):
                raise ValueError("Resume checkpoint contract mismatch")
            if args.resume:
                if old_spec.get("training_round", 1) != args.training_round:
                    raise ValueError("Use --init-from for a new reward round")
                runner.load(str(parent))
                runner.current_learning_iteration += 1
            else:
                runner.load(str(parent), load_cfg={"actor": True, "critic": True})
                runner.current_learning_iteration = 0
                env.common_step_counter = 0
                runner.save(str(run / "model_initial.pt"))
        else:
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                initialization = transfer(runner, source)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf32
            write_json(run / "initialization.json", initialization)
            runner.save(str(run / "model_initial.pt"))
        first_iteration = runner.current_learning_iteration
        status["first_iteration"] = first_iteration
        original_log = runner.logger.log
        started = time.perf_counter()

        def tracked_log(*a, **kw):
            original_log(*a, **kw)
            losses = {k: float(v) for k, v in kw["loss_dict"].items()}
            if not all(math.isfinite(v) for v in losses.values()):
                raise FloatingPointError("Nonfinite PPO loss")
            status.update(
                state="running",
                iteration=runner.current_learning_iteration,
                iterations_completed=runner.current_learning_iteration
                - first_iteration
                + 1,
                transitions=(runner.current_learning_iteration - first_iteration + 1)
                * args.envs
                * 32,
                elapsed_s=time.perf_counter() - started,
                checkpoint=getattr(runner, "last_checkpoint", None),
                curriculum=env_targets(env),
                losses=losses,
                mean_reward=(
                    statistics.mean(runner.logger.rewbuffer)
                    if runner.logger.rewbuffer
                    else None
                ),
                mean_episode_length_steps=(
                    statistics.mean(runner.logger.lenbuffer)
                    if runner.logger.lenbuffer
                    else None
                ),
                updated=timestamp(),
            )
            if args.training_round >= 3:
                balance = env.reward_manager.get_term_cfg(
                    "phase_matched_clearance"
                ).func
                status["balance_phase_pair_fraction"] = float(
                    balance.last_paired.float().mean()
                )
                status["balance_phase_cost_unweighted"] = float(
                    balance.last_cost.mean()
                )
            write_json(run / "status.json", status)
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(status, allow_nan=False) + "\n")
            if (run / "STOP").exists():
                runner.save(
                    str(run / f"model_{runner.current_learning_iteration}_stopped.pt")
                )
                raise StopTraining()

        runner.logger.log = tracked_log
        runner.learn(
            num_learning_iterations=args.iterations, init_at_random_ep_len=False
        )
        status.update(
            state="completed",
            checkpoint=runner.last_checkpoint,
            gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
            updated=timestamp(),
        )
        write_json(run / "status.json", status)
        print("WEEK7_TRAINING_COMPLETED " + json.dumps(status), flush=True)
    except (StopTraining, KeyboardInterrupt):
        status.update(state="stopped", updated=timestamp())
        write_json(run / "status.json", status)
        print("WEEK7_TRAINING_STOPPED", flush=True)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), updated=timestamp())
        write_json(run / "status.json", status)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
