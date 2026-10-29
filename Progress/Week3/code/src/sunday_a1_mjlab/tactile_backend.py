"""Actual mjlab runtime for the interactive four-finger demo."""
import mujoco
import numpy as np
import torch


class MjlabTactileBackend:
    prefix = "robot/"

    def __init__(self, device="cuda:0", task="SundayA1-FixedBase-FlexiTac-v0"):
        import mjlab.tasks
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        cfg = load_env_cfg(task, play=True)
        cfg.scene.num_envs = 1
        cfg.terminations = {}  # Manual viewer runs until the user closes it.
        self.env = ManagerBasedRlEnv(cfg, device=device)
        self.env.reset()
        self.model = self.env.sim.mj_model
        self.data = mujoco.MjData(self.model)  # Display only; physics stays on GPU.
        self.robot = self.env.scene["robot"]
        _, names = self.robot.find_joints_by_actuator_names((".*",))
        self.action_names = names
        self.finger_joints = ("l_finger_slide_bottom", "l_finger_slide_top",
                              "r_finger_slide_bottom", "r_finger_slide_top")
        self.action_ids = [names.index(name) for name in self.finger_joints]
        self.action = torch.zeros(self.env.action_space.shape, device=device)
        self.sensor = self.env.scene["flexitac"]
        self._sync_display()

    def _sync_display(self):
        raw = self.env.sim.data
        for key in ("qpos", "qvel", "mocap_pos", "mocap_quat", "ctrl"):
            getattr(self.data, key)[:] = getattr(raw, key)[0].detach().cpu().numpy()
        self.data.time = self.env.episode_length_buf[0].item() * self.env.step_dt
        mujoco.mj_forward(self.model, self.data)

    def step(self, targets):
        self.action[:, self.action_ids] = torch.as_tensor(
            (np.asarray(targets) - 0.015) / 0.015, dtype=torch.float32, device=self.env.device)
        self.env.step(self.action)
        self._sync_display()
        return self.sensor.data

    def actual_positions(self):
        return np.array([self.data.joint(self.prefix + name).qpos[0] for name in self.finger_joints])

    def set_foot_tilts(self, left_degrees=0., right_degrees=0.):
        """Ankle rotation around local X for the fixed-base sole-contact demo."""
        for hand, degrees in (("l", left_degrees), ("r", right_degrees)):
            if not np.isfinite(degrees) or abs(degrees) > 5.001:
                raise ValueError("Foot tilt must be within +/-5 degrees")
            self.action[:, self.action_names.index(hand + "_ankle_roll")] = np.deg2rad(degrees) / .1

    def close(self):
        self.env.close()
