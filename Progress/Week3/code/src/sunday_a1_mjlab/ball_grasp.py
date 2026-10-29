"""Fixed-base, independently resettable dual-hand sphere grasps."""
import json

import mujoco
import numpy as np

from .flexitac import FINGERS

TASK_ID = "SundayA1-FixedBase-BallGrasp-v0"
BALL_RADIUS = 0.014
BALL_MASS = 0.04
GRIP_TARGET = 0.012
BALL_ENTITIES = {"left": "ball", "right": "right_ball"}
CONTACT_SENSORS = {"left": "grip_contacts", "right": "right_grip_contacts"}
HAND_INDICES = {"left": (0, 1), "right": (2, 3)}


def grasp_robot_spec():
    from .tasks import robot_spec
    spec = robot_spec()
    for finger in FINGERS:
        body = spec.body(finger.link)
        # MuJoCo mesh collision uses a convex hull. A planar pad represents the
        # contact surface without filling the concave space between the fingers.
        for geom in body.geoms:
            if geom.contype:
                geom.contype = 16
        center = np.asarray(finger.center).copy()
        center[1] -= finger.inward * 0.0005
        body.add_geom(name="grip_pad_" + finger.name,
                      type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=center, size=(0.012, 0.0005, 0.032), mass=0,
                      contype=8, conaffinity=4, condim=4,
                      friction=(1.0, 0.005, 0.0001),
                      solref=(0.01, 1), solimp=(0.95, 0.99, 0.001, 0.5, 2),
                      rgba=(*finger.color, 0.5), group=3)
    return spec


def ball_spec():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="free")
    body.add_geom(name="sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=(BALL_RADIUS, 0, 0), mass=BALL_MASS,
                  contype=4, conaffinity=15, condim=4,
                  friction=(1.0, 0.005, 0.0001),
                  solref=(0.01, 1), solimp=(0.95, 0.99, 0.001, 0.5, 2),
                  rgba=(1.0, 0.35, 0.05, 1), group=1)
    return spec


def right_ball_spec():
    spec = ball_spec()
    spec.geom("sphere").rgba = (.1, .5, 1., 1.)
    return spec


def initial_ball_position(hand="left"):
    from .tasks import ASSETS, robot_spec
    model = robot_spec().compile()
    data = mujoco.MjData(model)
    for joint in ("l_finger_slide_bottom", "l_finger_slide_top",
                  "r_finger_slide_bottom", "r_finger_slide_top"):
        data.joint(joint).qpos[0] = 0.015
    mujoco.mj_forward(model, data)
    centers = []
    for index in HAND_INDICES[hand]:
        finger = FINGERS[index]
        bid = model.body(finger.link).id
        centers.append(data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ finger.center)
    position = np.mean(centers, axis=0)
    position[2] += json.loads((ASSETS / "model_info.json").read_text())["base_height"]
    return tuple(position)


def ball_observation(env):
    import torch
    values = []
    for entity in BALL_ENTITIES.values():
        ball = env.scene[entity].data
        values.extend((ball.root_link_pos_w - env.scene.env_origins, ball.root_link_vel_w))
    return torch.cat(values, dim=-1)


def make_ball_env_cfg(num_envs=1):
    from mjlab.entity import EntityCfg
    from mjlab.envs import mdp
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.sensor import ContactSensorCfg
    from mjlab.sensor.contact_sensor import ContactMatch
    from .tasks import make_tactile_env_cfg
    from .feet_tactile import STANDING_LOWER_M
    from .mjlab_flexitac import FootTactileSensorCfg, foot_tactile_observation
    cfg = make_tactile_env_cfg(num_envs)
    cfg.scene.entities["robot"].spec_fn = grasp_robot_spec
    pos = list(cfg.scene.entities["robot"].init_state.pos)
    pos[2] -= STANDING_LOWER_M
    cfg.scene.entities["robot"].init_state.pos = tuple(pos)
    cfg.scene.sensors += (FootTactileSensorCfg(name="feet_tactile"),)
    for hand, entity in BALL_ENTITIES.items():
        ball_pos = list(initial_ball_position(hand))
        ball_pos[2] -= STANDING_LOWER_M
        cfg.scene.entities[entity] = EntityCfg(
            spec_fn=ball_spec if hand == "left" else right_ball_spec,
            init_state=EntityCfg.InitialStateCfg(pos=tuple(ball_pos)),
        )
        cfg.scene.sensors += (ContactSensorCfg(
            name=CONTACT_SENSORS[hand],
            primary=ContactMatch(mode="geom", pattern=f"grip_pad_{hand}_.*", entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="sphere", entity=entity),
            fields=("found", "force"), reduce="netforce",
        ),)
        cfg.events["reset_" + entity] = EventTermCfg(
            func=mdp.reset_root_state_uniform, mode="reset",
            params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg(entity)},
        )
    for group in cfg.observations.values():
        group.terms["ball"] = ObservationTermCfg(func=ball_observation)
        group.terms["feet_tactile"] = ObservationTermCfg(func=foot_tactile_observation)
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.impratio = 10.0
    return cfg


class BallGraspBackend:
    """Compose the viewer backend with isolated, explicit per-hand reset writes."""
    def __init__(self, device="cuda:0"):
        from .tactile_backend import MjlabTactileBackend
        self.runtime = MjlabTactileBackend(device=device, task=TASK_ID)
        self.hand_joint_ids = {}
        for hand, indices in HAND_INDICES.items():
            names = tuple(self.runtime.finger_joints[i] for i in indices)
            self.hand_joint_ids[hand], _ = self.runtime.robot.find_joints(names, preserve_order=True)

    def reset_hand(self, hand, offset_mm=(0., 0., 0.)):
        """Place one free ball at a hand-relative offset, then allow real physics.

        Only this hand's two slider positions/velocities and this ball's state
        are written. No whole-environment reset, clock reset, or arm reset.
        """
        import torch
        runtime = self.runtime
        offset = np.asarray(offset_mm, dtype=float) / 1000
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Ball offset must be a finite XYZ vector")
        if np.any(np.abs(offset) > (.008001, .006001, .020001)):
            raise ValueError("Ball offset exceeds fingertip workspace (8, 6, 20 mm)")
        ids = self.hand_joint_ids[hand]
        # Make room for the shifted sphere before closing; avoid deep overlaps.
        opening = .016 + abs(offset[1])
        q = torch.full((1, 2), opening, device=runtime.env.device)
        runtime.robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids)
        runtime.env.sim.forward()
        runtime._sync_display()
        centers = []
        for index in HAND_INDICES[hand]:
            finger = FINGERS[index]
            bid = runtime.model.body(runtime.prefix + finger.link).id
            rotation = runtime.data.xmat[bid].reshape(3, 3)
            centers.append(runtime.data.xpos[bid] + rotation @ finger.center)
        position = np.mean(centers, axis=0) + rotation @ offset
        state = torch.zeros((1, 13), dtype=torch.float32, device=runtime.env.device)
        state[0, :3] = torch.as_tensor(position, device=runtime.env.device)
        state[0, 3] = 1.
        runtime.env.scene[BALL_ENTITIES[hand]].write_root_state_to_sim(state)
        runtime.env.sim.forward()
        runtime.env.sim.sense()
        runtime.env.scene.update(0.)
        runtime._sync_display()
        return position

    def close(self):
        self.runtime.close()
