"""mjlab sensor lifecycle adapter for the four-finger Warp array."""
from dataclasses import dataclass

from mjlab.sensor.sensor import Sensor, SensorCfg

from .flexitac import FourFingerFlexiTac, FlexiTacData, attach_patch_sites


@dataclass
class FlexiTacSensorCfg(SensorCfg):
    entity: str = "robot"
    sphere_geom: str | tuple[str, ...] | None = None

    def build(self):
        return FlexiTacSensor(self)


class FlexiTacSensor(Sensor[FlexiTacData]):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def edit_spec(self, scene_spec, entities):
        if self.cfg.entity not in entities:
            raise ValueError(f"Missing robot entity: {self.cfg.entity}")
        attach_patch_sites(scene_spec, prefix=self.cfg.entity + "/")

    def initialize(self, mj_model, model, data, device):
        del model
        self.sim_data = data
        raw = data.xpos.wp_array if hasattr(data.xpos, "wp_array") else data.xpos
        self.engine = FourFingerFlexiTac(mj_model, raw.shape[0], device,
                                       self.cfg.entity + "/", self.cfg.sphere_geom)

    def _compute_data(self):
        arrays = [getattr(self.sim_data, key) for key in ("xpos", "xmat", "geom_xpos", "geom_xmat")]
        return self.engine.sample_warp(*[a.wp_array if hasattr(a, "wp_array") else a for a in arrays])

    def debug_vis(self, visualizer):
        values = self.data.tactile_points_w_per_sensor.detach().cpu().numpy()
        for env in visualizer.get_env_indices(values.shape[0]):
            for sensor in values[env]:
                for x, y, z, value in sensor:
                    visualizer.add_sphere(center=(x, y, z), radius=0.00065,
                                          color=(float(value), 0.15, 1.0 - float(value), 1.0))


def tactile_observation(env):
    """Each hand/finger occupies a distinct 384-value section."""
    return env.scene["flexitac"].data.response.flatten(start_dim=1)


@dataclass
class FootTactileSensorCfg(SensorCfg):
    entity: str = "robot"

    def build(self):
        return FootTactileSensor(self)


class FootTactileSensor(FlexiTacSensor):
    def edit_spec(self, scene_spec, entities):
        from .feet_tactile import attach_foot_sites
        if self.cfg.entity not in entities:
            raise ValueError(f"Missing robot entity: {self.cfg.entity}")
        attach_foot_sites(scene_spec, prefix=self.cfg.entity + "/")

    def initialize(self, mj_model, model, data, device):
        from .feet_tactile import FootTactileEngine
        del model
        self.sim_data = data
        raw = data.xpos.wp_array if hasattr(data.xpos, "wp_array") else data.xpos
        self.engine = FootTactileEngine(mj_model, raw.shape[0], device, self.cfg.entity + "/")


def foot_tactile_observation(env):
    return env.scene["feet_tactile"].data.response.flatten(start_dim=1)
