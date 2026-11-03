"""Four moving 12x32 FlexiTac-style arrays using Warp triangle distances.

The grid and unsigned 1.5 mm shell follow Week 2, with normals aligned
to the actual inner collision-pad faces on both hands. Values are
normalized proximity responses, not measured contact forces or pressure.
No Isaac Sim, USD, PhysX, convex hull, or stale world poses are used here.
"""
from dataclasses import dataclass
import mujoco
import numpy as np
import torch
import warp as wp

ROWS, COLS, PITCH_M = 12, 32, 0.002
SHELL_M, STIFFNESS, MAX_RESPONSE = 0.0015, 5000.0, 2.4
ACTIVE_THRESHOLD = 0.01


@dataclass(frozen=True)
class FingerPatch:
    name: str
    link: str
    opposite: str
    surface_y: float
    inward: int
    color: tuple[float, float, float]

    @property
    def center(self):
        return (-0.134, self.surface_y + self.inward * 0.0005, -0.035)


# Inward means toward the ball in the actual finger-local frame.
# Right-hand CAD axes require the same signs as the left-hand axes.
FINGERS = (
    FingerPatch("left_bottom", "l_finger_bottom_1", "l_finger_top_1", -0.0494, -1, (0.1, 0.9, 0.2)),
    FingerPatch("left_top", "l_finger_top_1", "l_finger_bottom_1", 0.0289, 1, (0.1, 0.8, 0.9)),
    FingerPatch("right_bottom", "r_finger_bottom_1", "r_finger_top_1", 0.0488, -1, (0.9, 0.8, 0.1)),
    FingerPatch("right_top", "r_finger_top_1", "r_finger_bottom_1", -0.0309, 1, (0.9, 0.2, 0.8)),
)


def local_taxels():
    u, v = np.meshgrid((np.arange(ROWS) - (ROWS - 1) / 2) * PITCH_M,
                       (np.arange(COLS) - (COLS - 1) / 2) * PITCH_M, indexing="ij")
    result = []
    for finger in FINGERS:
        # Week 2 rotates negative-inward patches by pi about local X.
        points = np.stack((u.ravel(), np.zeros(ROWS * COLS),
                           v.ravel() * finger.inward), axis=-1)
        result.append(points + np.asarray(finger.center))
    return np.asarray(result, dtype=np.float32)


@wp.kernel(enable_backward=False)
def initialize_surface_taxels(
    body_pos: wp.array2d(dtype=wp.vec3),
    body_rot: wp.array2d(dtype=wp.mat33),
    source_ids: wp.array(dtype=wp.int32),
    points_local: wp.array2d(dtype=wp.vec3),
    output: wp.array4d(dtype=wp.float32),
    distance: wp.array3d(dtype=wp.float32),
):
    env, sensor, taxel = wp.tid()
    source = source_ids[sensor]
    p = body_pos[env, source] + body_rot[env, source] * points_local[sensor, taxel]
    output[env, sensor, taxel, 0] = p[0]
    output[env, sensor, taxel, 1] = p[1]
    output[env, sensor, taxel, 2] = p[2]
    output[env, sensor, taxel, 3] = 0.0
    distance[env, sensor, taxel] = 0.1


@wp.kernel(enable_backward=False)
def sample_taxels(
    body_pos: wp.array2d(dtype=wp.vec3),
    body_rot: wp.array2d(dtype=wp.mat33),
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_rot: wp.array2d(dtype=wp.mat33),
    source_ids: wp.array(dtype=wp.int32),
    target_ids: wp.array(dtype=wp.int32),
    meshes: wp.array(dtype=wp.uint64),
    points_local: wp.array2d(dtype=wp.vec3),
    shell: float,
    stiffness: float,
    max_response: float,
    output: wp.array4d(dtype=wp.float32),
    distance: wp.array3d(dtype=wp.float32),
):
    env, sensor, taxel = wp.tid()
    source = source_ids[sensor]
    target = target_ids[sensor]
    point_w = body_pos[env, source] + body_rot[env, source] * points_local[sensor, taxel]
    query = wp.transpose(geom_rot[env, target]) * (point_w - geom_pos[env, target])
    hit = wp.mesh_query_point(meshes[sensor], query, 0.1)
    d = float(0.1)
    if hit.result:
        closest = wp.mesh_eval_position(meshes[sensor], hit.face, hit.u, hit.v)
        d = wp.length(query - closest)
    response = wp.clamp(stiffness * wp.max(shell - d, 0.0) / max_response, 0.0, 1.0)
    output[env, sensor, taxel, 0] = point_w[0]
    output[env, sensor, taxel, 1] = point_w[1]
    output[env, sensor, taxel, 2] = point_w[2]
    output[env, sensor, taxel, 3] = response
    distance[env, sensor, taxel] = d


@wp.kernel(enable_backward=False)
def merge_scene_taxels(
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_rot: wp.array2d(dtype=wp.mat33),
    source_ids: wp.array(dtype=wp.int32),
    target_ids: wp.array(dtype=wp.int32),
    target_bodies: wp.array(dtype=wp.int32),
    types: wp.array(dtype=wp.int32),
    sizes: wp.array(dtype=wp.vec3),
    bounds: wp.array(dtype=wp.float32),
    meshes: wp.array(dtype=wp.uint64),
    shell: float,
    stiffness: float,
    max_response: float,
    output: wp.array4d(dtype=wp.float32),
    distance: wp.array3d(dtype=wp.float32),
):
    env, sensor, taxel = wp.tid()
    p = wp.vec3(output[env, sensor, taxel, 0], output[env, sensor, taxel, 1],
                output[env, sensor, taxel, 2])
    # Keep opposing-finger response; also sense every external scene object.
    d = distance[env, sensor, taxel]
    for i in range(target_ids.shape[0]):
        if target_bodies[i] == source_ids[sensor]:
            continue
        gid = target_ids[i]
        kind = types[i]
        delta = p - geom_pos[env, gid]
        if kind != 0 and wp.length(delta) - bounds[i] > d:
            continue
        q = wp.transpose(geom_rot[env, gid]) * delta
        size = sizes[i]
        candidate = float(0.1)
        if kind == 0:  # Plane: normal is local +Z.
            candidate = q[2]
        elif kind == 2:  # Sphere.
            candidate = wp.length(q) - size[0]
        elif kind == 3:  # Capsule: local Z axis.
            center_z = wp.clamp(q[2], -size[1], size[1])
            candidate = wp.length(q - wp.vec3(0.0, 0.0, center_z)) - size[0]
        elif kind == 4:  # Ellipsoid: first-order distance approximation.
            a = wp.length(wp.cw_div(q, size))
            b = wp.length(wp.cw_div(q, wp.cw_mul(size, size)))
            if b > 1.0e-8:
                candidate = a * (a - 1.0) / b
            else:
                candidate = -wp.min(size[0], wp.min(size[1], size[2]))
        elif kind == 5:  # Cylinder.
            cylinder_q = wp.vec2(wp.length(wp.vec2(q[0], q[1])) - size[0], wp.abs(q[2]) - size[1])
            candidate = wp.length(wp.max(cylinder_q, wp.vec2(0.0))) + wp.min(wp.max(cylinder_q[0], cylinder_q[1]), 0.0)
        elif kind == 6:  # Box.
            box_q = wp.vec3(wp.abs(q[0]), wp.abs(q[1]), wp.abs(q[2])) - size
            candidate = wp.length(wp.max(box_q, wp.vec3(0.0))) + wp.min(wp.max(box_q[0], wp.max(box_q[1], box_q[2])), 0.0)
        elif kind == 7:  # Full triangle surface, not the collision convex hull.
            hit = wp.mesh_query_point(meshes[i], q, 0.1)
            if hit.result:
                closest = wp.mesh_eval_position(meshes[i], hit.face, hit.u, hit.v)
                candidate = wp.length(q - closest)
        d = wp.min(d, candidate)
    response = wp.clamp(stiffness * wp.max(shell - d, 0.0) / max_response, 0.0, 1.0)
    output[env, sensor, taxel, 3] = response
    distance[env, sensor, taxel] = d


@dataclass
class FlexiTacData:
    tactile_points_w_per_sensor: torch.Tensor  # [E,patches,taxels,4] => x,y,z,response
    distance_m: torch.Tensor                 # [E,patches,taxels]
    patches: tuple = FINGERS
    rows: int = ROWS
    cols: int = COLS
    active_threshold: float = ACTIVE_THRESHOLD

    @property
    def tactile_points_w(self):
        return self.tactile_points_w_per_sensor.flatten(start_dim=1, end_dim=2)

    @property
    def response(self):
        return self.tactile_points_w_per_sensor[..., 3]

    @property
    def heatmaps(self):
        return self.response.reshape(-1, len(self.patches), self.rows, self.cols)

    def summary(self, env=0):
        values = self.response[env].detach().cpu().numpy()
        distances = self.distance_m[env].detach().cpu().numpy()
        return {f.name: {"max": float(v.max()), "mean": float(v.mean()),
                         "active": int((v > self.active_threshold).sum()),
                         "min_distance_mm": float(d.min() * 1000)}
                for f, v, d in zip(self.patches, values, distances)}


class FourFingerFlexiTac:
    """Batched surface grids; finger defaults also serve sole-grid adapters."""
    def __init__(self, model, num_envs=1, device="cuda:0", prefix="", sphere_geom=None,
                 *, patches=FINGERS, points_local=None, grid_shape=(ROWS, COLS), shell_m=SHELL_M,
                 response_depth_m=None):
        self.device, self.num_envs = device, num_envs
        self.patches = patches
        self.rows, self.cols = grid_shape
        self.dim = (num_envs, len(patches), self.rows * self.cols)
        self.shell_m = shell_m
        self.stiffness = STIFFNESS if response_depth_m is None else 1.
        self.max_response = MAX_RESPONSE if response_depth_m is None else response_depth_m
        # Retained argument for older callers; sensing no longer binds to names.
        del sphere_geom
        self.source_ids = [model.body(prefix + f.link).id for f in patches]
        self.target_ids, self.meshes = [], []
        self.mesh_arrays = []
        mesh_cache = {}

        def load_mesh(gid):
            mid = int(model.geom_dataid[gid])
            if mid in mesh_cache:
                return mesh_cache[mid]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            vertices = np.array(model.mesh_vert[va:va + vn], dtype=np.float32)
            faces = np.array(model.mesh_face[fa:fa + fn], dtype=np.int32).ravel()
            vp = wp.array(vertices, dtype=wp.vec3, device=device)
            fi = wp.array(faces, dtype=wp.int32, device=device)
            self.mesh_arrays.append((vp, fi))
            mesh = wp.Mesh(points=vp, indices=fi)
            self.meshes.append(mesh)
            mesh_cache[mid] = mesh.id
            return mesh.id

        opposite_mesh_ids = []
        self.has_opposite = all(f.opposite is not None for f in patches)
        for finger in patches if self.has_opposite else ():
            bid = model.body(prefix + finger.opposite).id
            gids = np.flatnonzero((model.geom_bodyid == bid) &
                                 (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH))
            if not len(gids):
                raise ValueError(f"No triangle mesh on target {finger.opposite}")
            # Visual and collision geoms share the same full triangle mesh.
            gid = int(gids[0])
            opposite_mesh_ids.append(load_mesh(gid))
            self.target_ids.append(gid)
        self.sources_wp = wp.array(self.source_ids, dtype=wp.int32, device=device)
        self.targets_wp = wp.array(self.target_ids, dtype=wp.int32, device=device)
        self.mesh_ids_wp = wp.array(opposite_mesh_ids, dtype=wp.uint64, device=device)
        # Ignore non-colliding visual duplicates and this robot's disabled
        # self-collisions, while always including other finger surfaces.
        candidates = []
        robot_root = model.body(prefix + "base_link").id
        robot_bodies = set()
        for bid in range(1, model.nbody):
            ancestor = bid
            while ancestor:
                if ancestor == robot_root:
                    robot_bodies.add(bid)
                    break
                ancestor = int(model.body_parentid[ancestor])
        for gid in range(model.ngeom):
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                continue
            bid = int(model.geom_bodyid[gid])
            if bid in robot_bodies and bid not in self.source_ids:
                continue
            kind = int(model.geom_type[gid])
            if kind not in (0, 2, 3, 4, 5, 6, 7):
                raise ValueError(f"Tactile distance unsupported for geom {model.geom(gid).name}, type {kind}")
            candidates.append(gid)
        self.scene_target_ids = candidates
        self.scene_ids_wp = wp.array(candidates, dtype=wp.int32, device=device)
        self.scene_bodies_wp = wp.array(model.geom_bodyid[candidates], dtype=wp.int32, device=device)
        self.scene_types_wp = wp.array(model.geom_type[candidates], dtype=wp.int32, device=device)
        self.scene_sizes_wp = wp.array(model.geom_size[candidates], dtype=wp.vec3, device=device)
        self.scene_bounds_wp = wp.array(model.geom_rbound[candidates], dtype=wp.float32, device=device)
        self.scene_meshes_wp = wp.array([
            load_mesh(gid) if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH else 0
            for gid in candidates], dtype=wp.uint64, device=device)
        self.local_wp = wp.array(local_taxels() if points_local is None else points_local,
                                 dtype=wp.vec3, device=device)
        self.output_wp = wp.zeros((*self.dim, 4), device=device)
        self.distance_wp = wp.zeros(self.dim, device=device)
        self.result = FlexiTacData(wp.to_torch(self.output_wp), wp.to_torch(self.distance_wp),
                                  patches, self.rows, self.cols)

    def sample_warp(self, body_pos, body_rot, geom_pos, geom_rot):
        if self.has_opposite:
            wp.launch(sample_taxels, dim=self.dim, inputs=[
                body_pos, body_rot, geom_pos, geom_rot, self.sources_wp, self.targets_wp,
                self.mesh_ids_wp, self.local_wp, self.shell_m, self.stiffness, self.max_response,
                self.output_wp, self.distance_wp], device=self.device)
        else:
            wp.launch(initialize_surface_taxels, dim=self.dim, inputs=[
                body_pos, body_rot, self.sources_wp, self.local_wp,
                self.output_wp, self.distance_wp], device=self.device)
        wp.launch(merge_scene_taxels, dim=self.dim, inputs=[
            geom_pos, geom_rot, self.sources_wp, self.scene_ids_wp, self.scene_bodies_wp,
            self.scene_types_wp, self.scene_sizes_wp, self.scene_bounds_wp, self.scene_meshes_wp,
            self.shell_m, self.stiffness, self.max_response,
            self.output_wp, self.distance_wp], device=self.device)
        if str(self.device).startswith("cuda"):
            stream = wp.get_stream(self.device)
            torch.cuda.current_stream(self.device).wait_stream(
                torch.cuda.ExternalStream(stream.cuda_stream, device=self.device))
        return self.result

    def sample_mujoco(self, data):
        """Native CPU MuJoCo poses, sampled by the same Warp kernel as mjlab."""
        arrays = [wp.array(np.asarray(a, dtype=np.float32)[None], dtype=dtype, device=self.device)
                  for a, dtype in ((data.xpos, wp.vec3), (data.xmat.reshape(-1, 3, 3), wp.mat33),
                                   (data.geom_xpos, wp.vec3), (data.geom_xmat.reshape(-1, 3, 3), wp.mat33))]
        result = self.sample_warp(*arrays)
        wp.synchronize_device(self.device)  # Uploaded pose arrays must outlive the kernel.
        return result


def attach_patch_sites(spec, prefix=""):
    """Non-colliding placement guides persisted in MJCF / mjlab scene specs."""
    for finger in FINGERS:
        name = prefix + "flexitac_" + finger.name
        if any(site.name == name for site in spec.sites):
            continue
        body = spec.body(prefix + finger.link)
        body.add_site(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=finger.center, size=(0.012, 0.00015, 0.032),
                      rgba=(*finger.color, 0.45), group=3)
