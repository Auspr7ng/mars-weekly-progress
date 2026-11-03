"""Apply the reviewed mass budget without changing geometry or actuator limits."""
import bootstrap
from bootstrap import WEEK
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    model = WEEK / 'model/sunday_a1_locomotion.xml'
    profile_path = WEEK / 'config/dynamics_profile.json'
    old = json.loads(profile_path.read_text())
    audit_path = WEEK / 'research/hardware_metadata/2026-11-01/fusion_verified_comparison.json'
    audit = json.loads(audit_path.read_text())
    if old.get('revision') == 19:
        raise RuntimeError('Metadata dynamics already applied; do not scale a second time')
    assert digest(model) == audit['model_sha256'], 'Unexpected source model'
    backup = WEEK / 'model/archive/pre_metadata_r18'
    backup.mkdir(parents=True, exist_ok=False)
    for relative in ['model/sunday_a1_locomotion.xml', 'model/home.json', 'model/arm_home.json',
                     'config/dynamics_profile.json', 'config/simulation_assumptions.json']:
        shutil.copy2(WEEK / relative, backup / (WEEK / relative).name)
    total = old['total_estimated_mass_kg']
    root_mass = .7
    factor = (total-root_mass) / (total-old['torso']['mass_kg'])
    # Assembly CAD axes -> canonical +X forward, +Y left. Rotate both COM and tensor.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    com = rotation @ np.array(audit['CAD_base_COM_in_URDF_reference_m'])
    inertia = rotation @ np.array(audit['CAD_base_uniform_mass_0p7kg_inertia_ESTIMATE_kgm2']) @ rotation.T
    full = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    tree = ET.parse(model)
    root = tree.getroot()
    for body in root.findall('.//body'):
        inertial = body.find('inertial')
        if inertial is None:
            continue
        if body.get('name') == 'base_link':
            inertial.attrib.clear()
            inertial.set('mass', str(root_mass))
            inertial.set('pos', ' '.join(map(str, com)))
            inertial.set('fullinertia', ' '.join(map(str, full)))
        else:
            inertial.set('mass', str(float(inertial.get('mass'))*factor))
            for attr in ('fullinertia', 'diaginertia'):
                if attr in inertial.attrib:
                    inertial.set(attr, ' '.join(str(float(x)*factor) for x in inertial.get(attr).split()))
    ET.indent(root)
    candidate = model.with_name('sunday_a1_metadata_candidate.xml')
    tree.write(candidate, encoding='utf-8', xml_declaration=True)
    compiled = mujoco.MjModel.from_xml_path(str(candidate))
    assert abs(compiled.body_mass.sum()-total) < 1e-8
    assert np.all(compiled.body_inertia[compiled.body_mass > 0] > 0)
    assert np.all(compiled.body_inertia.sum(axis=1)[:, None] + 1e-10 >= 2*compiled.body_inertia)
    before = mujoco.MjModel.from_xml_path(str(model))
    for attr in ['jnt_range', 'jnt_axis', 'jnt_pos', 'body_pos', 'body_quat', 'geom_pos', 'geom_size',
                 'actuator_forcerange', 'jnt_actfrcrange', 'dof_armature']:
        assert np.array_equal(getattr(before, attr), getattr(compiled, attr)), attr
    candidate.replace(model)
    new = dict(old)
    new.update(revision=19, status='APPROXIMATE MEASURED TOTAL; METADATA ROOT MASS; CAD-SHAPE INERTIA ESTIMATES',
               mass_mode='root_0p7kg_plus_nonroot_normalized_to_2p14kg',
               global_mass_inertia_scale=old['global_mass_inertia_scale']*factor,
               superseded_model_sha256=digest(backup/'sunday_a1_locomotion.xml'),
               metadata_audit_sha256=digest(audit_path), nonroot_rescale_factor=factor,
               source_backup=str(backup.relative_to(WEEK)))
    new['torso'] = dict(mass_kg=root_mass, com_canonical_m=com.tolist(),
                        fullinertia_canonical_kg_m2=full.tolist(), uncertainty_fraction=.3,
                        source='Word base_link approximately700g; reviewed Fusion geometric COM and uniformly scaled shape inertia; not measured inertial properties')
    new['segments'] = {name: {**value, 'mass_kg': value['mass_kg']*factor,
                              'method': 'Original modified-link relative distribution, normalized to remaining2.14kg budget; inertia scaled with mass'}
                       for name, value in old['segments'].items()}
    new['caveats'] = ['Approximate3kg minus160g battery exactly once gives2.84kg.',
                      'Root700g is approximate metadata; COM/inertia assume uniform CAD geometry.',
                      'Remaining link masses remain estimates, not steel CAD masses or measured per-link masses.',
                      'Actuator caps and supply assumption retained pending real calibration; old results do not validate the revised dynamics.']
    new['model_sha256'] = digest(model)
    profile_path.write_text(json.dumps(new, indent=2)+'\n')
    assumptions_path = WEEK/'config/simulation_assumptions.json'
    assumptions = json.loads(assumptions_path.read_text())
    assumptions.update(status=new['status'], source=new['torso']['source'], revision=19,
                       global_mass_inertia_scale=new['global_mass_inertia_scale'], added_root_mass_kg=root_mass,
                       root_com_canonical_m=com.tolist(), root_fullinertia_canonical_kg_m2=full.tolist(),
                       nonroot_rescale_factor=factor, model_sha256=new['model_sha256'])
    assumptions_path.write_text(json.dumps(assumptions, indent=2)+'\n')
    report = dict(total_mass_kg=float(compiled.body_mass.sum()), root_mass_kg=root_mass,
                  nonroot_mass_kg=float(compiled.body_mass.sum()-root_mass), root_com_canonical_m=com.tolist(),
                  root_fullinertia_canonical_kg_m2=full.tolist(), nonroot_rescale_factor=factor,
                  model_sha256=new['model_sha256'], backup=str(backup.relative_to(WEEK)),
                  geometry_axes_limits_actuators_unchanged=True, inertia_positive_definite=True,
                  joint_order=[compiled.joint(i).name for i in range(1, compiled.njnt)], simulation_only=True)
    (WEEK/'results/validation/metadata_dynamics_r19.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
