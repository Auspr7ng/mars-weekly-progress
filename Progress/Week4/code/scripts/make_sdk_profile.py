"""Normalize the modified shape/inertia distribution to the confirmed mass budget."""
import bootstrap
from bootstrap import WEEK
import hashlib,json
import xml.etree.ElementTree as ET
import numpy as np


def main():
    root=WEEK/'research/vibe_sdk/viberobotics/assets/mujoco'
    old=ET.parse(root/'SundayA1/robot.xml').getroot()
    def by_joint(xml,joint):
        return next(b.find('inertial') for b in xml.findall('.//body') if b.find(f"joint[@name='{joint}']") is not None)
    source_root=old.find('worldbody/body/inertial')
    x=np.fromstring(source_root.get('fullinertia'),sep=' ')
    I=np.array([[x[0],x[3],x[4]],[x[3],x[1],x[5]],[x[4],x[5],x[2]]])
    # SDK +Y forward / +X right to canonical +X forward / +Y left.
    R=np.array([[0.,1,0],[-1,0,0],[0,0,1]])
    inertia=R@I@R.T
    com=R@np.fromstring(source_root.get('pos'),sep=' ')
    variants=[]
    for path in root.rglob('robot.xml'):
        doc=ET.parse(path).getroot()
        variants.append({'name':path.parent.name,'total_explicit_mass_kg':sum(float(i.get('mass',0)) for i in doc.findall('.//inertial')),
                         'root_inertial':doc.find('worldbody/body/inertial').attrib,
                         'joint_count':len(doc.findall('.//body/joint'))})
    confirmation=json.loads((WEEK/'config/hardware_confirmation.json').read_text(encoding='utf-8'))
    modified=ET.parse(WEEK/'model/source/sunday_a1.urdf').getroot()
    raw_total=sum(float(link.find('inertial/mass').get('value')) for link in modified.findall('link'))
    target=confirmation['simulation_total_mass_kg']
    scale=target/(raw_total+float(source_root.get('mass')))
    rows={link.get('name'):{'mass_kg':float(link.find('inertial/mass').get('value'))*scale,
          'source_mass_kg':float(link.find('inertial/mass').get('value')),
          'source_variant':'Week3 modified Sunday A1 URDF','source_joint':None,
          'method':'preserve geometry, COM and inertial radius; mass and inertia scaled together to measured-total budget'}
          for link in modified.findall('link') if link.get('name')!='base_link'}
    profile={'status':'APPROXIMATE MEASURED TOTAL; ESTIMATED PER-LINK DISTRIBUTION',
             'mass_mode':'normalize_to_measured_total',
             'global_mass_inertia_scale':scale,
             'measured_total_with_battery_kg':confirmation['assembled_mass_approx_kg'],
             'battery_removed_kg':confirmation['battery_mass_kg'],
             'repository':'https://github.com/viberobotics/vibe_robotics_sdk','ref':'4c29bee4683e97d4580bee308d97d3c6c8c2418e',
             'torso':{'mass_kg':float(source_root.get('mass'))*scale,'com_canonical_m':com.tolist(),
                      'fullinertia_canonical_kg_m2':(np.array([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]])*scale).tolist(),
                      'source':'SundayA1/robot.xml root template, included in the normalized total; not an additional mass on top of measured robot',
                      'uncertainty_fraction':.3},
             'segments':rows,'modified_raw_declared_mass_kg':raw_total,
             'total_estimated_mass_kg':target,
             'hardware_clarification':'Jaeeun measured approximately 3 kg including 160 g battery; use 2.84 kg with wall supply. ST3215 at 7.2 V. The URDF 100 limits are placeholders.',
             'actuators':{'rotary_effort_cap_Nm':1.6,'leg_position_kp_Nm_per_rad':50.,'rotary_kd_Nm_s_per_rad':1.,
                 'rotary_speed_cap_rad_s':4.,'wrist_effort_cap_Nm':.8,'head_effort_cap_Nm':.5,'finger_effort_cap_N':15.,
                 'evidence':'1.6 Nm is an SDK simulation motor cap in SundayA1_Mujoco_V1.2, not a verified continuous rating. 4.9 Nm exists in a different leg-only MPC asset and is NOT applied blindly.',
                 'model':'ST3215','supply_voltage_V':7.2,
                 'missing':'Exact servo suffix/gear variant, measured torque-speed curve, continuous-duty behavior and modified-finger transmission remain unmeasured.'},
             'source_files':{str(p.relative_to(WEEK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('robot.xml')},
             'caveats':['Stock models have 12/21/23/25 joints; modified model has 27. Do not load stock policies directly.',
                        'The total is anchored to an approximate measurement; per-link distribution is still an estimate.',
                        'Battery is excluded from the total budget once; its unknown position is not fabricated.',
                        'Previous 11.714/12.889 kg models and their checkpoints are superseded; start a fresh policy.']}
    (WEEK/'config/dynamics_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    (WEEK/'research/vibe_sdk_model_comparison.json').write_text(json.dumps(variants,indent=2),encoding='utf-8')
    print(json.dumps({'total_estimated_mass_kg':profile['total_estimated_mass_kg'],'torso':profile['torso'],'variants':[{'name':v['name'],'mass':v['total_explicit_mass_kg'],'joints':v['joint_count']} for v in variants]},indent=2))


if __name__=='__main__':main()
