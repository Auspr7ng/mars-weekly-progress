"""Package the validated native IK candidate for a separate residual-RL task."""
import bootstrap
from bootstrap import WEEK
import json,math
import numpy as np

source=WEEK/'research/reference_candidate/gait.npz'
ref=np.load(source)
home=json.loads((WEEK/'model/gait_home.json').read_text())
names=ref['joint_names'].tolist()
home['joint_pos']=dict(zip(names,ref['joint_pos'][10].tolist()))
home['root_height']=float(ref['root_height'])
pitch=float(ref['pelvis_pitch'])
home['root_quat']=[math.cos(pitch/2),0,math.sin(pitch/2),0]
home['reference_period']=float(ref['period'])
home['reference_speed']=float(ref['speed'])
home['walking_revision']='unused_reference_candidate'
home['source']='IK gait candidate with separate native PD validation; not a learned policy'
(WEEK/'model/reference_home.json').write_text(json.dumps(home,indent=2),encoding='utf-8')
(WEEK/'model/reference_gait.npz').write_bytes(source.read_bytes())
print(json.dumps({k:home[k] for k in ('root_height','root_quat','reference_period','reference_speed')}))
