"""Extra residual authority at fast commands; physical limits remain separate."""
import numpy as np


def expanded_joints(names,revision=23):
    if revision>=24:return [n.endswith(('ankle_roll','ankle_pitch')) for n in names]
    return [any(n.endswith(s) for s in ('ankle_roll','ankle_pitch','thigh_yaw',
            'shoulder_roll','shoulder_pitch','elbow_yaw','elbow_pitch')) for n in names]


def native_scales(base,names,command,revision):
    if revision<23 or revision>=25:return base
    # The old forward/yaw envelope remains exactly unchanged.
    level=max(np.clip((abs(command[0])-.08)/.07,0.,1.),np.clip((abs(command[2])-.12)/.28,0.,1.))
    if revision>=24:level=np.clip((abs(command[0])-.08)/.07,0.,1.)
    return base*(1.+.5*level*np.asarray(expanded_joints(names,revision)))
