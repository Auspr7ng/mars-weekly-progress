"""A continuous phase oscillator speeds up small steps at larger commands."""
import numpy as np


def torch_period(command):
    import torch
    forward=((command[:,0]-.06)/.09).clamp(0,1)
    reverse=((-command[:,0]-.005)/.025).clamp(0,1)
    yaw=((command[:,2].abs()-.12)/.28).clamp(0,1)
    return 1.4-.4*torch.maximum(torch.maximum(forward,reverse),yaw)


def native_period(command):
    c=np.asarray(command,dtype=np.float32)
    f=np.clip((c[0]-np.float32(.06))/np.float32(.09),0,1)
    b=np.clip((-c[0]-np.float32(.005))/np.float32(.025),0,1)
    y=np.clip((abs(c[2])-np.float32(.12))/np.float32(.28),0,1)
    return np.float32(np.float32(1.4)-np.float32(.4)*max(f,b,y))
