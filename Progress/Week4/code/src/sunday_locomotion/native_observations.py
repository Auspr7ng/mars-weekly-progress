"""Native actor history, shared by independent audits and realtime playback."""
import math
import numpy as np
import mujoco


class NativeHistory:
    def __init__(self,m,default,names,action_count,revision=4):
        self.m=m;self.default=default;self.names=names;self.action_count=action_count
        self.qadr=[m.joint('robot/'+n).qposadr[0] for n in names]
        self.vadr=[m.joint('robot/'+n).dofadr[0] for n in names]
        self.root=m.body('robot/base_link').id;self.history=None
        self.period=1.4 if revision>=5 else 1.;self.offset=.075 if revision>=5 else 0.
        self.velocity_feedback=revision>=10
        self.phase_gate=revision>=13;self.gate=0.
        self.variable_cadence=revision>=25;self.gait_phase=np.float32(self.offset)
    def next_phase(self,command):
        from sunday_locomotion.cadence import native_period
        return np.float32((self.gait_phase+np.float32(.02)/native_period(command))%np.float32(1.))
    def push(self,d,command,last,step,target_yaw=0.):
        rotation=d.xmat[self.root].reshape(3,3)
        # XBODY is the link origin/frame. BODY uses the inertial COM frame;
        # its translational velocity differs by omega x COM offset.
        velocity=np.zeros(6);mujoco.mj_objectVelocity(self.m,d,mujoco.mjtObj.mjOBJ_XBODY,self.root,velocity,1)
        yaw=math.atan2(rotation[1,0],rotation[0,0])
        phase_signal=np.array([math.sin((step*.02/self.period+self.offset)*2*math.pi),math.cos((step*.02/self.period+self.offset)*2*math.pi)])
        if self.variable_cadence:
            if step>0:self.gait_phase=self.next_phase(command)
            phase_signal=np.array([math.sin(float(self.gait_phase)*2*math.pi),math.cos(float(self.gait_phase)*2*math.pi)])
        if self.phase_gate:
            if step>0:
                desired=np.clip((np.linalg.norm(command[:2])+.375*abs(command[2]))/.03,0,1)
                self.gate+=(1-math.exp(-.02/.18))*(desired-self.gate)
            phase_signal*=self.gate
        terms=[velocity[:3]*.25,rotation.T@np.array([0,0,-1.]),command,
               d.qpos[self.qadr]-self.default,d.qvel[self.vadr]*.05,last,
               phase_signal,
               np.array([math.sin(target_yaw-yaw),math.cos(target_yaw-yaw)])]
        if self.velocity_feedback:terms.append(velocity[3:])
        terms=[np.asarray(t,dtype=np.float32) for t in terms]
        if self.history is None:self.history=[np.tile(t,(5,1)) for t in terms]
        else:self.history=[np.concatenate([h[1:],t[None]]) for h,t in zip(self.history,terms)]
        return np.concatenate([h.ravel() for h in self.history])
